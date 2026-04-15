"""
姿态/动作色气评分模块 v2.0 —— SDPose-Wholebody 版
替换原 DWPose ONNX 方案，改用 SDPose（Stable Diffusion U-Net backbone）
专为 anime/stylized/OOD 域优化，133 wholebody keypoints

推理流程（top-down）：
  YOLO11-x 人体检测 → SDPose Pipeline 关键点估计 → 规则打分

依赖：见 requirements.txt 新增片段
模型文件：见模块末尾 MODEL_DOWNLOAD 说明字符串
"""

import sys
import os
import math
import logging
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

# ============================================================
# SDPose-OOD 代码路径注入
# 须将 GitHub repo t-s-liang/SDPose-OOD clone 至 models/SDPose-OOD/
# ============================================================
# 尝试多种路径解析方式
_PROJECT_ROOT = Path(__file__).parent.parent  # src -> 项目根目录

# 方法1：相对于项目根目录的绝对路径
_SDPOSE_REPO_ABS = _PROJECT_ROOT / "models" / "SDPose-OOD"
# 方法2：相对于当前工作目录的相对路径
_SDPOSE_REPO_REL = Path("models/SDPose-OOD")

# 优先使用绝对路径
_SDPOSE_REPO = _SDPOSE_REPO_ABS if _SDPOSE_REPO_ABS.exists() else _SDPOSE_REPO_REL

if _SDPOSE_REPO.exists():
    if str(_SDPOSE_REPO) not in sys.path:
        sys.path.insert(0, str(_SDPOSE_REPO))
        logger.info(f"[SDPose] 已添加路径到 sys.path: {_SDPOSE_REPO} (绝对路径: {_SDPOSE_REPO_ABS.exists()})")
    else:
        logger.info(f"[SDPose] 路径已在 sys.path 中: {_SDPOSE_REPO}")
else:
    logger.warning(f"[SDPose] 路径不存在: abs={_SDPOSE_REPO_ABS}, rel={_SDPOSE_REPO_REL}")

# ============================================================
# COCO Wholebody 133点 关键点索引
# 0-16   : body (COCO17)
# 17-22  : foot (6点)
# 23-90  : face (68点)
# 91-111 : right hand (21点)
# 112-132: left hand (21点)
# ============================================================
KP_BODY = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}
# 手部（右手首点=91，左手首点=112）
KP_RHAND_BASE = 91
KP_LHAND_BASE = 112
# 面部（68点，首点=23）
KP_FACE_BASE = 23
# 脸部关键子集（下颌轮廓、嘴唇、眼睛）
FACE_JAW_IDX   = list(range(23, 40))   # 下颌线 17点
FACE_LIP_IDX   = list(range(60, 68)) + list(range(48, 60))  # 嘴唇 20点
FACE_LEYE_IDX  = list(range(42, 48))   # 左眼
FACE_REYE_IDX  = list(range(36, 42))   # 右眼

# 置信度阈值（降低以提高检测率）
CONF_THR = 0.2

# SDPose 模型输入尺寸（W×H）
SDPOSE_W, SDPOSE_H = 768, 1024

# ============================================================
# 全局单例
# ============================================================
_sdpose_engine = None   # SDPoseInference 实例（来自 SDPose_gradio.py）
_yolo_model    = None   # ultralytics YOLO 实例


# ============================================================
# 工具函数
# ============================================================

def _pt(kps: np.ndarray, name: str) -> Optional[Tuple[float, float]]:
    """取 body 关键点坐标；置信度不足返回 None。kps: (133,3) x,y,conf"""
    idx = KP_BODY.get(name)
    if idx is None or idx >= len(kps):
        return None
    x, y, c = kps[idx]
    return (float(x), float(y)) if c >= CONF_THR else None


def _pt_raw(kps: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
    """取任意索引关键点；置信度不足返回 None。"""
    if idx >= len(kps):
        return None
    x, y, c = kps[idx]
    return (float(x), float(y)) if c >= CONF_THR else None


def _dist(a: Tuple, b: Tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a: Tuple, b: Tuple) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _angle_three(a: Tuple, b: Tuple, c: Tuple) -> float:
    """b 为顶点的夹角（度，0~180）"""
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    norm = math.hypot(*ba) * math.hypot(*bc)
    if norm < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/norm))))


def _horiz_tilt(a: Tuple, b: Tuple) -> float:
    """向量 a->b 偏离水平线的角度（度，0=完全水平）"""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return abs(math.degrees(math.atan2(dy, dx + 1e-9)))


def _vec_angle_vertical(a: Tuple, b: Tuple) -> float:
    """向量 a->b 与垂直方向的夹角（度，0=竖直）"""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return abs(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9)))


# ============================================================
# SDPose 模型加载（懒加载单例）
# ============================================================

def _load_sdpose() -> Optional[Any]:
    """
    加载 SDPose-Wholebody pipeline。
    依赖 models/SDPose-OOD（GitHub repo）及 models/SDPose-Wholebody（HF 权重）。
    若任一不存在则返回 None（静默降级）。
    """
    global _sdpose_engine

    if _sdpose_engine is not None:
        return _sdpose_engine

    repo_dir    = _PROJECT_ROOT / "models" / "SDPose-OOD"
    weight_dir  = _PROJECT_ROOT / "models" / "SDPose-Wholebody"
    
    logger.info(f"[SDPose] 开始加载模型，工作目录: {os.getcwd()}")
    logger.info(f"[SDPose] 项目根目录: {_PROJECT_ROOT}")
    logger.info(f"[SDPose] SDPose-OOD 绝对路径: {repo_dir}")
    logger.info(f"[SDPose] SDPose-OOD 存在: {repo_dir.exists()}")
    logger.info(f"[SDPose] SDPose-Wholebody 权重路径: {weight_dir}")
    logger.info(f"[SDPose] SDPose-Wholebody 存在: {weight_dir.exists()}")

    if not repo_dir.exists():
        logger.warning("[SDPose] models/SDPose-OOD 未找到，姿态模块降级为零分。")
        return None
    if not weight_dir.exists():
        logger.warning("[SDPose] models/SDPose-Wholebody 未找到，姿态模块降级为零分。")
        return None

    try:
        # 确保 SDPose-OOD 路径在 sys.path 最前面
        repo_path = str(repo_dir)
        if repo_path in sys.path:
            sys.path.remove(repo_path)
        sys.path.insert(0, repo_path)
        logger.info(f"[SDPose] 确保路径在最前: {repo_path}")
        
        # 清除可能缓存的模块
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('gradio_app'):
                logger.debug(f"[SDPose] 移除缓存模块: {mod_name}")
                sys.modules.pop(mod_name, None)
        
        # 检查 gradio_app 模块可导入性
        import importlib.util
        spec = importlib.util.find_spec("gradio_app")
        logger.info(f"[SDPose] gradio_app 模块查找: {spec}")
        if spec:
            logger.info(f"[SDPose] gradio_app 位置: {spec.origin}")
        
        # 动态导入 SDPose 推理类（避免 sys.path 和模块缓存问题）
        sdpose_gradio_path = repo_dir / "gradio_app" / "SDPose_gradio.py"
        logger.info(f"[SDPose] 动态导入文件: {sdpose_gradio_path}")
        
        if not sdpose_gradio_path.exists():
            raise ImportError(f"SDPose_gradio.py 不存在: {sdpose_gradio_path}")
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("SDPose_gradio", str(sdpose_gradio_path))
        if spec is None:
            raise ImportError(f"无法创建模块 spec: {sdpose_gradio_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # 注册到 sys.modules
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"加载模块失败: {e}") from e
        
        # 从模块中获取类
        if not hasattr(module, "SDPoseInference"):
            raise ImportError(f"模块中未找到 SDPoseInference 类")
        
        SDPoseInference = module.SDPoseInference
        logger.info(f"[SDPose] 动态导入成功: {SDPoseInference}")

        engine = SDPoseInference()
        ok = engine.load_model(
            model_path=str(weight_dir),
            keypoint_scheme="wholebody",
            device="auto",
        )
        if not ok:
            logger.warning("[SDPose] 模型加载失败，降级为零分。")
            return None

        # 显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _sdpose_engine = engine
        logger.info("[SDPose] 模型加载成功（wholebody 133点）")
        return _sdpose_engine

    except Exception as e:
        logger.error(f"[SDPose] 加载异常：{type(e).__name__}: {e}")
        logger.error(f"[SDPose] sys.path 前10个:")
        for i, p in enumerate(sys.path[:10]):
            logger.error(f"  {i}: {p}")
        # 检查 gradio_app 模块是否存在
        try:
            import importlib
            spec = importlib.util.find_spec("gradio_app")
            logger.error(f"[SDPose] gradio_app 模块查找: {spec}")
            if spec:
                logger.error(f"[SDPose] gradio_app 位置: {spec.origin}")
        except Exception as ex:
            logger.error(f"[SDPose] 模块查找错误: {ex}")
        logger.warning("[SDPose] 降级为零分。")
        return None


def _load_yolo() -> Optional[Any]:
    """加载 YOLO11-x 人体检测器（懒加载单例）。"""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    yolo_pt = _PROJECT_ROOT / "models" / "SDPose-Wholebody" / "yolo11x.pt"
    if not yolo_pt.exists():
        # 尝试备用路径
        yolo_pt = _PROJECT_ROOT / "models" / "yolo11x.pt"
    if not yolo_pt.exists():
        logger.warning("[YOLO] yolo11x.pt 未找到，将使用全图推理。")
        return None

    try:
        from ultralytics import YOLO  # type: ignore
        _yolo_model = YOLO(str(yolo_pt))
        logger.info(f"[YOLO] 加载成功：{yolo_pt}")
        return _yolo_model
    except Exception as e:
        logger.warning(f"[YOLO] 加载失败：{e}")
        return None


# ============================================================
# 推理：YOLO 检测 + SDPose 估计
# ============================================================

def _detect_and_estimate(pil_img: Image.Image) -> Optional[np.ndarray]:
    """
    top-down 推理主流程。
    返回 (133, 3) ndarray [x, y, conf]（原图像素坐标），失败返回 None。
    """
    engine = _load_sdpose()
    if engine is None:
        return None

    img_rgb = np.array(pil_img.convert("RGB"))

    try:
        # 首先尝试使用YOLO检测（提高检测率）
        yolo_path = None
        if Path("models/SDPose-Wholebody/yolo11x.pt").exists():
            yolo_path = str(Path("models/SDPose-Wholebody/yolo11x.pt"))
        elif Path("models/yolo11x.pt").exists():
            yolo_path = str(Path("models/yolo11x.pt"))
        
        # 第一次尝试：使用YOLO检测
        _, all_kps, all_scores, _, _ = engine.predict_image(
            img_rgb,
            enable_yolo=True,
            yolo_model_path=yolo_path,
            score_threshold=CONF_THR,
            restore_coords=True,
            flip_test=False,
            process_all_persons=True,  # 检测所有人物，选择置信度最高的
        )

        # 如果YOLO检测失败或无结果，尝试禁用YOLO进行全图推理
        if all_kps is None or len(all_kps) == 0:
            logger.info("[SDPose] YOLO检测失败，尝试全图推理...")
            _, all_kps, all_scores, _, _ = engine.predict_image(
                img_rgb,
                enable_yolo=False,  # 禁用YOLO，全图推理
                score_threshold=CONF_THR,
                restore_coords=True,
                flip_test=False,
                process_all_persons=True,
            )
        
        if all_kps is None or len(all_kps) == 0:
            logger.warning("[SDPose] 全图推理也未检测到人体关键点")
            return None

        # 选择平均置信度最高的人物
        best_idx = 0
        best_avg = 0.0
        for i in range(len(all_kps)):
            if len(all_scores[i]) > 0:
                avg_conf = np.mean(all_scores[i])
                if avg_conf > best_avg:
                    best_avg = avg_conf
                    best_idx = i
        
        kps    = all_kps[best_idx]    # (K, 2)
        scores = all_scores[best_idx] # (K,)
        logger.info(f"[SDPose] 选择第{best_idx+1}个人物，平均置信度: {best_avg:.3f}")

        # 合并为 (K, 3)
        n = len(kps)
        kps_conf = np.zeros((n, 3), dtype=np.float32)
        kps_conf[:, :2] = kps
        kps_conf[:, 2]  = scores

        # 确保至少 133 点
        if n < 133:
            pad = np.zeros((133 - n, 3), dtype=np.float32)
            kps_conf = np.vstack([kps_conf, pad])

        return kps_conf[:133]

    except Exception as e:
        logger.warning(f"[SDPose] 推理异常：{e}")
        return None


# ============================================================
# 规则打分引擎（133点，7大维度）
# ============================================================

def _score_leg_spread(kps: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    腿部张开角度（0~5分）。
    髋中点为顶点，左右膝方向夹角 >100° 满分，降低阈值提高灵敏度。
    """
    lh = _pt(kps, "left_hip");  rh = _pt(kps, "right_hip")
    lk = _pt(kps, "left_knee"); rk = _pt(kps, "right_knee")
    if not all([lh, rh, lk, rk]):
        return 0.0, None
    mid = _midpoint(lh, rh)
    angle = _angle_three(lk, mid, rk)
    if angle > 100:
        return 5.0, f"腿部大幅张开（{angle:.0f}°）"
    elif angle > 80:
        return 3.0, f"腿部明显张开（{angle:.0f}°）"
    elif angle > 60:
        return 1.5, f"腿部轻度张开（{angle:.0f}°）"
    elif angle > 50:
        return 0.5, f"腿部微张（{angle:.0f}°）"
    return 0.0, None


def _score_back_arch(kps: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    背弓/腰胯弧度（0~4分）。
    降低阈值提高灵敏度，更容易检测轻微背弓。
    """
    ls = _pt(kps, "left_shoulder"); rs = _pt(kps, "right_shoulder")
    lh = _pt(kps, "left_hip");     rh = _pt(kps, "right_hip")
    la = _pt(kps, "left_ankle");   ra = _pt(kps, "right_ankle")
    if not all([ls, rs, lh, rh]):
        return 0.0, None
    mid_s = _midpoint(ls, rs)
    mid_h = _midpoint(lh, rh)
    if la and ra:
        mid_a = _midpoint(la, ra)
        lv = (mid_a[0]-mid_s[0], mid_a[1]-mid_s[1])
        ll = math.hypot(*lv)
        if ll < 1e-6:
            return 0.0, None
        hv = (mid_h[0]-mid_s[0], mid_h[1]-mid_s[1])
        cross = abs(lv[0]*hv[1] - lv[1]*hv[0])
        body_h = _dist(mid_s, mid_a)
        ratio = cross / (ll * body_h + 1e-6)
        if ratio > 0.10:
            return 4.0, "明显背弓/腰胯前推"
        elif ratio > 0.05:
            return 2.0, "轻度背弓姿态"
        elif ratio > 0.02:
            return 0.8, "微幅背弓"
    else:
        tilt = _vec_angle_vertical(mid_s, mid_h)
        if tilt > 15:
            return 2.0, f"躯干倾斜（{tilt:.0f}°）"
        elif tilt > 8:
            return 0.8, f"微幅躯干倾斜（{tilt:.0f}°）"
    return 0.0, None


def _score_hip_shoulder_tilt(kps: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    臀/肩扭转评分（0~4分）。
    降低阈值提高灵敏度，更容易检测轻微倾斜。
    """
    ls = _pt(kps, "left_shoulder"); rs = _pt(kps, "right_shoulder")
    lh = _pt(kps, "left_hip");     rh = _pt(kps, "right_hip")
    score = 0.0; reasons = []
    if ls and rs:
        tilt = abs(90 - _horiz_tilt(ls, rs))  # 0=肩水平
        if tilt > 12:
            score += 2.0; reasons.append(f"肩部明显倾斜（{tilt:.0f}°）")
        elif tilt > 6:
            score += 1.0
        elif tilt > 3:
            score += 0.3
    if lh and rh:
        tilt = abs(90 - _horiz_tilt(lh, rh))
        if tilt > 12:
            score += 2.0; reasons.append(f"臀部扭胯（{tilt:.0f}°）")
        elif tilt > 6:
            score += 1.0
        elif tilt > 3:
            score += 0.3
    return min(score, 4.0), ("、".join(reasons) if reasons else None)


def _score_hand_position(kps: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    手部位置评分（0~5分）。
    提高灵敏度，手腕/手指尖靠近腰胯或胸部加分；增加靠近胸部的检测。
    """
    lw = _pt(kps, "left_wrist");  rw = _pt(kps, "right_wrist")
    lh = _pt(kps, "left_hip");    rh = _pt(kps, "right_hip")
    ls = _pt(kps, "left_shoulder"); rs = _pt(kps, "right_shoulder")
    if not (lh and rh):
        return 0.0, None
    mid_hip = _midpoint(lh, rh)
    mid_shoulder = _midpoint(ls, rs) if ls and rs else mid_hip
    ref = _dist(mid_shoulder, mid_hip) or 100.0
    score = 0.0; reasons = []

    # 手腕距髋（提高灵敏度）
    for side, wrist in [("左手腕", lw), ("右手腕", rw)]:
        if wrist is None:
            continue
        r = _dist(wrist, mid_hip) / ref
        if r < 0.6:
            score += 2.0; reasons.append(f"{side}置于腰胯附近")
        elif r < 1.2:
            score += 0.8

    # 手腕距胸部（新增检测）
    if ls and rs:
        mid_chest = _midpoint(ls, rs)
        for side, wrist in [("左手腕", lw), ("右手腕", rw)]:
            if wrist is None:
                continue
            r = _dist(wrist, mid_chest) / ref
            if r < 0.8:
                score += 1.5; reasons.append(f"{side}靠近胸部")
                break  # 避免重复加分

    # 手指尖（利用133点，右手 fingertip indices: 95,99,103,107,111）
    r_tips = [95, 99, 103, 107, 111]   # 右手各指指尖
    l_tips = [116, 120, 124, 128, 132] # 左手各指指尖
    suggestive_tips = 0
    for idx in r_tips + l_tips:
        tip = _pt_raw(kps, idx)
        if tip and _dist(tip, mid_hip) / ref < 0.8:
            suggestive_tips += 1
    if suggestive_tips >= 2:
        score += 1.5; reasons.append(f"多根手指靠近腰胯（{suggestive_tips}根）")
    elif suggestive_tips >= 1:
        score += 0.5

    return min(score, 5.0), ("、".join(reasons) if reasons else None)


def _score_face_tilt(kps: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    面部倾斜/仰头评分（0~3分）——利用 68点面部关键点。
    降低阈值提高灵敏度，更容易检测轻微头部倾斜。
    """
    nose = _pt(kps, "nose")
    # 下颌中心：face 第8点（正面下颌最低点），索引 = 23 + 8 = 31
    jaw_center = _pt_raw(kps, 31)
    # 双眼中点用于辅助
    leye = _pt(kps, "left_eye"); reye = _pt(kps, "right_eye")

    if nose and jaw_center:
        # 下颌→鼻尖向量与竖直方向夹角
        tilt = _vec_angle_vertical(jaw_center, nose)
        if tilt > 25:
            return 3.0, f"头部大幅倾斜/仰视（{tilt:.0f}°）"
        elif tilt > 12:
            return 1.5, f"头部轻度倾斜（{tilt:.0f}°）"
        elif tilt > 6:
            return 0.5, f"头部微倾（{tilt:.0f}°）"
    elif leye and reye and nose:
        # 退而用双眼连线相对鼻子位置判断头部旋转
        eye_mid = _midpoint(leye, reye)
        tilt = _vec_angle_vertical(eye_mid, nose)
        if tilt > 20:
            return 2.0, f"面部大幅倾斜（{tilt:.0f}°）"
        elif tilt > 10:
            return 1.0, f"面部轻度倾斜（{tilt:.0f}°）"
        elif tilt > 5:
            return 0.3, f"面部微倾（{tilt:.0f}°）"
    return 0.0, None


def _score_s_curve(kps: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    S 型曲线评分（0~4分）。
    降低阈值提高灵敏度，更容易检测轻微S型曲线。
    """
    ls = _pt(kps, "left_shoulder"); rs = _pt(kps, "right_shoulder")
    lh = _pt(kps, "left_hip");     rh = _pt(kps, "right_hip")
    la = _pt(kps, "left_ankle");   ra = _pt(kps, "right_ankle")
    if not all([ls, rs, lh, rh]):
        return 0.0, None
    mid_s = _midpoint(ls, rs)
    mid_h = _midpoint(lh, rh)
    if la and ra:
        mid_a = _midpoint(la, ra)
        # 三点 x 坐标的 zigzag 量
        xs = mid_s[0]; xh = mid_h[0]; xa = mid_a[0]
        body_span = _dist(mid_s, mid_a) or 100.0
        # S型：肩→髋→踝 x 方向先偏一侧再偏另一侧
        d1 = xh - xs  # 髋相对肩的横移
        d2 = xa - xh  # 踝相对髋的横移
        if d1 * d2 < 0:  # 方向相反 = S型
            magnitude = (abs(d1) + abs(d2)) / body_span
            if magnitude > 0.15:
                return 4.0, f"躯干呈S型曲线（幅度{magnitude:.2f}）"
            elif magnitude > 0.08:
                return 2.0, f"轻度S型曲线（幅度{magnitude:.2f}）"
            elif magnitude > 0.04:
                return 0.8, f"微幅S型曲线"
    return 0.0, None


def _score_dynamic_pose(kps: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    动态感评分（0~3分）。
    提高灵敏度，膝弯曲、手臂伸展、重心侧移综合评估。
    """
    score = 0.0; reasons = []
    lh = _pt(kps, "left_hip");  rh = _pt(kps, "right_hip")
    lk = _pt(kps, "left_knee"); rk = _pt(kps, "right_knee")
    la = _pt(kps, "left_ankle"); ra = _pt(kps, "right_ankle")
    ls = _pt(kps, "left_shoulder"); rs = _pt(kps, "right_shoulder")
    lw = _pt(kps, "left_wrist"); rw = _pt(kps, "right_wrist")

    # 膝盖弯曲（降低阈值）
    for side, (hip, knee, ankle) in [("左", (lh, lk, la)), ("右", (rh, rk, ra))]:
        if hip and knee and ankle:
            a = _angle_three(hip, knee, ankle)
            if a < 110:
                score += 0.8; reasons.append(f"{side}腿弯曲（膝角{a:.0f}°）")
            elif a < 130:
                score += 0.3; reasons.append(f"{side}腿微弯（膝角{a:.0f}°）")

    # 手臂伸展（降低阈值）
    if ls and rs and lh and rh:
        body_c = _midpoint(_midpoint(ls, rs), _midpoint(lh, rh))
        scale  = _dist(_midpoint(ls, rs), _midpoint(lh, rh)) or 100.0
        for w in [lw, rw]:
            if w and _dist(w, body_c) / scale > 1.0:
                score += 0.6; reasons.append("手臂伸展"); break
            elif w and _dist(w, body_c) / scale > 0.8:
                score += 0.2; reasons.append("手臂微伸"); break

    return min(score, 3.0), ("、".join(reasons) if reasons else None)


def _score_supine(kps: np.ndarray) -> Tuple[float, Optional[str]]:
    """
    躺卧姿态评分（0~3分）。
    降低阈值提高灵敏度，更容易检测半躺姿势。
    """
    ls = _pt(kps, "left_shoulder"); rs = _pt(kps, "right_shoulder")
    lh = _pt(kps, "left_hip");     rh = _pt(kps, "right_hip")
    if not all([ls, rs, lh, rh]):
        return 0.0, None
    mid_s = _midpoint(ls, rs); mid_h = _midpoint(lh, rh)
    dx = abs(mid_s[0] - mid_h[0]); dy = abs(mid_s[1] - mid_h[1])
    horiz = dx / (dx + dy + 1e-6)
    if horiz > 0.65:
        return 3.0, "仰卧/躺姿（躯干近水平）"
    elif horiz > 0.45:
        return 1.5, "半躺姿态"
    elif horiz > 0.30:
        return 0.5, "微倾躺姿"
    return 0.0, None


# ============================================================
# 骨骼可视化（热力色，红=高分，蓝=低分）
# ============================================================

# 身体骨架连接（COCO17 索引）
_BODY_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]
# 手部连接（相对于各手首点的偏移）
_HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]


def _heat_color(score: float, max_score: float = 30.0) -> Tuple[int, int, int]:
    """蓝→黄→红渐变，返回 BGR"""
    r = min(1.0, score / max_score)
    if r < 0.5:
        t = r * 2
        return (int(255*(1-t)), int(255*t), 0)
    else:
        t = (r-0.5)*2
        return (0, int(255*(1-t)), int(255*t))


def draw_skeleton_overlay(
    pil_img: Image.Image,
    kps: Optional[np.ndarray],
    pose_score: float,
) -> np.ndarray:
    """
    绘制骨骼叠加图（BGR ndarray）。
    身体骨架 + 双手骨架 + 面部点云，颜色随 pose_score 热力渐变。
    """
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    if kps is None:
        return img_bgr

    color = _heat_color(pose_score)
    H, W  = img_bgr.shape[:2]

    def _valid(pt):
        return pt is not None and 0 <= pt[0] < W and 0 <= pt[1] < H

    # 身体骨架
    for a, b in _BODY_PAIRS:
        pa = _pt_raw(kps, a); pb = _pt_raw(kps, b)
        if _valid(pa) and _valid(pb):
            cv2.line(img_bgr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2, cv2.LINE_AA)

    # 身体关键点
    for i in range(17):
        pt = _pt_raw(kps, i)
        if _valid(pt):
            cv2.circle(img_bgr, (int(pt[0]), int(pt[1])), 5, color, -1, cv2.LINE_AA)
            cv2.circle(img_bgr, (int(pt[0]), int(pt[1])), 5, (255,255,255), 1, cv2.LINE_AA)

    # 手部骨架（右手=91，左手=112）
    hand_color = (0, 100, 255)  # 橙色（BGR）
    for base in [KP_RHAND_BASE, KP_LHAND_BASE]:
        for a, b in _HAND_EDGES:
            pa = _pt_raw(kps, base+a); pb = _pt_raw(kps, base+b)
            if _valid(pa) and _valid(pb):
                cv2.line(img_bgr, (int(pa[0]),int(pa[1])), (int(pb[0]),int(pb[1])), hand_color, 1, cv2.LINE_AA)

    # 面部关键点（白色小点）
    for i in range(KP_FACE_BASE, KP_FACE_BASE + 68):
        pt = _pt_raw(kps, i)
        if _valid(pt):
            cv2.circle(img_bgr, (int(pt[0]), int(pt[1])), 2, (220, 220, 220), -1)

    # 分数标注
    cv2.putText(img_bgr, f"Pose Score: {pose_score:.1f}/20",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    return img_bgr


# ============================================================
# 主入口
# ============================================================

def analyze_pose(pil_img: Image.Image) -> Dict[str, Any]:
    """
    SDPose-Wholebody 姿态色气分析主函数。

    Args:
        pil_img: PIL Image (RGB)

    Returns:
        {
          "keypoints"          : np.ndarray (133,3) 或 None,
          "pose_score"         : float (0~20),
          "suggestive_reasons" : list[str],
          "skeleton_overlay"   : np.ndarray (BGR),
        }
    """
    # 1. 推理
    kps: Optional[np.ndarray] = _detect_and_estimate(pil_img)

    # 2. 规则打分
    total  = 0.0
    reasons: List[str] = []

    if kps is not None:
        for scorer in [
            _score_leg_spread,
            _score_back_arch,
            _score_hip_shoulder_tilt,
            _score_hand_position,
            _score_face_tilt,
            _score_s_curve,
            _score_dynamic_pose,
            _score_supine,
        ]:
            try:
                s, r = scorer(kps)
                total += s
                if r:
                    reasons.append(r)
            except Exception as e:
                logger.debug(f"[scorer] {scorer.__name__} 异常: {e}")

    total = float(max(0.0, min(30.0, total)))  # 提高上限至30分

    # 3. 骨骼叠加图
    overlay = draw_skeleton_overlay(pil_img, kps, total)

    # 4. 显存清理
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    return {
        "keypoints":          kps,
        "pose_score":         total,
        "suggestive_reasons": reasons,
        "skeleton_overlay":   overlay,
    }


# ============================================================
# MODEL_DOWNLOAD — 模型文件下载指引（供人工阅读）
# ============================================================
MODEL_DOWNLOAD = """
【第一步：SDPose-OOD 代码库（GitHub）】
  仓库：https://github.com/t-s-liang/SDPose-OOD
  操作：git clone https://github.com/t-s-liang/SDPose-OOD.git models/SDPose-OOD
  说明：含 models/HeatmapHead.py、models/ModifiedUNet.py、
        pipelines/SDPose_D_Pipeline.py、gradio_app/SDPose_gradio.py 等推理代码。

【第二步：SDPose-Wholebody 权重（HuggingFace）】
  仓库：https://huggingface.co/teemosliang/SDPose-Wholebody
  目标路径：models/SDPose-Wholebody/
  需下载以下文件（保持目录结构）：
    decoder/decoder.safetensors          （~约数十 MB，heatmap head 权重）
    unet/config.json
    unet/diffusion_pytorch_model.safetensors  （~约3 GB，SD U-Net 主干）
    vae/config.json
    vae/diffusion_pytorch_model.safetensors   （~约800 MB）
    text_encoder/config.json
    text_encoder/model.safetensors            （~约1.2 GB）
    tokenizer/merges.txt
    tokenizer/special_tokens_map.json
    tokenizer/tokenizer_config.json
    tokenizer/vocab.json
    scheduler/scheduler_config.json
    yolo11x.pt                                （~约130 MB，人体检测器）

  推荐下载方式（需 huggingface-hub）：
    pip install huggingface-hub
    huggingface-cli download teemosliang/SDPose-Wholebody --local-dir models/SDPose-Wholebody

【目录结构验证】
  img_evaluator/
  ├── models/
  │   ├── SDPose-OOD/          ← GitHub repo clone
  │   │   ├── models/
  │   │   │   ├── HeatmapHead.py
  │   │   │   └── ModifiedUNet.py
  │   │   ├── pipelines/
  │   │   │   └── SDPose_D_Pipeline.py
  │   │   └── gradio_app/
  │   │       └── SDPose_gradio.py
  │   └── SDPose-Wholebody/    ← HF 权重
  │       ├── decoder/decoder.safetensors
  │       ├── unet/
  │       ├── vae/
  │       ├── text_encoder/
  │       ├── tokenizer/
  │       ├── scheduler/
  │       └── yolo11x.pt
"""
