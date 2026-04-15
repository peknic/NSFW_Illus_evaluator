"""
Gradio Web UI v2.1
新增姿态色气评分模块（第二阶段优化版）
  总分融合：裸露检测 40% + 姿态评分 30%（合计70%，归一化至100分）
  姿态评分上限提高至30分，阈值优化，检测率提升
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image

from src.inference import create_detector
from src.pose_module import analyze_pose
from utils.visualization import create_score_visualization, combine_images


# ============================================================
# 全局单例
# ============================================================

_detector = None


def get_detector():
    """获取裸露检测器实例（单例懒加载）"""
    global _detector
    if _detector is None:
        _detector = create_detector()
    return _detector


# ============================================================
# 工具：图像输入统一处理
# ============================================================

def _resolve_image(input_image):
    """
    统一处理 Gradio 传入的图像格式。
    返回 (image_path: str, temp_path: str|None)
    """
    if isinstance(input_image, str):
        return input_image, None
    if isinstance(input_image, np.ndarray):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = f.name
        cv2.imwrite(tmp, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
        time.sleep(0.05)
        return tmp, tmp
    return None, None


def _cleanup_temp(temp_path):
    """安全删除临时文件"""
    if temp_path and os.path.exists(temp_path):
        try:
            os.unlink(temp_path)
        except Exception:
            pass


# ============================================================
# 裸露检测（第一阶段，原有逻辑）
# ============================================================

def analyze_image(input_image):
    """
    裸露检测分析。
    返回 (nudity_score, score_chart, annotated_img, combined_img, text)
    """
    detector = get_detector()
    image_path, temp_path = _resolve_image(input_image)

    if image_path is None:
        return 0, None, None, None, "[错误] 图片格式不支持，请上传 jpg/png/webp"

    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            _cleanup_temp(temp_path)
            return 0, None, None, None, f"[错误] 无法读取图片: {image_path}"

        result = detector.analyze(image_path)
        _cleanup_temp(temp_path)

        score_chart = create_score_visualization(
            result["overall_score"], result["part_scores"]
        )
        output_image = result["annotated_image"]

        output_image_rgb = (
            cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            if output_image is not None else None
        )
        score_chart_rgb = (
            cv2.cvtColor(score_chart, cv2.COLOR_BGR2RGB)
            if score_chart is not None else None
        )
        if output_image is not None and score_chart is not None:
            combined = combine_images(original_image, output_image, score_chart)
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        else:
            combined_rgb = None

        part_names_cn = {
            "face": "脸部", "breast": "胸部", "buttocks": "臀部",
            "thighs": "大腿", "waist": "腰部", "genitalia": "私密部位",
        }
        part_text = "[部位分析]:\n"
        for part, score in result["part_scores"].items():
            cn = part_names_cn.get(part, part)
            bars = "■" * (score // 10)
            part_text += f"{cn} {bars} {score}/100\n"

        out_text = (
            f"[裸露色气值]: {result['overall_score']}/100\n\n"
            + part_text + "\n"
            + f"[评价]: {result['comment']}\n\n"
            + f"[建议]: {result['suggestions']}"
        )
        return result["overall_score"], score_chart_rgb, output_image_rgb, combined_rgb, out_text

    except Exception as e:
        _cleanup_temp(temp_path)
        return 0, None, None, None, f"[错误] 分析出错: {e}"


# ============================================================
# 姿态分析（第二阶段）
# ============================================================

def analyze_pose_tab(input_image):
    """
    姿态色气分析。
    返回 (skeleton_img, pose_score_text, reasons_text)
    """
    image_path, temp_path = _resolve_image(input_image)
    if image_path is None:
        return None, "0 / 20", "[错误] 图片格式不支持"

    try:
        pil_img = Image.open(image_path).convert("RGB")
        _cleanup_temp(temp_path)

        pose_result = analyze_pose(pil_img)
        pose_score = pose_result["pose_score"]
        reasons    = pose_result["suggestive_reasons"]
        skeleton   = pose_result["skeleton_overlay"]

        skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
        score_text   = f"{pose_score:.1f} / 30"

        if reasons:
            reasons_text = "检测到以下暗示性姿态要素：\n" + "\n".join(
                f"  • {r}" for r in reasons
            )
        else:
            reasons_text = "未检测到明显暗示性姿态要素。\n（若未安装 SDPose 模型，请参阅 MODEL_DOWNLOAD.md）"

        return skeleton_rgb, score_text, reasons_text

    except Exception as e:
        _cleanup_temp(temp_path)
        return None, "0 / 30", f"[错误] 姿态分析出错: {e}"


# ============================================================
# 综合分析（裸露 40% + 姿态 30%，归一化至100分）
# ============================================================

def analyze_combined(input_image):
    """
    综合分析入口。
      裸露分(0~100) * 0.4 + (姿态分/30*100) * 0.3，合计70分制，归一化至100分
    返回值顺序须与 _combined_outputs 完全一致。
    """
    image_path, temp_path = _resolve_image(input_image)
    if image_path is None:
        err = "[错误] 图片格式不支持"
        return 0, None, None, None, err, None, "0 / 30", err, 0

    try:
        detector = get_detector()
        original_image = cv2.imread(image_path)
        if original_image is None:
            _cleanup_temp(temp_path)
            return 0, None, None, None, "[错误] 无法读取图片", None, "0 / 30", "", 0

        # 裸露检测
        result = detector.analyze(image_path)

        # 姿态分析（复用同一 image_path，temp 已读入不删）
        pil_img = Image.open(image_path).convert("RGB")
        _cleanup_temp(temp_path)

        pose_result = analyze_pose(pil_img)
        pose_score  = pose_result["pose_score"]
        reasons     = pose_result["suggestive_reasons"]
        skeleton    = pose_result["skeleton_overlay"]

        # 综合得分
        nudity_score   = result["overall_score"]
        combined_score = int((nudity_score * 0.4 + pose_score * 1.0) * 100 / 70)
        combined_score = max(0, min(100, combined_score))

        # 裸露可视化
        score_chart  = create_score_visualization(result["overall_score"], result["part_scores"])
        output_image = result["annotated_image"]

        output_image_rgb = (
            cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB) if output_image is not None else None
        )
        score_chart_rgb = (
            cv2.cvtColor(score_chart, cv2.COLOR_BGR2RGB) if score_chart is not None else None
        )
        if output_image is not None and score_chart is not None:
            combined_vis = combine_images(original_image, output_image, score_chart)
            combined_vis_rgb = cv2.cvtColor(combined_vis, cv2.COLOR_BGR2RGB)
        else:
            combined_vis_rgb = None

        # 裸露文字
        part_names_cn = {
            "face": "脸部", "breast": "胸部", "buttocks": "臀部",
            "thighs": "大腿", "waist": "腰部", "genitalia": "私密部位",
        }
        part_text = "[部位分析]:\n"
        for part, score in result["part_scores"].items():
            cn = part_names_cn.get(part, part)
            bars = "■" * (score // 10)
            part_text += f"{cn} {bars} {score}/100\n"

        nudity_text = (
            f"[裸露色气值]: {nudity_score}/100\n\n"
            + part_text + "\n"
            + f"[评价]: {result['comment']}\n\n"
            + f"[建议]: {result['suggestions']}"
        )

        # 姿态可视化
        skeleton_rgb    = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
        pose_score_text = f"{pose_score:.1f} / 20"
        reasons_text    = (
            "检测到以下暗示性姿态要素：\n" + "\n".join(f"  • {r}" for r in reasons)
            if reasons else "未检测到明显暗示性姿态要素。"
        )

        return (
            nudity_score,       # nudity_score_out
            score_chart_rgb,    # chart_out
            output_image_rgb,   # annotated_out
            combined_vis_rgb,   # combined_img_out
            nudity_text,        # nudity_text_out
            skeleton_rgb,       # skeleton_out
            pose_score_text,    # pose_score_out
            reasons_text,       # pose_reasons_out
            combined_score,     # combined_score_out
        )

    except Exception as e:
        _cleanup_temp(temp_path)
        err = f"[错误] 综合分析出错: {e}"
        return 0, None, None, None, err, None, "0 / 20", err, 0


# ============================================================
# Gradio UI
# ============================================================

def create_ui():
    """创建 Gradio 界面 v2.0（含姿态分析 Tab）"""

    with gr.Blocks(title="插画色气值检测器 v2.0") as demo:

        gr.Markdown("""
        # 插画色气值检测器 v2.0
        **专门针对动漫/日漫/插画风格的 NSFW 检测工具**
         第一阶段：NudeNet v3 裸露检测 | 第二阶段：SDPose 姿态色气评分
         综合得分 = 裸露检测 x40% + 姿态评分 x20%（合计60%，归一化至100分）
        """)

        with gr.Row():
            # 左列：上传 + 按钮 + 总分
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="上传插画图片",
                    type="numpy",
                    sources=["upload"],
                    interactive=True,
                )
                btn_combined = gr.Button("综合分析（推荐）", variant="primary", size="lg")
                btn_nudity   = gr.Button("仅裸露检测", variant="secondary")
                btn_pose     = gr.Button("仅姿态分析", variant="secondary")

                with gr.Accordion("高级设置", open=False):
                    gr.Markdown("""
                    当前模型配置:
                    - NudeNet v3 部位检测（ONNX）
                    - Falconsai NSFW 分类器（ViT）
                     - SDPose-Wholebody 关键点（Stable Diffusion backbone）
                     - 姿态评分：纯规则，无需训练
                    """)

                combined_score_out = gr.Number(
                    label="综合色气值（满分100）",
                    value=0,
                    interactive=False,
                )

            # 右列：结果 Tabs
            with gr.Column(scale=2):

                with gr.Tabs():

                    with gr.TabItem("综合分析"):
                        combined_img_out = gr.Image(
                            label="裸露检测汇总图",
                            interactive=False,
                        )

                    with gr.TabItem("部位检测"):
                        annotated_out = gr.Image(
                            label="部位标注可视化",
                            interactive=False,
                        )

                    with gr.TabItem("分数图表"):
                        chart_out = gr.Image(
                            label="裸露分数分布",
                            interactive=False,
                        )

                    with gr.TabItem("姿态分析"):
                        skeleton_out = gr.Image(
                            label="骨骼叠加图（红=高色气 蓝=低色气）",
                            interactive=False,
                        )
                        pose_score_out = gr.Textbox(
                            label="姿态色气得分（满分20）",
                            value="0 / 20",
                            interactive=False,
                        )
                        pose_reasons_out = gr.Textbox(
                            label="姿态评分理由",
                            lines=6,
                            interactive=False,
                        )

                nudity_text_out = gr.Textbox(
                    label="裸露检测报告",
                    lines=10,
                    interactive=False,
                )
                nudity_score_out = gr.Number(
                    label="裸露色气值（满分100）",
                    value=0,
                    interactive=False,
                )

        gr.Markdown("""
        ---
        ### 使用说明
        1. 上传插画/动漫图片
        2. 点击「综合分析」获取完整报告，或分别点击单项按钮
        3. 查看「姿态分析」Tab 中的骨骼叠加图与评分理由

        ### 评分说明
        | 维度 | 权重 | 满分 |
        |------|------|------|
        | 裸露检测（NudeNet+NSFW分类器）| 40% | 100 |
        | 姿态色气（SDPose 规则评分）| 20% | 20换算100 |

        姿态评分细则：腿部张开角度 / 背弓腰胯弧度 / 肩臀倾斜 / 手部位置 / 肩髋比例 / 动态感 / 躺卧姿态

        ### 注意
        - SDPose-Wholebody 模型须自行下载，详见 MODEL_DOWNLOAD.md
        - 未安装 SDPose 时姿态分恒为 0，裸露分仍正常工作
        """)

        # 事件绑定
        _combined_outputs = [
            nudity_score_out,
            chart_out,
            annotated_out,
            combined_img_out,
            nudity_text_out,
            skeleton_out,
            pose_score_out,
            pose_reasons_out,
            combined_score_out,
        ]

        btn_combined.click(fn=analyze_combined, inputs=image_input, outputs=_combined_outputs)
        image_input.upload(fn=analyze_combined, inputs=image_input, outputs=_combined_outputs)

        btn_nudity.click(
            fn=analyze_image,
            inputs=image_input,
            outputs=[nudity_score_out, chart_out, annotated_out, combined_img_out, nudity_text_out],
        )
        btn_pose.click(
            fn=analyze_pose_tab,
            inputs=image_input,
            outputs=[skeleton_out, pose_score_out, pose_reasons_out],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, debug=True)
