"""
核心推理模块
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Any
import logging

from src.config import MODEL_PATHS, PART_WEIGHTS, SCORE_PARAMS, NUDE_NET_MAPPING, NUDENET_CONFIG, PART_THRESHOLD_ADJUSTMENTS
from src.model_manager import ModelManager
from utils.image_processor import preprocess_image, resize_image
from utils.score_calculator import calculate_overall_score, calculate_part_scores
from utils.comment_generator import generate_comment_with_suggestions
from utils.visualization import visualize_detections

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IllustrationNSFWDetector:
    """插画色气值检测器"""
    
    def __init__(self):
        self.nudenet_model = None
        self.nsfw_model = None
        self.initialized = False
        
    def initialize_models(self):
        """初始化模型"""
        try:
            logger.info("初始化模型...")
            
            # 使用模型管理器
            self.model_manager = ModelManager()
            
            # 初始化 NudeNet v3
            logger.info("初始化 NudeNet v3...")
            from nudenet import NudeDetector
            self.nudenet_model = NudeDetector(
                inference_resolution=NUDENET_CONFIG['inference_resolution'],
                providers=NUDENET_CONFIG['providers']
            )
            logger.info("NudeNet v3 初始化成功")
            
            # 初始化 NSFW 分类器
            logger.info("初始化 NSFW 分类器...")
            
            # 检查模型状态
            status = self.model_manager.check_model_status('nsfw_detector')
            if not status.get('ready', False):
                logger.warning("NSFW 分类器未就绪，尝试下载...")
                result = self.model_manager.download_model('nsfw_detector')
                if not result['success']:
                    logger.error(f"模型下载失败: {result.get('error', '未知错误')}")
                    # 提供手动下载说明
                    if 'instructions' in result:
                        logger.info(result['instructions'])
                    raise RuntimeError("NSFW 分类器初始化失败")
            
            # 加载模型
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            model_path = self.model_manager.models['nsfw_detector']['local_path']
            logger.info(f"从本地加载模型: {model_path}")
            
            try:
                # 先尝试从本地加载
                processor = AutoImageProcessor.from_pretrained(str(model_path))
                model = AutoModelForImageClassification.from_pretrained(str(model_path))
                logger.info("从本地缓存加载模型成功")
            except Exception as e:
                logger.warning(f"本地加载失败，尝试从Hugging Face下载: {e}")
                # 回退到在线下载
                processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
                model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
                logger.info("从Hugging Face下载模型成功")
            
            self.nsfw_model = (processor, model)
            
            self.initialized = True
            logger.info("[OK] 所有模型初始化完成")
            
        except Exception as e:
            logger.error(f"[ERROR] 模型初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 提供更友好的错误信息
            error_msg = f"""
            模型初始化失败！
            可能的原因:
            1. 网络连接问题
            2. Hugging Face模型下载失败
            3. 磁盘空间不足
            
            解决方法:
            1. 检查网络连接
            2. 手动下载模型:
               - 访问: https://huggingface.co/Falconsai/nsfw_image_detection
               - 下载所有文件到: models/nsfw_detector/
               - 必需文件: config.json, pytorch_model.bin, preprocessor_config.json
            3. 运行 python -m src.model_manager 进行模型设置
            """
            logger.error(error_msg)
            raise RuntimeError(f"模型初始化失败: {e}") from e
    
    def detect_body_parts(self, image: np.ndarray) -> Dict[str, Any]:
        """使用 NudeNet 检测身体部位"""
        if not self.initialized:
            self.initialize_models()
        
        # 转换BGR到RGB（numpy数组）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 进行检测 - NudeNet期望numpy数组
        detections = self.nudenet_model.detect(rgb_image)
        
        # 调试：打印原始检测
        print(f"[DEBUG] NudeNet原始检测数量: {len(detections)}")
        for i, det in enumerate(detections):
            part_name = det.get('class', '')
            confidence = det.get('score', 0.0)
            bbox = det.get('box', [])
            exposed = det.get('exposed', 'N/A')
            print(f"[DEBUG] 检测{i}: {part_name}, 置信度={confidence:.4f}, bbox={bbox}, exposed={exposed}")
        
        # 按部位整理检测结果（使用标准部位名称）
        parts_data = {}
        
        # 获取阈值参数
        multi_person_threshold = SCORE_PARAMS.get('multi_person_confidence_threshold', 0.2)
        standard_threshold = SCORE_PARAMS.get('confidence_threshold', 0.3)
        
        # 计算图像总面积
        height, width = image.shape[:2]
        image_area = height * width
        
        for det in detections:
            part_name = det.get('class', '')
            confidence = det.get('score', 0.0)
            bbox = det.get('box', [])
            
            if part_name and len(bbox) >= 4:
                # 映射到标准部位名称
                mapped_name = NUDE_NET_MAPPING.get(part_name, part_name)
                # 只保留映射到标准部位的检测
                if mapped_name in PART_WEIGHTS:
                    # 计算检测框面积和占比
                    x, y, w, h = bbox[:4]
                    detection_area = w * h
                    area_ratio = detection_area / image_area if image_area > 0 else 0
                    
                    # 尺寸感知的置信度阈值
                    # 小检测使用更低的阈值
                    size_adjusted_threshold = self._get_size_adjusted_threshold(
                        mapped_name, confidence, area_ratio, 
                        standard_threshold, multi_person_threshold
                    )
                    
                    # 应用尺寸感知阈值
                    if confidence < size_adjusted_threshold:
                        # 调试：记录被阈值过滤的检测
                        if mapped_name in ['buttocks', 'breast', 'face']:
                            print(f"[DEBUG] 阈值过滤: {part_name}->{mapped_name}, 置信度={confidence:.3f}, 阈值={size_adjusted_threshold:.3f}, 面积占比={area_ratio:.6f}")
                        continue
                    
                    # 根据class名称判断暴露状态（NudeNet对动漫图片的exposed标志可能不准确）
                    # 如果class名称包含"EXPOSED"则视为暴露
                    exposed_from_name = 'EXPOSED' in part_name
                    # 优先使用detection中的exposed标志，如果不存在则使用名称判断
                    exposed = det.get('exposed', exposed_from_name)
                    
                    # 调试：记录通过的检测
                    if mapped_name in ['buttocks', 'breast', 'face']:
                        print(f"[DEBUG] 检测通过: {part_name}->{mapped_name}, 置信度={confidence:.3f}, 面积占比={area_ratio:.6f}, 暴露={exposed}")
                    
                    parts_data.setdefault(mapped_name, []).append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'exposed': exposed,
                        'original_confidence': confidence,  # 保存原始置信度
                        'area_ratio': area_ratio,  # 保存面积占比
                        'is_small': area_ratio < 0.001  # 标记是否为小检测
                    })
        
        # 优化多人物场景检测（包括根据场景类型进行最终过滤）
        filtered_parts_data = self._optimize_multi_person_detections(parts_data, image.shape, standard_threshold)
        
        return filtered_parts_data
    
    def _get_size_adjusted_threshold(self, part_name: str, confidence: float, area_ratio: float, 
                                    standard_threshold: float, multi_person_threshold: float) -> float:
        """
        根据检测尺寸获取调整后的置信度阈值
        
        小检测使用更低的阈值，特别是对于关键部位如胸部、脸部、私密部位
        
        Args:
            part_name: 部位名称
            confidence: 原始置信度
            area_ratio: 检测框面积占比
            standard_threshold: 标准阈值
            multi_person_threshold: 多人物阈值
            
        Returns:
            调整后的阈值
        """
        # 基础阈值（使用更宽松的多人物阈值作为起点）
        base_threshold = min(standard_threshold, multi_person_threshold)
        
        # 获取部位特定的阈值调整因子
        part_adjustment = PART_THRESHOLD_ADJUSTMENTS.get(part_name, 1.0)
        
        # 根据尺寸调整阈值
        if area_ratio < 0.0003:  # 面积占比小于0.03%
            # 非常小的检测，使用极低阈值
            adjusted = 0.03  # 极低阈值（进一步降低）
            
            # 对于关键部位，进一步降低阈值
            if part_name in ['breast', 'face', 'genitalia', 'buttocks']:
                adjusted = 0.02  # 非常低的阈值
                if confidence > 0.01:  # 只要有非零置信度就记录
                    print(f"[DEBUG] 极小{part_name}检测: 置信度={confidence:.3f}, 面积占比={area_ratio:.6f}, 使用阈值={adjusted:.3f}")
            
            return adjusted
        
        elif area_ratio < 0.0005:  # 面积占比小于0.05%
            # 非常小的检测，使用很低阈值
            adjusted = 0.05
            
            # 对于关键部位，进一步降低阈值
            if part_name in ['breast', 'face', 'genitalia', 'buttocks']:
                adjusted = 0.03
                if confidence > 0.03:
                    print(f"[DEBUG] 小{part_name}检测: 置信度={confidence:.3f}, 面积占比={area_ratio:.6f}, 使用阈值={adjusted:.3f}")
            
            return adjusted
        
        elif area_ratio < 0.001:  # 面积占比小于0.1%
            # 小检测，使用较低阈值
            adjusted = base_threshold * 0.4  # 进一步降低
            
            # 应用部位特定调整因子
            adjusted = adjusted * part_adjustment
            
            # 确保阈值不低于最小阈值
            min_threshold = 0.03
            if part_name in ['breast', 'face', 'genitalia', 'buttocks']:
                min_threshold = 0.02
            
            return max(adjusted, min_threshold)
        
        elif area_ratio < 0.002:  # 面积占比小于0.2%
            # 中等偏小检测
            adjusted = base_threshold * 0.6  # 进一步降低
            
            # 应用部位特定调整因子
            adjusted = adjusted * part_adjustment
            
            return max(adjusted, 0.05)
        
        else:
            # 正常尺寸检测，使用基础阈值并应用部位特定调整
            adjusted = base_threshold * part_adjustment
            
            # 确保阈值合理
            return max(adjusted, 0.08)
    
    def _optimize_multi_person_detections(self, parts_data: Dict[str, Any], image_shape: tuple, standard_threshold: float) -> Dict[str, Any]:
        """
        优化多人物场景的检测结果
        
        主要优化点:
        1. 增加每个部位的最大检测数量
        2. 根据空间位置过滤重叠检测
        3. 确保不同人物的相同部位都能被检测到
        4. 根据场景类型进行最终置信度过滤
        
        Args:
            parts_data: 原始检测数据
            image_shape: 图像尺寸 (height, width, channels)
            standard_threshold: 标准置信度阈值
            
        Returns:
            优化后的检测数据
        """
        if not parts_data:
            return {}
        
        height, width = image_shape[:2]
        image_area = height * width
        
        # 判断是否为多人物场景
        is_multi_person = self._is_multi_person_scene(parts_data, image_shape)
        
        # 根据场景类型调整参数
        if is_multi_person:
            nms_threshold = 0.4  # 多人物场景使用放宽的NMS（从0.3提高到0.4）
            final_confidence_threshold = standard_threshold * 0.6  # 多人物场景使用更低的最终阈值（从0.8降到0.6）
            print(f"[DEBUG] 检测到多人物场景，使用放宽NMS阈值: {nms_threshold}, 最终置信度阈值: {final_confidence_threshold}")
        else:
            nms_threshold = 0.5  # 单人物场景使用更宽松的NMS（从0.4提高到0.5）
            final_confidence_threshold = standard_threshold * 0.8  # 单人物场景使用降低的阈值（从1.0降到0.8）
        
        filtered_parts_data = {}
        
        for part_name, detections in parts_data.items():
            if not detections:
                continue
                
            # 按置信度排序
            sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            # 应用NMS过滤重叠检测
            nms_filtered = []
            while sorted_detections:
                # 取置信度最高的检测
                best_det = sorted_detections.pop(0)
                nms_filtered.append(best_det)
                
                # 计算与剩余检测的IoU，移除重叠度高的
                keep_indices = []
                for i, det in enumerate(sorted_detections):
                    iou = self._calculate_iou(best_det['bbox'], det['bbox'])
                    if iou < nms_threshold:
                        keep_indices.append(i)
                
                # 保留非重叠检测
                sorted_detections = [sorted_detections[i] for i in keep_indices]
            
            # 步骤2: 根据场景类型和检测尺寸进行最终置信度过滤
            # 对于多人物场景和小检测，保留更多低置信度检测
            confidence_filtered = []
            
            # 获取部位特定阈值调整因子
            part_adjustment = PART_THRESHOLD_ADJUSTMENTS.get(part_name, 1.0)
            
            for det in nms_filtered:
                # 使用original_confidence如果存在，否则使用confidence
                conf = det.get('original_confidence', det.get('confidence', 0.0))
                area_ratio = det.get('area_ratio', 0.0)
                is_small = det.get('is_small', False)
                
                # 尺寸感知的最终阈值
                # 基础阈值应用部位特定调整
                base_final_threshold = final_confidence_threshold * part_adjustment
                size_adjusted_final_threshold = base_final_threshold
                
                if is_small or area_ratio < 0.001:  # 小检测
                    # 小检测使用更低的阈值
                    size_adjusted_final_threshold = base_final_threshold * 0.5  # 从0.7降到0.5
                    
                    # 对于关键部位的小检测，特别照顾
                    if part_name in ['breast', 'face', 'genitalia', 'buttocks'] and area_ratio < 0.0005:
                        size_adjusted_final_threshold = base_final_threshold * 0.3  # 从0.4降到0.3
                        if conf >= size_adjusted_final_threshold:
                            print(f"[DEBUG] 保留小{part_name}检测: 置信度={conf:.3f}, 阈值={size_adjusted_final_threshold:.3f}, 面积占比={area_ratio:.6f}")
                
                if conf >= size_adjusted_final_threshold:
                    # 确保返回的检测中只包含必要字段
                    filtered_det = {
                        'bbox': det.get('bbox', []),
                        'confidence': det.get('confidence', 0.0),
                        'exposed': det.get('exposed', False)
                    }
                    # 移除额外的字段
                    for key in ['original_confidence', 'area_ratio', 'is_small']:
                        if key in det:
                            # 已经在过滤时使用了，现在移除
                            pass
                    confidence_filtered.append(filtered_det)
            
            # 步骤3: 根据图像尺寸和部位类型设置最大检测数量
            max_detections_per_part = self._get_max_detections_for_part(
                part_name, image_area, len(confidence_filtered), is_multi_person
            )
            
            # 步骤4: 取前N个检测（经过NMS和置信度过滤后）
            filtered_parts_data[part_name] = confidence_filtered[:max_detections_per_part]
        
        return filtered_parts_data
    
    def _is_multi_person_scene(self, parts_data: Dict[str, Any], image_shape: tuple) -> bool:
        """
        判断是否为多人物场景
        
        判断依据:
        1. 检测到的部位总数
        2. 同一部位检测数量
        3. 检测框的空间分布
        
        Args:
            parts_data: 检测数据
            image_shape: 图像尺寸
            
        Returns:
            bool: 是否为多人物场景
        """
        if not parts_data:
            return False
        
        height, width = image_shape[:2]
        
        # 统计检测总数
        total_detections = sum(len(dets) for dets in parts_data.values())
        
        # 检查关键部位（胸部、脸部）的检测数量
        key_parts = ['breast', 'face']
        key_part_detections = 0
        for part in key_parts:
            if part in parts_data:
                key_part_detections += len(parts_data[part])
        
        # 判断逻辑
        # 1. 如果总检测数超过10，可能是多人物
        # 2. 如果胸部检测数超过2，可能是多人物
        # 3. 如果脸部检测数超过1，可能是多人物
        
        if total_detections >= 10:
            return True
        
        if 'breast' in parts_data and len(parts_data['breast']) >= 2:
            # 进一步检查这些检测是否在图像的不同区域
            breast_detections = parts_data['breast']
            if len(breast_detections) >= 2:
                # 计算检测框中心点的距离
                centers = []
                for det in breast_detections[:2]:  # 取前2个最高置信度
                    bbox = det.get('bbox', [])
                    if len(bbox) >= 4:
                        x, y, w, h = bbox[:4]
                        center_x = x + w / 2
                        center_y = y + h / 2
                        centers.append((center_x, center_y))
                
                # 如果有2个检测中心，计算它们之间的距离
                if len(centers) == 2:
                    dx = centers[0][0] - centers[1][0]
                    dy = centers[0][1] - centers[1][1]
                    distance = (dx**2 + dy**2) ** 0.5
                    
                    # 如果距离大于图像宽度的1/4，可能是不同人物
                    if distance > width * 0.25:
                        return True
        
        return False
    
    def _calculate_iou(self, bbox1: list, bbox2: list) -> float:
        """
        计算两个边界框的交并比(IoU)
        
        Args:
            bbox1: [x, y, width, height] 格式
            bbox2: [x, y, width, height] 格式
            
        Returns:
            IoU值 (0-1)
        """
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        # 转换为 [x1, y1, x2, y2] 格式
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _get_max_detections_for_part(self, part_name: str, image_area: float, num_detections: int, is_multi_person: bool = False) -> int:
        """
        根据部位类型和图像尺寸获取最大检测数量
        
        Args:
            part_name: 部位名称
            image_area: 图像面积
            num_detections: 原始检测数量
            is_multi_person: 是否为多人物场景
            
        Returns:
            最大检测数量
        """
        # 基础限制
        base_limits = {
            'face': 6,        # 脸部检测可以多一些
            'breast': 8,      # 胸部检测（考虑多人物）
            'buttocks': 6,    # 臀部
            'thighs': 8,      # 大腿
            'waist': 4,       # 腰部
            'genitalia': 4,   # 私密部位
        }
        
        max_limit = base_limits.get(part_name, 4)
        
        # 多人物场景：增加最大检测数量
        if is_multi_person:
            if part_name in ['breast', 'buttocks', 'thighs']:
                # 关键部位增加更多
                max_limit = int(max_limit * 1.8)
            else:
                max_limit = int(max_limit * 1.5)
            print(f"[DEBUG] 多人物场景: {part_name} 最大检测数增加到 {max_limit}")
        
        # 根据图像面积调整
        # 对于大图像，可以允许更多检测
        if image_area > 1000000:  # 1000x1000像素以上
            max_limit = int(max_limit * 1.5)
        
        # 确保不超过原始检测数量
        return min(max_limit, num_detections)
    
    def predict_nsfw_score(self, image: np.ndarray) -> float:
        """使用 NSFW 分类器预测色气值"""
        if not self.initialized:
            self.initialize_models()
        
        processor, model = self.nsfw_model
        
        # 预处理图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_image, return_tensors="pt")
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # 获取 NSFW 分数（假设第二个类别是 NSFW）
        # Falconsai/nsfw_image_detection 的标签: ['normal', 'nsfw']
        nsfw_score = probabilities[0][1].item()  # nsfw 类别的概率
        
        return nsfw_score
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        分析插画图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: 包含总体色气值、部位分数、评价、可视化图片等
        """
        # 1. 加载并预处理图片
        image = preprocess_image(image_path)
        
        # 2. 并行推理
        nsfw_score = self.predict_nsfw_score(image)
        parts_data = self.detect_body_parts(image)
        
        # 3. 计算分数
        overall_score = calculate_overall_score(nsfw_score, parts_data)
        part_scores = calculate_part_scores(parts_data)
        
        # 4. 根据插画风格调整分数
        # 尝试检测是否为动漫/插画风格
        style_hints = {}
        
        # 简单启发式：如果检测到身体部位但置信度普遍偏低，可能是动漫
        if parts_data:
            total_detections = sum(len(dets) for dets in parts_data.values())
            if total_detections > 0:
                # 计算平均置信度
                all_confidences = []
                for detections in parts_data.values():
                    all_confidences.extend([d.get('confidence', 0.0) for d in detections])
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
                
                # 如果平均置信度较低，可能是动漫图片
                if avg_confidence < 0.5:
                    style_hints['anime_style'] = True
        
        # 应用调整
        from utils.score_calculator import adjust_score_for_illustration_style
        overall_score = adjust_score_for_illustration_style(overall_score, style_hints)
        
        # 4. 生成评价和建议
        comment, suggestions = generate_comment_with_suggestions(overall_score, part_scores)
        
        # 5. 可视化结果
        annotated_image = visualize_detections(image, parts_data)
        
        return {
            'overall_score': overall_score,      # 0-100
            'nsfw_raw_score': nsfw_score,        # 0-1
            'part_scores': part_scores,          # 字典 {部位: 分数}
            'comment': comment,
            'suggestions': suggestions,
            'annotated_image': annotated_image,
            'parts_data': parts_data             # 原始检测数据
        }


def create_detector() -> IllustrationNSFWDetector:
    """创建检测器单例"""
    detector = IllustrationNSFWDetector()
    detector.initialize_models()
    return detector