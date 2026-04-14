"""
分数计算工具
"""

from typing import Dict, Any
import numpy as np

from src.config import PART_WEIGHTS, SCORE_PARAMS


def calculate_overall_score(nsfw_raw_score: float, parts_data: Dict[str, Any]) -> int:
    """
    计算总体色气值 (0-100分)
    
    Args:
        nsfw_raw_score: NSFW分类器原始分数 (0-1)
        parts_data: 部位检测数据
        
    Returns:
        int: 0-100的整数分数
    """
    # 1. 计算部位暴露度分数
    parts_score = calculate_parts_exposure_score(parts_data)
    
    # 2. 加权融合
    nsfw_weight = SCORE_PARAMS['nsfw_weight']
    parts_weight = SCORE_PARAMS['parts_weight']
    
    combined_score = (nsfw_raw_score * nsfw_weight + parts_score * parts_weight)
    
    # 3. 转换为0-100整数
    overall_score = int(round(combined_score * 100))
    
    # 4. 确保在范围内
    return max(0, min(100, overall_score))


def calculate_parts_exposure_score(parts_data: Dict[str, Any]) -> float:
    """
    计算部位暴露度综合分数
    
    Args:
        parts_data: 部位检测数据
        
    Returns:
        float: 0-1的分数
    """
    if not parts_data:
        return 0.0
    
    total_weight = 0.0
    weighted_sum = 0.0
    
    for part_name, detections in parts_data.items():
        if part_name in PART_WEIGHTS:
            weight = PART_WEIGHTS[part_name]
            
            # 计算该部位的平均置信度（应用阈值过滤）
            if detections:
                # 过滤低置信度检测
                confidence_threshold = SCORE_PARAMS.get('confidence_threshold', 0.3)
                filtered_detections = [
                    d for d in detections 
                    if d.get('confidence', 0.0) >= confidence_threshold
                ]
                
                if filtered_detections:
                    confidences = [d.get('confidence', 0.0) for d in filtered_detections]
                    avg_confidence = np.mean(confidences)
                    
                    # 检查是否暴露（如果检测到exposed标志）
                    exposed_flags = [d.get('exposed', True) for d in filtered_detections]
                    exposed_ratio = np.mean([1.0 if exposed else 0.5 for exposed in exposed_flags])
                    
                    # 部位分数 = 置信度 × 暴露比例
                    part_score = avg_confidence * exposed_ratio
                    
                    weighted_sum += part_score * weight
                    total_weight += weight
    
    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def calculate_part_scores(parts_data: Dict[str, Any]) -> Dict[str, int]:
    """
    计算各部位色气值分数
    
    Args:
        parts_data: 部位检测数据
        
    Returns:
        Dict[str, int]: 部位名到0-100分数的映射
    """
    part_scores = {}
    
    for part_name, detections in parts_data.items():
        if detections:
            # 计算平均置信度
            confidences = [d.get('confidence', 0.0) for d in detections]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # 检查暴露度
            exposed_flags = [d.get('exposed', True) for d in detections]
            exposed_bonus = 1.2 if any(exposed_flags) else 1.0
            
            # 计算分数并加入权重影响
            weight = PART_WEIGHTS.get(part_name, 0.1)
            raw_score = avg_confidence * exposed_bonus * (1.0 + weight)
            
            # 转换为0-100，使用非线性映射增强差异
            score = int(round(100 * (raw_score ** 0.7)))
            part_scores[part_name] = max(0, min(100, score))
        else:
            part_scores[part_name] = 0
    
    # 确保所有预期部位都有分数
    expected_parts = ['face', 'breast', 'buttocks', 'thighs', 'waist', 'genitalia']
    for part in expected_parts:
        if part not in part_scores:
            part_scores[part] = 0
    
    return part_scores


def adjust_score_for_illustration_style(original_score: int, style_hints: Dict[str, Any] = None) -> int:
    """
    根据插画风格调整分数
    
    Args:
        original_score: 原始分数
        style_hints: 风格提示（可选）
        
    Returns:
        int: 调整后的分数
    """
    # 基础调整：使用配置中的调整因子
    from src.config import SCORE_PARAMS
    adjustment_factor = SCORE_PARAMS.get('anime_adjustment_factor', 1.2)
    
    # 默认假设是动漫/插画风格，应用调整因子
    adjusted = int(original_score * adjustment_factor)
    
    if style_hints:
        # 如果有额外风格信息，进行微调
        if style_hints.get('hentai_style', False):
            # Hentai风格额外提高分数
            adjusted = int(adjusted * 1.1)
    
    return max(0, min(100, adjusted))