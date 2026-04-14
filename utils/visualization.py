"""
可视化工具
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from src.config import VISUALIZATION_CONFIG


def _get_font(font_size=20):
    """获取中文字体，优先使用系统字体"""
    try:
        # Windows系统字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
    except:
        pass
    
    # 如果找不到系统字体，使用默认字体（可能不支持中文）
    return ImageFont.load_default()


def _draw_chinese_text(image: np.ndarray, text: str, position: Tuple[int, int], 
                      font_size: int = 20, color: Tuple[int, int, int] = (255, 255, 255),
                      bg_color: Tuple[int, int, int] = None) -> np.ndarray:
    """
    使用PIL在图像上绘制中文文本
    
    Args:
        image: BGR格式的numpy数组
        text: 要绘制的文本
        position: (x, y)位置
        font_size: 字体大小
        color: 文字颜色 (RGB)
        bg_color: 背景颜色 (RGB)，如果为None则透明
        
    Returns:
        np.ndarray: 绘制文本后的图像
    """
    # 将BGR转换为RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    
    # 获取字体
    font = _get_font(font_size)
    
    # 如果需要绘制背景
    if bg_color is not None:
        # 获取文本尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 绘制背景矩形
        x, y = position
        draw.rectangle(
            [(x, y - text_height), (x + text_width, y)],
            fill=bg_color
        )
    
    # 绘制文本
    draw.text(position, text, font=font, fill=color)
    
    # 转换回BGR格式
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result_image


def visualize_detections(image: np.ndarray, parts_data: Dict[str, Any]) -> np.ndarray:
    """
    可视化检测结果
    
    Args:
        image: 原始图像 (BGR格式)
        parts_data: 部位检测数据
        
    Returns:
        np.ndarray: 标注后的图像 (BGR格式)
    """
    # 创建副本
    annotated = image.copy()
    
    # 部位英文名映射（避免中文编码问题）
    part_names_english = {
        'face': 'Face',
        'breast': 'Breast', 
        'buttocks': 'Buttocks',
        'thighs': 'Thighs',
        'waist': 'Waist',
        'genitalia': 'Genital',
        'female_breast': 'Breast',
        'female_genitalia': 'Genital'
    }
    
    # 绘制每个检测到的部位
    for part_name, detections in parts_data.items():
        if not detections:
            continue
            
        # 获取颜色
        color = VISUALIZATION_CONFIG['part_colors'].get(part_name, (255, 255, 255))
        
        for detection in detections:
            bbox = detection.get('bbox', [])
            confidence = detection.get('confidence', 0.0)
            exposed = detection.get('exposed', False)
            
            if len(bbox) >= 4:
                # NudeNet返回 [x, y, width, height] 格式，需要转换为 [x1, y1, x2, y2]
                x, y, w, h = map(int, bbox[:4])
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                
                # 绘制边界框
                cv2.rectangle(
                    annotated,
                    (x1, y1), (x2, y2),
                    color,
                    VISUALIZATION_CONFIG['box_thickness']
                )
                
                # 准备标签文本（使用英文避免编码问题）
                english_name = part_names_english.get(part_name, part_name)
                exposed_text = "Exposed" if exposed else "Covered"
                label = f"{english_name} {confidence:.2f} ({exposed_text})"
                
                # 计算标签背景位置
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           VISUALIZATION_CONFIG['font_scale'], 1)[0]
                
                # 绘制标签背景
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_size[1] - 5),
                    (x1 + label_size[0] + 5, y1),
                    color,
                    -1  # 填充
                )
                
                # 绘制标签文字（英文，不会出现编码问题）
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    VISUALIZATION_CONFIG['font_scale'],
                    (255, 255, 255),  # 白色文字
                    1,
                    cv2.LINE_AA
                )
    
    return annotated


def create_score_visualization(overall_score: int, part_scores: Dict[str, int]) -> np.ndarray:
    """
    创建分数可视化图表
    
    Args:
        overall_score: 总体色气值
        part_scores: 部位分数字典
        
    Returns:
        np.ndarray: 可视化图表图像
    """
    # 创建空白图像
    height = 400
    width = 600
    chart = np.ones((height, width, 3), dtype=np.uint8) * 240  # 浅灰色背景
    
    # 绘制总体分数（英文）
    cv2.putText(chart, f"Overall Score: {overall_score}/100", 
               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 绘制分数条
    bar_width = 400
    bar_height = 20
    bar_x = 50
    bar_y = 80
    
    # 背景条
    cv2.rectangle(chart, (bar_x, bar_y), 
                 (bar_x + bar_width, bar_y + bar_height), 
                 (200, 200, 200), -1)
    
    # 分数条（根据分数使用不同颜色）
    fill_width = int(bar_width * (overall_score / 100))
    
    # 颜色渐变：绿色(低分) -> 黄色(中分) -> 红色(高分)
    if overall_score < 40:
        color = (100, 200, 100)  # 绿色
    elif overall_score < 70:
        color = (100, 200, 200)  # 黄色
    else:
        color = (100, 100, 200)  # 红色
    
    cv2.rectangle(chart, (bar_x, bar_y), 
                 (bar_x + fill_width, bar_y + bar_height), 
                 color, -1)
    
    # 绘制边框
    cv2.rectangle(chart, (bar_x, bar_y), 
                 (bar_x + bar_width, bar_y + bar_height), 
                 (100, 100, 100), 1)
    
    # 绘制部位分数条（使用英文避免编码问题）
    y_offset = 120
    part_names_english = {
        'face': 'Face',
        'breast': 'Breast', 
        'buttocks': 'Buttocks',
        'thighs': 'Thighs',
        'waist': 'Waist',
        'genitalia': 'Genital'
    }
    
    for i, (part, score) in enumerate(part_scores.items()):
        if part in part_names_english:
            english_name = part_names_english[part]
            
            # 部位名称（英文）
            cv2.putText(chart, f"{english_name}:", 
                       (50, y_offset + i * 30 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # 分数文本
            cv2.putText(chart, f"{score}", 
                       (450, y_offset + i * 30 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # 分数条
            part_bar_width = 200
            part_fill_width = int(part_bar_width * (score / 100))
            part_bar_x = 150
            
            # 背景
            cv2.rectangle(chart, 
                         (part_bar_x, y_offset + i * 30 + 10),
                         (part_bar_x + part_bar_width, y_offset + i * 30 + 20),
                         (220, 220, 220), -1)
            
            # 填充（使用部位对应颜色）
            part_color = VISUALIZATION_CONFIG['part_colors'].get(part, (200, 200, 200))
            cv2.rectangle(chart, 
                         (part_bar_x, y_offset + i * 30 + 10),
                         (part_bar_x + part_fill_width, y_offset + i * 30 + 20),
                         part_color, -1)
            
            # 边框
            cv2.rectangle(chart, 
                         (part_bar_x, y_offset + i * 30 + 10),
                         (part_bar_x + part_bar_width, y_offset + i * 30 + 20),
                         (150, 150, 150), 1)
    
    return chart


def combine_images(original: np.ndarray, annotated: np.ndarray, chart: np.ndarray) -> np.ndarray:
    """
    合并原始图像、标注图像和图表
    
    Args:
        original: 原始图像
        annotated: 标注图像
        chart: 分数图表
        
    Returns:
        np.ndarray: 合并后的图像
    """
    # 调整尺寸使宽度一致
    target_width = 400
    
    # 调整原始图像
    h1, w1 = original.shape[:2]
    scale1 = target_width / w1
    new_w1 = target_width
    new_h1 = int(h1 * scale1)
    resized_original = cv2.resize(original, (new_w1, new_h1))
    
    # 调整标注图像
    h2, w2 = annotated.shape[:2]
    scale2 = target_width / w2
    new_w2 = target_width
    new_h2 = int(h2 * scale2)
    resized_annotated = cv2.resize(annotated, (new_w2, new_h2))
    
    # 调整图表尺寸
    h3, w3 = chart.shape[:2]
    scale3 = target_width / w3
    new_w3 = target_width
    new_h3 = int(h3 * scale3)
    resized_chart = cv2.resize(chart, (new_w3, new_h3))
    
    # 计算总高度
    total_height = new_h1 + new_h2 + new_h3 + 20  # 加上间隔
    
    # 创建空白画布
    combined = np.ones((total_height, target_width, 3), dtype=np.uint8) * 255
    
    # 放置图像
    y_offset = 0
    combined[y_offset:y_offset+new_h1, :] = resized_original
    y_offset += new_h1 + 10
    combined[y_offset:y_offset+new_h2, :] = resized_annotated
    y_offset += new_h2 + 10
    combined[y_offset:y_offset+new_h3, :] = resized_chart
    
    # 添加标题（英文）
    titles = ["Original", "Detection", "Score Analysis"]
    y_positions = [15, new_h1 + 15, new_h1 + new_h2 + 15]
    
    for title, y in zip(titles, y_positions):
        cv2.putText(combined, title, 
                   (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return combined