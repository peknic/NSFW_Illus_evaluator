"""
图像预处理工具
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def preprocess_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    加载并预处理图片
    
    Args:
        image_path: 图片路径
        target_size: 目标尺寸 (宽, 高)，如果为None则根据图像尺寸自动决定
        
    Returns:
        np.ndarray: 预处理后的BGR图像
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 转换为RGB确保一致性
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 如果没有指定目标尺寸，根据原始尺寸自动决定
    if target_size is None:
        h, w = image_rgb.shape[:2]
        
        # 动态确定目标尺寸
        # 对于NudeNet检测，使用更高分辨率以保留细节
        # 对于小图像，保持原尺寸或稍微放大
        # 对于大图像，适当缩小但保持足够细节
        max_dimension = max(h, w)
        
        if max_dimension <= 1024:
            # 小图像：保持原尺寸
            target_size = (w, h)
        elif max_dimension <= 2048:
            # 中等图像：缩放到1600
            scale = 1600 / max_dimension
            target_size = (int(w * scale), int(h * scale))
        else:
            # 大图像：缩放到2048（提高分辨率以保留细节）
            scale = 2048 / max_dimension
            target_size = (int(w * scale), int(h * scale))
        
        print(f"[DEBUG] 自动确定目标尺寸: {target_size[0]}x{target_size[1]} (原始: {w}x{h})")
    
    # 调整尺寸（保持宽高比）
    resized_image = resize_image(image_rgb, target_size)
    
    # 转换为BGR返回（OpenCV格式）
    return cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    调整图片尺寸，保持宽高比
    
    Args:
        image: 输入图像 (H, W, C)
        target_size: 目标尺寸 (宽, 高)
        
    Returns:
        np.ndarray: 调整后的图像
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    
    # 计算新尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 调整尺寸
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 填充到目标尺寸
    if new_w != target_w or new_h != target_h:
        padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return padded
    
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    标准化图像到 [0, 1] 范围
    
    Args:
        image: 输入图像
        
    Returns:
        np.ndarray: 标准化后的图像
    """
    return image.astype(np.float32) / 255.0


def convert_to_pil(image: np.ndarray) -> Image.Image:
    """
    numpy数组转换为PIL图像
    
    Args:
        image: BGR格式的numpy数组
        
    Returns:
        PIL.Image: RGB格式的PIL图像
    """
    # 转换BGR到RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def convert_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    PIL图像转换为numpy数组
    
    Args:
        pil_image: PIL图像
        
    Returns:
        np.ndarray: BGR格式的numpy数组
    """
    # 转换为numpy数组 (RGB)
    rgb_array = np.array(pil_image)
    
    # 转换RGB到BGR
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)