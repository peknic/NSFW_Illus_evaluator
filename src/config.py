"""
配置参数
"""

# 模型路径配置
MODEL_PATHS = {
    'nudenet': 'models/nudenet_v3',  # 本地缓存路径
    'nsfw_detector': 'models/nsfw_detector',
}

# 部位权重（影响最终色气值计算）
PART_WEIGHTS = {
    'face': 0.10,      # 脸部 (降低)
    'breast': 0.25,    # 胸部 (保持不变)
    'buttocks': 0.23,  # 臀部 (增加)
    'thighs': 0.18,    # 大腿 (增加)
    'waist': 0.05,     # 腰部 (降低)
    'genitalia': 0.19, # 私密部位 (增加)
}

# 分数计算参数
SCORE_PARAMS = {
    'nsfw_weight': 0.6,      # NSFW分类器权重
    'parts_weight': 0.4,     # 部位检测权重
    'exposure_threshold': 0.3,  # 暴露度阈值
    'confidence_threshold': 0.25, # 检测置信度阈值（进一步降低以适应动漫，从0.3降到0.25）
    'multi_person_confidence_threshold': 0.15, # 多人物场景置信度阈值（从0.2降到0.15）
    'anime_adjustment_factor': 1.2,  # 动漫分数调整因子
}

# 部位特定阈值调整（进一步降低阈值以提高检测率，特别是对于动漫图片）
PART_THRESHOLD_ADJUSTMENTS = {
    'face': 0.4,      # 脸部阈值乘以0.4（降低60%）
    'genitalia': 0.5, # 私密部位阈值乘以0.5（降低50%）
    'breast': 0.3,    # 胸部阈值乘以0.3（降低70%）
    'buttocks': 0.5,  # 臀部阈值乘以0.5（降低50%）
    'thighs': 0.6,    # 大腿阈值乘以0.6（降低40%）
    'waist': 0.6,     # 腰部阈值乘以0.6（降低40%）
}

# 评价生成配置
COMMENT_CONFIG = {
    'low_score_range': (0, 30),      # 低分区
    'medium_score_range': (31, 70),  # 中分区
    'high_score_range': (71, 100),   # 高分区
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'part_colors': {
        'face': (255, 200, 0),      # 黄色
        'breast': (255, 100, 100),  # 红色
        'buttocks': (100, 255, 100),# 绿色
        'thighs': (100, 100, 255),  # 蓝色
        'waist': (255, 100, 255),   # 紫色
        'genitalia': (255, 150, 50),# 橙色
    },
    'box_thickness': 2,
    'font_scale': 0.5,
}

# NudeNet v3 标签到标准部位名称的映射
NUDE_NET_MAPPING = {
    # 胸部相关
    'FEMALE_BREAST_EXPOSED': 'breast',
    'FEMALE_BREAST_COVERED': 'breast',
    'MALE_BREAST_EXPOSED': 'breast',
    'MALE_BREAST_COVERED': 'breast',
    
    # 臀部相关
    'BUTTOCKS_EXPOSED': 'buttocks',
    'BUTTOCKS_COVERED': 'buttocks',
    
    # 腹部/腰部
    'BELLY_EXPOSED': 'waist',
    'BELLY_COVERED': 'waist',
    
    # 脸部
    'FACE_FEMALE': 'face',
    'FACE_MALE': 'face',
    
    # 私密部位
    'FEMALE_GENITALIA_EXPOSED': 'genitalia',
    'FEMALE_GENITALIA_COVERED': 'genitalia',
    'MALE_GENITALIA_EXPOSED': 'genitalia',
    'MALE_GENITALIA_COVERED': 'genitalia',
    
    # 大腿
    'THIGHS_EXPOSED': 'thighs',
    'THIGHS_COVERED': 'thighs',
    
    # 其他映射（可选）
    'FEET_EXPOSED': 'feet',
    'FEET_COVERED': 'feet',
}

# NudeNet 配置
NUDENET_CONFIG = {
    'inference_resolution': 640,  # 提高分辨率以改善小目标检测
    'providers': None,           # 使用默认ONNX运行时提供程序
}