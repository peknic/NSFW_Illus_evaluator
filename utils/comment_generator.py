"""
口语化评价生成器
"""

import random
from typing import Dict, Tuple, List
from src.config import COMMENT_CONFIG


def generate_comment_with_suggestions(overall_score: int, part_scores: Dict[str, int]) -> Tuple[str, str]:
    """
    生成口语化评价和改进建议
    
    Args:
        overall_score: 总体色气值 (0-100)
        part_scores: 部位分数字典
        
    Returns:
        Tuple[str, str]: (评价文字, 改进建议)
    """
    # 确定分数区间
    score_range = _get_score_range(overall_score)
    
    # 生成评价
    comment = _generate_comment(overall_score, part_scores, score_range)
    
    # 生成改进建议
    suggestions = _generate_suggestions(overall_score, part_scores, score_range)
    
    return comment, suggestions


def _get_score_range(score: int) -> str:
    """获取分数区间"""
    low_range = COMMENT_CONFIG['low_score_range']
    medium_range = COMMENT_CONFIG['medium_score_range']
    high_range = COMMENT_CONFIG['high_score_range']
    
    if low_range[0] <= score <= low_range[1]:
        return 'low'
    elif medium_range[0] <= score <= medium_range[1]:
        return 'medium'
    else:
        return 'high'


def _generate_comment(overall_score: int, part_scores: Dict[str, int], score_range: str) -> str:
    """生成评价文字"""
    
    # 获取高分部位
    top_parts = _get_top_parts(part_scores, 2)
    
    # 评价模板
    templates = {
        'low': [
            "这张图才 {score} 分，清纯得像张白纸嘛～ {top_parts_text}基本没啥看头。",
            "哇，{score} 分的纯情插画！{top_parts_text}画得好保守哦。",
            "这张色气值 {score} 分，太清纯了吧！{top_parts_text}都不带一点暗示的。"
        ],
        'medium': [
            "这张有 {score} 分，有点小色气～ {top_parts_text}画得挺有味道！",
            "喔！{score} 分，恰到好处的涩涩感！{top_parts_text}部位处理得不错～",
            "{score} 分这张图，撩人于无形！{top_parts_text}细节把握得很好。"
        ],
        'high': [
            "哇这张色气值直接 {score} 分！{top_parts_text}画得太犯规了～",
            "天哪！{score} 分的神仙图！{top_parts_text}部位简直绝了！",
            "{score} 分！这也太顶了吧！{top_parts_text}画得超会撩人！"
        ]
    }
    
    # 准备top_parts文本
    if top_parts:
        part_names_chinese = {
            'face': '脸部',
            'breast': '胸部', 
            'buttocks': '臀部',
            'thighs': '大腿',
            'waist': '腰部',
            'genitalia': '私密部位'
        }
        
        chinese_parts = [part_names_chinese.get(p, p) for p, _ in top_parts]
        if len(chinese_parts) == 1:
            top_parts_text = f"{chinese_parts[0]}"
        else:
            top_parts_text = f"{chinese_parts[0]}和{chinese_parts[1]}"
    else:
        top_parts_text = "整体构图"
    
    # 选择模板并填充
    template = random.choice(templates[score_range])
    comment = template.format(score=overall_score, top_parts_text=top_parts_text)
    
    # 添加语气词增强语气
    mood_words = {
        'low': [" (清纯可爱~)", " (萌萌哒~)", " (小清新~)"],
        'medium': [" (有点小色气~)", " (撩人于无形~)", " (恰到好处~)"],
        'high': [" (超顶的！)", " (简直绝了！)", " (色气满满！)"]
    }
    comment += random.choice(mood_words[score_range])
    
    return comment


def _generate_suggestions(overall_score: int, part_scores: Dict[str, int], score_range: str) -> str:
    """生成改进建议"""
    
    # 找出分数较低的部位
    low_score_parts = [(part, score) for part, score in part_scores.items() 
                      if score < 50 and part in ['breast', 'buttocks', 'thighs', 'waist']]
    
    # 建议模板
    suggestions_templates = {
        'low': [
            "建议：试试把{part}画得更突出一点，加点光影会更诱人哦～",
            "下次可以加强{part}的曲线，让人更有遐想空间！",
            "改进点：{part}部位可以再大胆一些，别太保守啦！"
        ],
        'medium': [
            "建议：把{part}再压低一点，光影再打狠一点，会更色更顶哦～",
            "下次{part}可以画得更饱满一些，曲线再夸张点！",
            "改进：{part}部位的光泽感可以加强，水润感会更撩人！"
        ],
        'high': [
            "建议：{part}已经超顶了！保持这个水平，再多点微表情就更完美了～",
            "下次试试{part}的不同姿势，突破极限会更刺激！",
            "改进：{part}的质感可以再细化，让视觉效果直接拉满！"
        ]
    }
    
    if low_score_parts:
        # 选择分数最低的部位进行建议
        low_score_parts.sort(key=lambda x: x[1])
        target_part = low_score_parts[0][0]
        
        # 部位中文名映射
        part_names_chinese = {
            'face': '脸部表情',
            'breast': '胸部', 
            'buttocks': '臀部',
            'thighs': '大腿根部',
            'waist': '腰部曲线',
            'genitalia': '私密部位'
        }
        chinese_part = part_names_chinese.get(target_part, target_part)
    else:
        # 如果没有低分部位，选择随机部位
        all_parts = ['胸部', '臀部', '大腿根部', '腰部曲线']
        chinese_part = random.choice(all_parts)
    
    # 选择模板
    template = random.choice(suggestions_templates[score_range])
    suggestion = template.format(part=chinese_part)
    
    # 添加额外建议
    extra_suggestions = [
        "服装可以再透一点，若隐若现最致命！",
        "试试增加一点汗珠或红晕，氛围感直接爆炸！",
        "姿势可以再大胆一点，肢体语言超重要的！",
        "光影对比再强一点，立体感和色气值都会飙升！",
        "眼神和嘴角微表情加强，撩人于无形！"
    ]
    
    # 50%概率添加额外建议
    if random.random() > 0.5:
        suggestion += " " + random.choice(extra_suggestions)
    
    return suggestion


def _get_top_parts(part_scores: Dict[str, int], n: int = 2) -> List[Tuple[str, int]]:
    """获取分数最高的前n个部位"""
    # 过滤掉分数为0的部位
    valid_parts = [(part, score) for part, score in part_scores.items() if score > 0]
    
    # 按分数排序
    valid_parts.sort(key=lambda x: x[1], reverse=True)
    
    return valid_parts[:n]


def _get_part_description(part: str, score: int) -> str:
    """获取部位描述"""
    descriptions = {
        'face': {
            'low': ['表情呆萌', '眼神清纯'],
            'medium': ['表情妩媚', '眼神撩人'],
            'high': ['表情超欲', '眼神拉丝']
        },
        'breast': {
            'low': ['小巧可爱', '保守含蓄'],
            'medium': ['丰满有料', '曲线优美'],
            'high': ['波涛汹涌', '性感爆棚']
        },
        'buttocks': {
            'low': ['圆润可爱', '保守包裹'],
            'medium': ['翘挺有型', '曲线诱人'],
            'high': ['蜜桃臀型', '性感炸裂']
        }
    }
    
    part_desc = descriptions.get(part, {})
    if score < 40:
        range_key = 'low'
    elif score < 70:
        range_key = 'medium'
    else:
        range_key = 'high'
    
    if part_desc and range_key in part_desc:
        return random.choice(part_desc[range_key])
    
    return ""