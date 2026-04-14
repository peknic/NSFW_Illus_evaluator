"""
Gradio Web UI 界面
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
import time

from src.inference import create_detector
from utils.visualization import create_score_visualization, combine_images


# 全局检测器实例
_detector = None


def get_detector():
    """获取检测器实例（单例）"""
    global _detector
    if _detector is None:
        _detector = create_detector()
    return _detector


def analyze_image(input_image) -> tuple:
    """
    分析上传的图片
    
    Args:
        input_image: gradio上传的图片（numpy数组或文件路径）
        
    Returns:
        dict: 包含所有输出组件的值
    """
    detector = get_detector()
    
    try:
        # 处理输入：可能是numpy数组或文件路径
        if isinstance(input_image, str):
            image_path = input_image
        elif isinstance(input_image, np.ndarray):
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                temp_path = f.name
                cv2.imwrite(temp_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
                # 确保文件写入完成
                time.sleep(0.05)
                image_path = temp_path
        else:
            error_text = "[错误] 图片格式不支持\n请上传jpg、png或webp格式的图片"
            return (
                0,
                None,
                None,
                None,
                error_text
            )
        
        # 读取原始图像用于后续合并
        original_image = cv2.imread(image_path)
        if original_image is None:
            # 如果无法读取图像，直接返回错误结果
            error_text = f"[错误] 无法读取图片文件: {image_path}"
            # 清理临时文件（如果存在）
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return (
                0,
                None,
                None,
                None,
                error_text
            )
        
        # 执行分析
        result = detector.analyze(image_path)
        
        # 清理临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        # 创建分数图表
        score_chart = create_score_visualization(
            result['overall_score'], 
            result['part_scores']
        )
        
        # 准备输出
        output_image = result['annotated_image']
        
        # 转换BGR到RGB用于显示
        if output_image is not None:
            output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        else:
            output_image_rgb = None
        
        if score_chart is not None:
            score_chart_rgb = cv2.cvtColor(score_chart, cv2.COLOR_BGR2RGB)
        else:
            score_chart_rgb = None
        
        # 创建合并图像（原始+标注+图表）
        if output_image is not None and score_chart is not None and original_image is not None:
            combined = combine_images(original_image, output_image, score_chart)
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        else:
            combined_rgb = None
        
        # 格式化部位分数显示
        part_scores_text = "[部位分析]:\n"
        part_names_chinese = {
            'face': '脸部',
            'breast': '胸部', 
            'buttocks': '臀部',
            'thighs': '大腿',
            'waist': '腰部',
            'genitalia': '私密部位'
        }
        
        for part, score in result['part_scores'].items():
            if part in part_names_chinese:
                chinese_name = part_names_chinese[part]
                # 创建进度条表示（使用ASCII字符避免编码问题）
                bar_count = score // 10
                bars = "■" * bar_count if bar_count > 0 else ""
                part_scores_text += f"{chinese_name} {bars} {score}/100\n"
        
        # 构建完整输出文本
        output_text = f"[总体色气值]: {result['overall_score']}/100\n\n"
        output_text += part_scores_text + "\n"
        output_text += f"[评价]: {result['comment']}\n\n"
        output_text += f"[建议]: {result['suggestions']}"
        
        return (
            result['overall_score'],
            score_chart_rgb,
            output_image_rgb,
            combined_rgb,
            output_text
        )
        
    except Exception as e:
        error_msg = f"[错误] 分析过程中出错: {str(e)}"
        return (
            0,
            None,
            None,
            None,
            error_msg
        )


def create_ui():
    """创建Gradio界面"""
    
    with gr.Blocks(title="插画色气值检测器") as demo:
        gr.Markdown("""
        # 插画色气值检测器 v1.0
        **专门针对动漫/日漫/插画风格的NSFW检测工具**  
        *上传一张插画图片，分析其色气程度和各部位表现*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 上传区域
                image_input = gr.Image(
                    label="上传插画图片",
                    type="numpy",
                    sources=["upload"],  # 只保留upload，避免clipboard报错
                    interactive=True
                )
                
                # 上传按钮
                upload_button = gr.Button("开始分析", variant="primary", size="lg")
                

                
                # 设置区域
                with gr.Accordion("高级设置", open=False):
                    gr.Markdown("""
                    **模型设置**（当前使用默认配置）:
                    - NudeNet v3 部位检测
                    - NSFW 分类器 (EVA ViT)
                    - 插画风格优化
                    """)
            
            with gr.Column(scale=2):
                # 结果显示区域
                with gr.Tab("综合分析"):
                    combined_output = gr.Image(
                        label="分析结果汇总",
                        interactive=False
                    )
                
                with gr.Tab("部位检测"):
                    annotated_output = gr.Image(
                        label="部位检测可视化",
                        interactive=False
                    )
                
                with gr.Tab("[统计] 分数图表"):
                    chart_output = gr.Image(
                        label="分数分布图表",
                        interactive=False
                    )
                
                # 文本结果
                text_output = gr.Textbox(
                    label="详细分析报告",
                    lines=12,
                    interactive=False
                )
                
                # 总体分数显示
                score_output = gr.Number(
                    label="总体色气值",
                    value=0,
                    interactive=False
                )
        
        # 底部信息
        gr.Markdown("""
        ---
        ### 使用说明
        1. 上传或拖拽一张**插画/动漫/日漫风格**的图片
        2. 点击"开始分析"按钮
        3. 查看色气值评分和详细分析
        
        ### 注意事项
        - 本工具专门针对**插画风格**优化，真人照片效果不佳
        - 分数0-100，越高表示色气程度越高
        - 评价和建议仅供参考，请理性看待
        
        *本工具仅供学习和娱乐用途*
        """)
        
        # 绑定事件
        upload_button.click(
            fn=analyze_image,
            inputs=image_input,
            outputs=[score_output, chart_output, annotated_output, combined_output, text_output]
        )
        
        # 回车键触发分析
        image_input.upload(
            fn=analyze_image,
            inputs=image_input,
            outputs=[score_output, chart_output, annotated_output, combined_output, text_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )