#!/usr/bin/env python3
"""
插画色气值检测器 - 主启动入口
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    print("插画色气值检测器 v1.0")
    print("=" * 50)
    
    # 检查依赖
    try:
        import gradio
        import torch
        import cv2
        print("核心依赖检查通过")
    except ImportError as e:
        print(f"[ERROR] 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return 1
    
    # 检查模型目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("examples/high_score", exist_ok=True)
    os.makedirs("examples/low_score", exist_ok=True)
    
    print("目录结构就绪")
    
    # 启动模式选择
    if len(sys.argv) > 1:
        if sys.argv[1] == "web":
            # 启动Web UI
            from gradio_app.app import create_ui
            print("启动Gradio Web UI...")
            print("服务地址: http://localhost:7861")
            demo = create_ui()
            import gradio as gr
            demo.launch(server_name="0.0.0.0", server_port=7861, share=False, theme=gr.themes.Soft())
        elif sys.argv[1] == "test":
            # 测试模式
            from src.inference import create_detector
            print("测试模式 - 验证模型加载")
            try:
                detector = create_detector()
                print("模型加载成功")
                
                # 简单测试
                import tempfile
                import numpy as np
                import cv2
                
                # 创建测试图像
                test_image = np.zeros((200, 200, 3), dtype=np.uint8)
                test_image[50:150, 50:150] = [255, 200, 200]
                
                # 创建临时文件
                import tempfile
                import time
                temp_path = os.path.join(tempfile.gettempdir(), f"test_model_{int(time.time())}.jpg")
                cv2.imwrite(temp_path, test_image)
                
                try:
                    result = detector.analyze(temp_path)
                    print(f"测试分析结果:")
                    print(f"  总体色气值: {result['overall_score']}/100")
                    print(f"  评价: {result['comment']}")
                    print(f"  建议: {result['suggestions']}")
                finally:
                    # 清理临时文件
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                    except:
                        pass
                    
            except Exception as e:
                print(f"模型测试失败: {e}")
                print("\n解决方法:")
                print("1. 运行: python -m src.model_manager 设置模型")
                print("2. 查看 MODEL_DOWNLOAD.md 了解手动下载方法")
                return 1
                
        elif sys.argv[1] == "setup":
            # 模型设置模式
            from src.model_manager import setup_models
            setup_models()
            
        elif sys.argv[1].endswith(('.jpg', '.png', '.jpeg', '.webp')):
            # 单张图片分析
            from src.inference import create_detector
            image_path = sys.argv[1]
            if os.path.exists(image_path):
                print(f"分析图片: {image_path}")
                try:
                    detector = create_detector()
                    result = detector.analyze(image_path)
                    print(f"总体色气值: {result['overall_score']}/100")
                    print(f"评价: {result['comment']}")
                    print(f"建议: {result['suggestions']}")
                except Exception as e:
                    print(f"分析失败: {e}")
                    print("运行 'python main.py setup' 设置模型")
                    return 1
            else:
                print(f"[ERROR] 图片不存在: {image_path}")
                return 1
        else:
            print_help()
    else:
        # 交互式模式
        print_help()
    
    return 0


def print_help():
    """打印帮助信息"""
    print("\n使用方式:")
    print("  python main.py setup       设置和下载模型")
    print("  python main.py test        测试模型加载")
    print("  python main.py web         启动Web界面")
    print("  python main.py <图片路径>  分析单张图片")
    print("\nWeb界面访问: http://localhost:7861")
    print("示例图片目录: examples/")
    print("\n配置:")
    print("  编辑 src/config.py 调整参数")
    print("  查看 MODEL_DOWNLOAD.md 了解模型下载")
    print("  添加示例图片到 examples/ 目录")
    print("\n第一次使用:")
    print("  1. 运行: python main.py setup")
    print("  2. 或查看: MODEL_DOWNLOAD.md")


if __name__ == "__main__":
    sys.exit(main())