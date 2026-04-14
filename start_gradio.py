#!/usr/bin/env python3
"""
直接启动Gradio Web UI
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("直接启动Gradio Web UI...")
    print("=" * 50)
    
    try:
        # 直接导入Gradio应用
        from gradio_app.app import create_ui
        import gradio as gr
        
        print("创建UI界面...")
        demo = create_ui()
        print("服务地址: http://localhost:7861")
        print("按 Ctrl+C 停止服务")
        
        # 启动服务
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,  # 使用7861端口避免冲突
            share=False,
            theme=gr.themes.Soft(),
            show_error=True
        )
        
    except ImportError as e:
        print(f"[ERROR] 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"[ERROR] 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())