"""
模型管理器 - 处理模型下载和缓存
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器"""
    
    def __init__(self, model_dir: str = "models"):
        """
        初始化模型管理器
        
        Args:
            model_dir: 模型存储目录
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # 模型配置
        self.models = {
            'nudenet': {
                'name': 'NudeNet v3',
                'type': 'nudenet',
                'local_path': self.model_dir / 'nudenet_v3',
                'auto_download': False,  # NudeNet自带模型
                'description': '身体部位检测模型'
            },
            'nsfw_detector': {
                'name': 'NSFW Image Detector',
                'type': 'huggingface',
                'repo_id': 'Falconsai/nsfw_image_detection',
                'local_path': self.model_dir / 'nsfw_detector',
                'auto_download': True,
                'description': 'NSFW分类器（针对插画优化）'
            },
            'backup_detector': {
                'name': 'DINOv2 NSFW Classifier',
                'type': 'huggingface', 
                'repo_id': 'facebook/dinov2-base',
                'local_path': self.model_dir / 'dinov2_backup',
                'auto_download': False,
                'description': '备用特征提取器（需要微调）'
            }
        }
    
    def check_model_status(self, model_key: str) -> Dict[str, Any]:
        """
        检查模型状态
        
        Args:
            model_key: 模型键名
            
        Returns:
            模型状态字典
        """
        if model_key not in self.models:
            return {'exists': False, 'error': f'未知模型: {model_key}'}
        
        model_info = self.models[model_key]
        
        if model_info['type'] == 'nudenet':
            # NudeNet模型是内置的
            try:
                from nudenet import NudeDetector
                detector = NudeDetector()
                return {
                    'exists': True,
                    'ready': True,
                    'path': 'builtin',
                    'description': model_info['description']
                }
            except Exception as e:
                return {
                    'exists': False,
                    'error': f'NudeNet初始化失败: {e}',
                    'description': model_info['description']
                }
        
        elif model_info['type'] == 'huggingface':
            # 检查Hugging Face模型
            local_path = model_info['local_path']
            if local_path.exists():
                # 检查是否有必要的文件
                required_files = ['config.json', 'pytorch_model.bin']
                has_files = all((local_path / f).exists() for f in required_files)
                
                return {
                    'exists': True,
                    'ready': has_files,
                    'path': str(local_path),
                    'missing_files': [] if has_files else required_files,
                    'description': model_info['description']
                }
            else:
                return {
                    'exists': False,
                    'ready': False,
                    'path': str(local_path),
                    'description': model_info['description']
                }
        
        return {'exists': False, 'error': f'未知模型类型: {model_info["type"]}'}
    
    def download_model(self, model_key: str, force: bool = False) -> Dict[str, Any]:
        """
        下载模型
        
        Args:
            model_key: 模型键名
            force: 是否强制重新下载
            
        Returns:
            下载结果字典
        """
        if model_key not in self.models:
            return {'success': False, 'error': f'未知模型: {model_key}'}
        
        model_info = self.models[model_key]
        
        if model_info['type'] == 'nudenet':
            # NudeNet无需下载
            return {'success': True, 'message': 'NudeNet模型已内置，无需下载'}
        
        elif model_info['type'] == 'huggingface':
            # 下载Hugging Face模型
            repo_id = model_info['repo_id']
            local_path = model_info['local_path']
            
            # 检查是否已存在
            status = self.check_model_status(model_key)
            if status['exists'] and status['ready'] and not force:
                return {
                    'success': True, 
                    'message': f'模型已存在: {local_path}',
                    'path': str(local_path)
                }
            
            logger.info(f"正在下载模型: {repo_id}")
            logger.info(f"保存到: {local_path}")
            
            try:
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                
                # 创建目录
                local_path.mkdir(parents=True, exist_ok=True)
                
                # 下载处理器和模型
                logger.info("下载图像处理器...")
                processor = AutoImageProcessor.from_pretrained(
                    repo_id, 
                    cache_dir=str(local_path),
                    local_files_only=False
                )
                
                logger.info("下载模型...")
                model = AutoModelForImageClassification.from_pretrained(
                    repo_id,
                    cache_dir=str(local_path),
                    local_files_only=False
                )
                
                # 保存到本地
                logger.info("保存模型到本地...")
                processor.save_pretrained(str(local_path))
                model.save_pretrained(str(local_path))
                
                return {
                    'success': True,
                    'message': f'模型下载完成: {repo_id}',
                    'path': str(local_path),
                    'model': model,
                    'processor': processor
                }
                
            except Exception as e:
                error_msg = f'模型下载失败: {e}'
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
                
                # 提供手动下载说明
                manual_instructions = f"""
                手动下载说明:
                1. 访问: https://huggingface.co/{repo_id}
                2. 下载所有文件到: {local_path}
                3. 必需文件:
                   - config.json
                   - pytorch_model.bin
                   - preprocessor_config.json (如果有)
                4. 确保目录结构:
                   {local_path}/
                     ├── config.json
                     ├── pytorch_model.bin
                     └── preprocessor_config.json
                """
                
                return {
                    'success': False,
                    'error': error_msg,
                    'instructions': manual_instructions
                }
        
        return {'success': False, 'error': f'未知模型类型: {model_info["type"]}'}
    
    def initialize_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        初始化所有模型
        
        Returns:
            各模型初始化状态字典
        """
        results = {}
        
        for model_key in ['nudenet', 'nsfw_detector']:
            logger.info(f"初始化模型: {model_key}")
            
            if model_key == 'nudenet':
                # NudeNet总是可用的
                results[model_key] = self.check_model_status(model_key)
                continue
            
            # 检查NSFW检测器
            status = self.check_model_status(model_key)
            
            if not status['exists'] or not status['ready']:
                logger.info(f"模型 {model_key} 不存在或不完整，尝试下载...")
                download_result = self.download_model(model_key)
                
                if download_result['success']:
                    results[model_key] = {
                        'success': True,
                        'ready': True,
                        'message': '下载并初始化成功'
                    }
                else:
                    results[model_key] = {
                        'success': False,
                        'ready': False,
                        'error': download_result.get('error', '未知错误'),
                        'instructions': download_result.get('instructions', '')
                    }
            else:
                results[model_key] = {
                    'success': True,
                    'ready': True,
                    'message': '模型已就绪'
                }
        
        return results
    
    def get_model_info(self) -> str:
        """
        获取模型信息报告
        
        Returns:
            模型信息字符串
        """
        report = "模型状态报告:\n"
        report += "=" * 50 + "\n"
        
        for model_key, model_info in self.models.items():
            status = self.check_model_status(model_key)
            
            report += f"\n{model_info['name']} ({model_key}):\n"
            report += f"  描述: {model_info['description']}\n"
            
            if status.get('exists', False):
                if status.get('ready', False):
                    report += "  状态: [OK] 就绪\n"
                else:
                    report += "  状态: [WARN] 存在但不完整\n"
                    if 'missing_files' in status:
                        report += f"  缺失文件: {status['missing_files']}\n"
            else:
                report += "  状态: [ERROR] 未找到\n"
            
            if 'path' in status:
                report += f"  路径: {status['path']}\n"
        
        # 检查依赖
        report += "\n依赖检查:\n"
        report += "-" * 30 + "\n"
        
        dependencies = [
            ('torch', 'PyTorch'),
            ('transformers', 'Hugging Face Transformers'),
            ('nudenet', 'NudeNet'),
            ('gradio', 'Gradio UI'),
            ('opencv-python', 'OpenCV'),
            ('PIL', 'Pillow')
        ]
        
        for import_name, display_name in dependencies:
            try:
                if import_name == 'PIL':
                    import PIL
                    version = PIL.__version__
                else:
                    module = __import__(import_name)
                    version = getattr(module, '__version__', 'unknown')
                report += f"  {display_name}: [OK] {version}\n"
            except ImportError:
                report += f"  {display_name}: [ERROR] 未安装\n"
        
        return report


def setup_models():
    """设置模型（主入口函数）"""
    manager = ModelManager()
    
    print("插画色气值检测器 - 模型设置")
    print("=" * 60)
    
    # 显示当前状态
    print("\n当前模型状态:")
    print(manager.get_model_info())
    
    # 询问是否下载模型
    print("\n" + "=" * 60)
    print("需要下载以下模型:")
    print("1. NSFW Image Detector (Falconsai/nsfw_image_detection)")
    print("   - 用于总体色气值评估")
    print("   - 自动从 Hugging Face 下载")
    print()
    
    response = input("是否现在下载模型? (y/N): ").strip().lower()
    
    if response == 'y':
        print("\n开始下载模型...")
        results = manager.initialize_all_models()
        
        print("\n下载结果:")
        for model_key, result in results.items():
            model_name = manager.models[model_key]['name']
            if result.get('success', False):
                print(f"  {model_name}: [OK] {result.get('message', '成功')}")
            else:
                print(f"  {model_name}: [ERROR] {result.get('error', '失败')}")
                if 'instructions' in result:
                    print(f"     手动下载说明:\n{result['instructions']}")
    else:
        print("\n跳过模型下载。")
        print("运行时将尝试自动下载模型。")
        print("如果遇到网络问题，请手动下载:")
        print("1. 访问: https://huggingface.co/Falconsai/nsfw_image_detection")
        print("2. 下载所有文件到: models/nsfw_detector/")
    
    print("\n" + "=" * 60)
    print("设置完成！")
    print("运行以下命令启动应用:")
    print("  python main.py web     # 启动Web界面")
    print("  python main.py test    # 测试模型")


if __name__ == "__main__":
    setup_models()