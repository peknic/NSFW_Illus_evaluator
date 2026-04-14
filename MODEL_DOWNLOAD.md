# 模型下载说明

## 必需的模型

### 1. NSFW Image Detector (主要模型)
- **用途**: 总体色气值评估
- **模型**: \Falconsai/nsfw_image_detection\
- **大小**: ~300MB
- **下载方式**: 自动或手动

#### 自动下载（推荐）
运行以下命令，程序会自动下载模型：

\\\ash
python -m src.model_manager
\\\

或者直接运行应用，第一次运行时会自动尝试下载：

\\\ash
python main.py test
\\\

#### 手动下载（如果自动下载失败）
1. **访问模型页面**: https://huggingface.co/Falconsai/nsfw_image_detection
2. **下载文件**:
   - 点击 \"Files and versions\" 标签页
   - 下载以下文件:
     - \config.json\
     - \pytorch_model.bin\
     - \preprocessor_config.json\
3. **创建目录结构**:
   \\\ash
   mkdir -p models/nsfw_detector
   \\\
4. **放置文件**:
   - 将所有下载的文件放到 \models/nsfw_detector/\ 目录下
5. **验证目录结构**:
   \\\
   models/nsfw_detector/
   ├── config.json
   ├── pytorch_model.bin
   └── preprocessor_config.json
   \\\

### 2. NudeNet v3 (已内置)
- **用途**: 身体部位检测
- **状态**: 已随 \
udenet\ 包安装，无需额外下载
- **验证**: 运行 \python main.py test\ 检查是否工作正常

## 可选模型（备用）

### DINOv2 Base
- **用途**: 备用特征提取器
- **模型**: \acebook/dinov2-base\
- **何时需要**: 如果主要模型不可用或需要更高级的特征提取
- **下载**: 仅在需要时自动下载

## 验证安装

运行以下命令验证所有模型是否就绪：

\\\ash
python main.py test
\\\

预期输出：
\\\
🎨 插画色气值检测器 v1.0
==================================================
✅ 核心依赖检查通过
📁 目录结构就绪
🧪 测试模式 - 验证模型加载
✅ 模型加载成功
\\\

如果看到错误信息，请按照错误提示进行操作。

## 常见问题

### Q: 下载模型时网络超时怎么办？
A: 使用手动下载方式，然后将文件放到指定目录。

### Q: 模型下载到什么地方了？
A: 自动下载的模型默认存储在：
- Windows: \C:\Users\<用户名>\.cache\huggingface\hub\
- Linux/macOS: \~/.cache/huggingface/hub/\

项目会优先使用 \models/\ 目录下的本地副本。

### Q: 磁盘空间不足怎么办？
A: 必需模型大约需要 300MB 空间，请确保有足够空间。

### Q: 如何重新下载模型？
A: 删除 \models/nsfw_detector/\ 目录，然后重新运行程序。

## 技术支持

如果遇到问题：
1. 检查网络连接
2. 确保磁盘空间充足
3. 查看错误日志中的详细提示
4. 尝试手动下载方式

项目 GitHub 仓库（如果有）或联系开发者获取帮助。
