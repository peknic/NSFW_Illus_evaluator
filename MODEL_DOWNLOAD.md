# 模型下载说明

## 必需的模型

> **注意**：第一阶段（裸露检测）需要 NSFW Image Detector 和 NudeNet v3。  
> 第二阶段（姿态色气评分）需要 SDPose-Wholebody 模型，若未安装则姿态评分降级为零分，但裸露检测仍可工作。

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
- **状态**: 已随 \nudenet\ 包安装，无需额外下载
- **验证**: 运行 \python main.py test\ 检查是否工作正常

### 3. SDPose-Wholebody (第二阶段：姿态色气评分)
- **用途**: 姿态/动作色气评分（SDPose-Wholebody 133关键点）
- **模型**: 需要下载两个部分：
  1. **SDPose-OOD 代码库** (GitHub)
     - 仓库：https://github.com/t-s-liang/SDPose-OOD
     - 操作：`git clone https://github.com/t-s-liang/SDPose-OOD.git models/SDPose-OOD`
     - 包含推理代码：`models/HeatmapHead.py`、`models/ModifiedUNet.py`、`pipelines/SDPose_D_Pipeline.py`、`gradio_app/SDPose_gradio.py`

  2. **SDPose-Wholebody 权重文件** (HuggingFace)
     - 仓库：https://huggingface.co/teemosliang/SDPose-Wholebody
     - 目标路径：`models/SDPose-Wholebody/`
     - **必须下载的文件列表**（保持目录结构）：
       - `decoder/decoder.safetensors` (~数十MB，heatmap head 权重)
       - `unet/config.json`
       - `unet/diffusion_pytorch_model.safetensors` (~约3 GB，SD U-Net 主干)
       - `vae/config.json`
       - `vae/diffusion_pytorch_model.safetensors` (~约800 MB)
       - `text_encoder/config.json`
       - `text_encoder/model.safetensors` (~约1.2 GB)
       - `tokenizer/merges.txt`
       - `tokenizer/special_tokens_map.json`
       - `tokenizer/tokenizer_config.json`
       - `tokenizer/vocab.json`
       - `scheduler/scheduler_config.json`
       - `yolo11x.pt` (~约130 MB，人体检测器)

- **下载方式**:
  1. **自动下载（推荐）**：
     ```bash
     pip install huggingface-hub
     huggingface-cli download teemosliang/SDPose-Wholebody --local-dir models/SDPose-Wholebody
     ```
  2. **手动下载**：
     - 访问上述 HuggingFace 页面，逐个下载文件
     - 按照目标路径创建目录并放置文件

- **目录结构验证**：
  ```
  models/
  ├── SDPose-OOD/                    # GitHub 代码库
  │   ├── models/
  │   │   ├── HeatmapHead.py
  │   │   └── ModifiedUNet.py
  │   ├── pipelines/
  │   │   └── SDPose_D_Pipeline.py
  │   └── gradio_app/
  │       └── SDPose_gradio.py
  └── SDPose-Wholebody/              # HuggingFace 权重
      ├── decoder/decoder.safetensors
      ├── unet/
      ├── vae/
      ├── text_encoder/
      ├── tokenizer/
      ├── scheduler/
      └── yolo11x.pt
  ```

- **注意**：
  - 若不安装 SDPose，姿态评分模块将降级为零分，但裸露检测仍正常工作
  - 模型总大小约 **5.2 GB**，请确保磁盘空间充足
  - 显存要求：RTX 5060 Ti 16GB VRAM 足以运行（使用 8bit/4bit 量化）

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
\\
🎨 插画色气值检测器 v2.0
==================================================
✅ 核心依赖检查通过
📁 目录结构就绪
🧪 测试模式 - 验证模型加载
✅ 模型加载成功
\\

**SDPose-Wholebody 验证**：
运行 Gradio 应用后，上传测试图片，查看「姿态分析」Tab：
- 若看到骨骼叠加图与姿态评分，则 SDPose 加载成功
- 若看到「未安装 SDPose 模型」警告，请按上方指引下载模型

如果看到错误信息，请按照错误提示进行操作。

## 常见问题

### Q: SDPose-Wholebody 模型太大（约5.2 GB），必须全部下载吗？
A: 是的，所有列出的文件都是必需的。SDPose 基于 Stable Diffusion U-Net，需要完整的权重文件才能运行。若磁盘空间不足，可仅使用第一阶段（裸露检测），此时姿态评分降级为零分。

### Q: 下载模型时网络超时怎么办？
A: 使用手动下载方式，然后将文件放到指定目录。

### Q: 模型下载到什么地方了？
A: 自动下载的模型默认存储在：
- Windows: \C:\Users\<用户名>\.cache\huggingface\hub\
- Linux/macOS: \~/.cache/huggingface/hub/\

项目会优先使用 \models/\ 目录下的本地副本。

### Q: 磁盘空间不足怎么办？
A: 第一阶段（裸露检测）需要约 300MB 空间。第二阶段（SDPose-Wholebody）需要约 5.2 GB 空间。若仅需裸露检测功能，可不下载 SDPose 模型，此时姿态评分降级为零分。

### Q: 如何重新下载模型？
A: 
- **NSFW 检测器**：删除 \models/nsfw_detector/\ 目录，然后重新运行程序。
- **SDPose-Wholebody**：删除 \models/SDPose-Wholebody/\ 目录，然后按上方指引重新下载。
- **SDPose-OOD 代码库**：删除 \models/SDPose-OOD/\ 目录，重新运行 \`git clone\`。

## 技术支持

如果遇到问题：
1. 检查网络连接
2. 确保磁盘空间充足
3. 查看错误日志中的详细提示
4. 尝试手动下载方式

项目 GitHub 仓库（如果有）或联系开发者获取帮助。
