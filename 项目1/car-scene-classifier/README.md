# 🚗 汽车测试场景工况分类系统

基于深度学习的视频帧级场景与工况分类模型。

**场景分类**（2 类）：沙漠场景 / 其他场景  
**工况分类**（5 类）：高速穿越 / 沙丘冲坡 / 刀锋侧倾 / 陷沙脱困 / 不适用

## 精度目标

| 指标 | 目标 |
|------|------|
| 场景识别准确率 | ≥ 95% |
| 工况识别准确率 | ≥ 90% |

## 项目结构

```
car-scene-classifier/
├── models/
│   └── classifier.py      # 模型定义（EfficientNet/ResNet 共享主干 + 双分类头）
├── data/
│   └── dataset.py         # 数据集加载与增强
├── utils/
│   └── video_utils.py     # 视频帧提取、标注叠加
├── config.py              # 全局配置
├── prepare_data.py        # 数据准备（标注视频 → 训练帧）
├── train.py               # 训练脚本
├── inference.py           # 推理引擎（命令行）
├── app.py                 # Gradio 可视化界面
├── requirements.txt       # 依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：PyTorch 请根据你的 CUDA 版本安装对应版本：
> ```bash
> # CPU 版本
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
>
> # CUDA 11.8
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
>
> # CUDA 12.1
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. 准备数据

将标注视频放入 `raw_videos/` 目录，创建标签文件 `labels.csv`：

```csv
video_filename,scene,condition
desert_highspeed_01.mp4,desert,high_speed_traversal
desert_dune_01.mp4,desert,dune_climbing
desert_knife_01.mp4,desert,knife_edge_tilt
desert_trap_01.mp4,desert,sand_trap_escape
city_road_01.mp4,other,not_applicable
```

运行数据提取：

```bash
python prepare_data.py --videos_dir ./raw_videos --labels ./labels.csv --output ./data/train --frame_interval 10
```

### 3. 训练模型

```bash
python train.py --epochs 50 --batch_size 32
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--model` | efficientnet_b0 | 主干网络 (efficientnet_b0/b1/b3/resnet50) |
| `--data` | ./data | 数据目录 |
| `--img_size` | 224 | 输入分辨率 |

训练完成后，最优模型保存在 `checkpoints/best_model.pt`。

### 4. 推理

**单张图片：**
```bash
python inference.py --image test.jpg --ckpt checkpoints/best_model.pt
```

**视频推理：**
```bash
python inference.py --video test.mp4 --ckpt checkpoints/best_model.pt --output annotated.mp4 --json_output results.json
```

### 5. 可视化界面

```bash
python app.py
```

浏览器打开 `http://localhost:7860`，上传视频即可交互式分析。

## 模型架构

```
输入图像 (224×224)
       │
  ┌────▼────────────────────┐
  │  EfficientNet-B0 主干   │  (ImageNet 预训练)
  │  (去除原始分类头)        │
  └────┬────────────────────┘
       │ 特征向量 (1280-d)
       │
  ┌────┴────────┐
  │             │
  ▼             ▼
场景头         工况头
(256→2)       (512→5)
  │             │
  ▼             ▼
沙漠/其他    高速穿越/沙丘冲坡
             /刀锋侧倾/陷沙脱困/不适用
```

## 数据目录格式

```
data/
├── train/
│   ├── desert/
│   │   ├── high_speed_traversal/
│   │   │   ├── video01_000010.jpg
│   │   │   └── ...
│   │   ├── dune_climbing/
│   │   ├── knife_edge_tilt/
│   │   └── sand_trap_escape/
│   └── other/
│       ├── city_000010.jpg
│       └── ...
├── val/        (可选)
└── test/       (可选)
```

## 技术说明

- **迁移学习**：使用 ImageNet 预训练的 EfficientNet 作为特征提取器，在目标数据上微调
- **双头架构**：共享主干网络，两个独立分类头分别预测场景和工况
- **数据增强**：随机裁剪、旋转、色彩抖动、高斯模糊，增强模型鲁棒性
- **批处理推理**：支持批量帧推理，充分利用 GPU 加速
- **时序平滑**：通过多数投票聚合帧级结果，输出视频级别分类

## 注意事项

1. 确保训练数据覆盖各种光照、天气、角度条件
2. "其他"场景应包含足够多样化的非沙漠场景（城市、高速、森林等）
3. 如模型未达标，可尝试：
   - 增加训练数据量
   - 使用更大的主干网络（`--model efficientnet_b3`）
   - 调整学习率和训练轮数
   - 增加数据增强强度
4. Windows 下建议 `NUM_WORKERS=0` 避免多进程问题
