"""
汽车测试场景工况分类模型 —— 全局配置
"""
import os
import torch

# ── 项目路径 ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR       = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "output")

for _d in [CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── 设备 ─────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 模型 ─────────────────────────────────────────────────
MODEL_NAME             = "efficientnet_b0"   # efficientnet_b0 / b1 / b3 / resnet50
PRETRAINED             = True
IMG_SIZE               = 224                 # 输入分辨率
NUM_SCENE_CLASSES      = 2                   # 场景：其他 / 沙漠
NUM_CONDITION_CLASSES  = 5                   # 工况：4 类沙漠工况 + 不适用

# ── 类别标签 ─────────────────────────────────────────────
SCENE_LABELS     = ["other", "desert"]
SCENE_LABELS_ZH  = ["其他场景", "沙漠场景"]

CONDITION_LABELS    = ["high_speed_traversal", "dune_climbing",
                       "knife_edge_tilt", "sand_trap_escape", "not_applicable"]
CONDITION_LABELS_ZH = ["高速穿越", "沙丘冲坡", "刀锋侧倾", "陷沙脱困", "不适用"]

# ── 训练 ─────────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_EPOCHS      = 50
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 0                     # Windows 下建议 0
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
EARLY_STOP_PATIENCE = 10

# ── 推理 ─────────────────────────────────────────────────
FRAME_SKIP            = 1               # 每隔 N 帧取 1 帧
CONFIDENCE_THRESHOLD  = 0.5
BATCH_SIZE_INFER      = 16
