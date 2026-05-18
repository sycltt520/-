"""
模型定义 —— 共享主干 + 双分类头（场景 / 工况）
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


class SceneConditionClassifier(nn.Module):
    """共享 EfficientNet / ResNet 主干，输出场景 + 工况两个分类结果。"""

    def __init__(self,
                 model_name: str = "efficientnet_b0",
                 num_scene_classes: int = 2,
                 num_condition_classes: int = 5,
                 pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_scene_classes = num_scene_classes
        self.num_condition_classes = num_condition_classes

        backbone, feature_dim = self._build_backbone(model_name, pretrained)
        self.backbone = backbone

        # ── 场景分类头 ──
        self.scene_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_scene_classes),
        )

        # ── 工况分类头 ──
        self.condition_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_condition_classes),
        )

    @staticmethod
    def _build_backbone(model_name: str, pretrained: bool):
        if model_name.startswith("efficientnet"):
            weights = "IMAGENET1K_V1" if pretrained else None
            if model_name == "efficientnet_b0":
                m = tv_models.efficientnet_b0(weights=weights)
                fd = 1280
            elif model_name == "efficientnet_b1":
                m = tv_models.efficientnet_b1(weights=weights)
                fd = 1280
            elif model_name == "efficientnet_b3":
                m = tv_models.efficientnet_b3(weights=weights)
                fd = 1536
            else:
                raise ValueError(f"Unknown efficientnet: {model_name}")
            m.classifier = nn.Identity()       # 去掉原始分类器
        elif model_name == "resnet50":
            weights = "IMAGENET1K_V2" if pretrained else None
            m = tv_models.resnet50(weights=weights)
            fd = m.fc.in_features
            m.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {model_name}")
        return m, fd

    def forward(self, x: torch.Tensor):
        """返回 (scene_logits, condition_logits)"""
        feats = self.backbone(x)
        return self.scene_head(feats), self.condition_head(feats)


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu"):
    """加载训练好的权重，处理 DataParallel 前缀。"""
    state = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    # 去掉可能存在的 "module." 前缀
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    model.load_state_dict(new_state, strict=True)
    return model
