"""
数据集加载与增强
期望目录结构：
  data/
    train/   (或 val/  test/)
      desert/
        high_speed_traversal/   ← 图片
        dune_climbing/
        knife_edge_tilt/
        sand_trap_escape/
      other/                    ← 图片直接放在此目录下
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Optional


class SceneConditionDataset(Dataset):
    def __init__(self,
                 root: str,
                 img_size: int = 224,
                 augment: bool = False,
                 scene_labels: Optional[List[str]] = None,
                 condition_labels: Optional[List[str]] = None):
        self.root = root
        self.augment = augment
        self.samples: List[Tuple[str, int, int]] = []  # (path, scene_id, condition_id)

        if scene_labels is None:
            scene_labels = ["other", "desert"]
        if condition_labels is None:
            condition_labels = ["high_speed_traversal", "dune_climbing",
                                "knife_edge_tilt", "sand_trap_escape", "not_applicable"]

        self.scene_to_id = {v: i for i, v in enumerate(scene_labels)}
        self.cond_to_id  = {v: i for i, v in enumerate(condition_labels)}

        # ── 扫描 desert 子目录 ──
        desert_dir = os.path.join(root, "desert")
        if os.path.isdir(desert_dir):
            for cond_name in os.listdir(desert_dir):
                cond_path = os.path.join(desert_dir, cond_name)
                if not os.path.isdir(cond_path):
                    continue
                if cond_name not in self.cond_to_id:
                    print(f"[警告] 未知工况目录: {cond_name}")
                    continue
                cond_id = self.cond_to_id[cond_name]
                for fname in os.listdir(cond_path):
                    if self._is_image(fname):
                        self.samples.append((
                            os.path.join(cond_path, fname),
                            self.scene_to_id["desert"],
                            cond_id,
                        ))

        # ── 扫描 other 目录 ──
        other_dir = os.path.join(root, "other")
        if os.path.isdir(other_dir):
            for fname in os.listdir(other_dir):
                fpath = os.path.join(other_dir, fname)
                if os.path.isfile(fpath) and self._is_image(fname):
                    self.samples.append((
                        fpath,
                        self.scene_to_id["other"],
                        self.cond_to_id["not_applicable"],
                    ))

        if len(self.samples) == 0:
            raise RuntimeError(f"在 {root} 中未找到任何图片。"
                               "请确认目录结构：desert/<工况>/xxx.jpg 或 other/xxx.jpg")

        # ── 数据增强 ──
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    @staticmethod
    def _is_image(fname: str) -> bool:
        ext = os.path.splitext(fname)[1].lower()
        return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, scene_id, cond_id = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[警告] 读取图片失败 {path}: {e}，返回空白图")
            img = Image.new("RGB", (224, 224))
        return self.transform(img), torch.tensor(scene_id, dtype=torch.long), torch.tensor(cond_id, dtype=torch.long)


def create_dataloaders(data_dir: str,
                       img_size: int = 224,
                       batch_size: int = 32,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """按 train / val / test 子目录创建 DataLoader"""
    loaders = {}
    for split, aug in [("train", True), ("val", False), ("test", False)]:
        root = os.path.join(data_dir, split)
        if not os.path.isdir(root):
            print(f"[提示] {root} 不存在，跳过 {split} 数据集")
            loaders[split] = None
            continue
        ds = SceneConditionDataset(root, img_size=img_size, augment=aug)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
        )
        print(f"[数据] {split}: {len(ds)} 张图片")
    return loaders.get("train"), loaders.get("val"), loaders.get("test")
