"""
推理引擎 —— 单图 / 视频批量推理

用法:
  python inference.py --video input.mp4 --ckpt checkpoints/best_model.pt
  python inference.py --image frame.jpg --ckpt checkpoints/best_model.pt
"""
import os
import sys
import argparse
import json
from typing import List

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from models.classifier import SceneConditionClassifier, load_checkpoint
from utils.video_utils import extract_frames, frame_to_pil, write_annotated_video


def get_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class InferenceEngine:
    """封装模型加载与推理。"""

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(device or cfg.DEVICE)

        # 加载 checkpoint 获取模型配置
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model_cfg = ckpt.get("config", {})
        self.model_name = model_cfg.get("model_name", cfg.MODEL_NAME)
        self.img_size = model_cfg.get("img_size", cfg.IMG_SIZE)
        num_scene = model_cfg.get("num_scene_classes", cfg.NUM_SCENE_CLASSES)
        num_cond = model_cfg.get("num_condition_classes", cfg.NUM_CONDITION_CLASSES)

        self.model = SceneConditionClassifier(
            model_name=self.model_name,
            num_scene_classes=num_scene,
            num_condition_classes=num_cond,
            pretrained=False,
        ).to(self.device)
        self.model = load_checkpoint(self.model, checkpoint_path, str(self.device))
        self.model.eval()

        self.transform = get_transform(self.img_size)
        print(f"[推理引擎] 模型={self.model_name}, 分辨率={self.img_size}, 设备={self.device}")

    def predict_image(self, image: Image.Image):
        """单张 PIL 图片推理，返回 (scene_label, cond_label, scene_conf, cond_conf)。"""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            s_logits, c_logits = self.model(tensor)
            s_probs = torch.softmax(s_logits, dim=1)[0]
            c_probs = torch.softmax(c_logits, dim=1)[0]

        scene_idx = s_probs.argmax().item()
        cond_idx = c_probs.argmax().item()

        return (cfg.SCENE_LABELS_ZH[scene_idx],
                cfg.CONDITION_LABELS_ZH[cond_idx],
                s_probs[scene_idx].item(),
                c_probs[cond_idx].item())

    def predict_video(self,
                      video_path: str,
                      frame_skip: int = 1,
                      max_frames: int = 0) -> List[dict]:
        """逐帧推理，返回结果列表。"""
        results = []
        n_total = 0
        batch_imgs = []
        batch_indices = []

        print(f"[推理] 开始处理视频: {video_path}")

        for frame_idx, frame_bgr in extract_frames(video_path, frame_skip, max_frames):
            pil_img = frame_to_pil(frame_bgr)
            tensor = self.transform(pil_img)
            batch_imgs.append(tensor)
            batch_indices.append(frame_idx)

            if len(batch_imgs) >= cfg.BATCH_SIZE_INFER:
                self._infer_batch(batch_imgs, batch_indices, results)
                batch_imgs.clear()
                batch_indices.clear()

            n_total += 1
            if n_total % 100 == 0:
                print(f"  已处理 {n_total} 帧...")

        # 处理剩余
        if batch_imgs:
            self._infer_batch(batch_imgs, batch_indices, results)

        print(f"[推理] 完成，共 {len(results)} 帧")
        return results

    def _infer_batch(self, imgs: list, indices: list, results: list):
        batch = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            s_logits, c_logits = self.model(batch)
            s_probs = torch.softmax(s_logits, dim=1)
            c_probs = torch.softmax(c_logits, dim=1)

        for i, idx in enumerate(indices):
            s_pred = s_probs[i].argmax().item()
            c_pred = c_probs[i].argmax().item()
            results.append({
                "frame_idx": idx,
                "scene_label": cfg.SCENE_LABELS_ZH[s_pred],
                "condition_label": cfg.CONDITION_LABELS_ZH[c_pred],
                "scene_conf": s_probs[i][s_pred].item(),
                "cond_conf": c_probs[i][c_pred].item(),
            })


def aggregate_results(results: List[dict]) -> dict:
    """汇总视频整体结果。"""
    if not results:
        return {}

    # 场景：多数投票
    from collections import Counter
    scene_votes = Counter(r["scene_label"] for r in results)
    dominant_scene = scene_votes.most_common(1)[0]

    # 沙漠帧中的工况分布
    desert_results = [r for r in results if r["scene_label"] == "沙漠场景"]
    if desert_results:
        cond_votes = Counter(r["condition_label"] for r in desert_results)
        dominant_cond = cond_votes.most_common(1)[0]
    else:
        cond_votes = Counter()
        dominant_cond = ("不适用", 0)

    # 统计
    total = len(results)
    desert_ratio = len(desert_results) / total if total else 0

    return {
        "total_frames": total,
        "dominant_scene": dominant_scene[0],
        "scene_confidence": dominant_scene[1] / total,
        "desert_frame_ratio": desert_ratio,
        "dominant_condition": dominant_cond[0],
        "condition_confidence": dominant_cond[1] / len(desert_results) if desert_results else 0,
        "condition_distribution": dict(cond_votes.most_common()),
    }


def main():
    parser = argparse.ArgumentParser(description="场景工况推理")
    parser.add_argument("--video", type=str, help="输入视频路径")
    parser.add_argument("--image", type=str, help="输入图片路径")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--output", type=str, default=None, help="输出标注视频路径")
    parser.add_argument("--frame_skip", type=int, default=cfg.FRAME_SKIP)
    parser.add_argument("--json_output", type=str, default=None, help="输出 JSON 结果路径")
    args = parser.parse_args()

    engine = InferenceEngine(args.ckpt)

    if args.image:
        img = Image.open(args.image).convert("RGB")
        scene, cond, sc, cc = engine.predict_image(img)
        print(f"\n图片: {args.image}")
        print(f"  场景: {scene} (置信度 {sc:.1%})")
        print(f"  工况: {cond} (置信度 {cc:.1%})")

    elif args.video:
        results = engine.predict_video(args.video, frame_skip=args.frame_skip)
        agg = aggregate_results(results)

        print(f"\n视频: {args.video}")
        print(f"  总帧数(已处理): {agg['total_frames']}")
        print(f"  主导场景: {agg['dominant_scene']} (置信度 {agg['scene_confidence']:.1%})")
        print(f"  沙漠帧占比: {agg['desert_frame_ratio']:.1%}")
        print(f"  主导工况: {agg['dominant_condition']} (置信度 {agg['condition_confidence']:.1%})")
        print(f"  工况分布: {agg['condition_distribution']}")

        # 导出 JSON
        if args.json_output:
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": agg,
                    "frames": results,
                }, f, indent=2, ensure_ascii=False)
            print(f"  JSON 结果: {args.json_output}")

        # 导出标注视频
        if args.output:
            write_annotated_video(args.video, args.output, results)
            print(f"  标注视频: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
