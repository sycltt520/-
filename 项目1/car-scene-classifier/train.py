"""
训练脚本 —— 双头分类器（场景 + 工况）

用法:
  python train.py                          # 使用 config.py 默认配置
  python train.py --epochs 100 --lr 5e-5   # 自定义参数
"""
import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from models.classifier import SceneConditionClassifier
from data.dataset import create_dataloaders


def train_one_epoch(model, loader, criterion_scene, criterion_cond, optimizer, device):
    model.train()
    total_loss = 0.0
    all_scene_preds, all_scene_labels = [], []
    all_cond_preds, all_cond_labels = [], []

    for imgs, scene_labels, cond_labels in loader:
        imgs = imgs.to(device)
        scene_labels = scene_labels.to(device)
        cond_labels = cond_labels.to(device)

        optimizer.zero_grad()
        scene_logits, cond_logits = model(imgs)
        loss_s = criterion_scene(scene_logits, scene_labels)
        loss_c = criterion_cond(cond_logits, cond_labels)
        loss = loss_s + loss_c
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        all_scene_preds.extend(scene_logits.argmax(dim=1).cpu().tolist())
        all_scene_labels.extend(scene_labels.cpu().tolist())
        all_cond_preds.extend(cond_logits.argmax(dim=1).cpu().tolist())
        all_cond_labels.extend(cond_labels.cpu().tolist())

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "scene_acc": accuracy_score(all_scene_labels, all_scene_preds),
        "cond_acc": accuracy_score(all_cond_labels, all_cond_preds),
        "scene_f1": f1_score(all_scene_labels, all_scene_preds, average="weighted"),
        "cond_f1": f1_score(all_cond_labels, all_cond_preds, average="weighted"),
    }


@torch.no_grad()
def evaluate(model, loader, criterion_scene, criterion_cond, device):
    model.eval()
    total_loss = 0.0
    all_scene_preds, all_scene_labels = [], []
    all_cond_preds, all_cond_labels = [], []

    for imgs, scene_labels, cond_labels in loader:
        imgs = imgs.to(device)
        scene_labels = scene_labels.to(device)
        cond_labels = cond_labels.to(device)

        scene_logits, cond_logits = model(imgs)
        loss_s = criterion_scene(scene_logits, scene_labels)
        loss_c = criterion_cond(cond_logits, cond_labels)
        total_loss += (loss_s + loss_c).item() * imgs.size(0)

        all_scene_preds.extend(scene_logits.argmax(dim=1).cpu().tolist())
        all_scene_labels.extend(scene_labels.cpu().tolist())
        all_cond_preds.extend(cond_logits.argmax(dim=1).cpu().tolist())
        all_cond_labels.extend(cond_labels.cpu().tolist())

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "scene_acc": accuracy_score(all_scene_labels, all_scene_preds),
        "cond_acc": accuracy_score(all_cond_labels, all_cond_preds),
        "scene_f1": f1_score(all_scene_labels, all_scene_preds, average="weighted"),
        "cond_f1": f1_score(all_cond_labels, all_cond_preds, average="weighted"),
        "scene_report": classification_report(all_scene_labels, all_scene_preds,
                                              target_names=cfg.SCENE_LABELS, zero_division=0),
        "cond_report": classification_report(all_cond_labels, all_cond_preds,
                                             target_names=cfg.CONDITION_LABELS, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="训练场景工况分类模型")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--model", type=str, default=cfg.MODEL_NAME)
    parser.add_argument("--data", type=str, default=cfg.DATA_DIR)
    parser.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    args = parser.parse_args()

    device = torch.device(cfg.DEVICE)
    print(f"[设备] {device}")
    print(f"[模型] {args.model}")

    # ── 数据 ─────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=cfg.NUM_WORKERS,
    )

    if train_loader is None:
        print("[错误] 未找到训练数据。请先运行 prepare_data.py 准备数据。")
        print("  期望的目录结构：")
        print(f"    {args.data}/")
        print("      train/")
        print("        desert/high_speed_traversal/*.jpg")
        print("        desert/dune_climbing/*.jpg")
        print("        desert/knife_edge_tilt/*.jpg")
        print("        desert/sand_trap_escape/*.jpg")
        print("        other/*.jpg")
        return

    # ── 模型 ─────────────────────────────────────────────
    model = SceneConditionClassifier(
        model_name=args.model,
        num_scene_classes=cfg.NUM_SCENE_CLASSES,
        num_condition_classes=cfg.NUM_CONDITION_CLASSES,
        pretrained=cfg.PRETRAINED,
    ).to(device)

    # 类别权重（处理不平衡）
    criterion_scene = nn.CrossEntropyLoss()
    criterion_cond = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
    patience_counter = 0
    history = {"train": [], "val": []}

    print(f"\n{'='*60}")
    print(f"开始训练 — {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader,
                                        criterion_scene, criterion_cond,
                                        optimizer, device)
        history["train"].append(train_metrics)

        # 验证
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader,
                                   criterion_scene, criterion_cond, device)
            history["val"].append(val_metrics)

            scene_acc = val_metrics["scene_acc"]
            cond_acc = val_metrics["cond_acc"]
            combined = (scene_acc + cond_acc) / 2

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"T_loss={train_metrics['loss']:.4f} | "
                  f"V_S={scene_acc:.3f} V_C={cond_acc:.3f} V_avg={combined:.3f}")

            if combined > best_val_acc:
                best_val_acc = combined
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": {
                        "model_name": args.model,
                        "img_size": args.img_size,
                        "num_scene_classes": cfg.NUM_SCENE_CLASSES,
                        "num_condition_classes": cfg.NUM_CONDITION_CLASSES,
                    },
                }, best_ckpt_path)
                print(f"  ✓ 保存最佳模型 (avg={combined:.3f})")
            else:
                patience_counter += 1
        else:
            # 无验证集时，每 5 epoch 保存一次
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"T_loss={train_metrics['loss']:.4f} | "
                  f"T_S={train_metrics['scene_acc']:.3f} T_C={train_metrics['cond_acc']:.3f}")
            if epoch % 5 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "model_name": args.model,
                        "img_size": args.img_size,
                        "num_scene_classes": cfg.NUM_SCENE_CLASSES,
                        "num_condition_classes": cfg.NUM_CONDITION_CLASSES,
                    },
                }, best_ckpt_path)

        scheduler.step()

        if patience_counter >= cfg.EARLY_STOP_PATIENCE:
            print(f"\n早停触发 (连续 {cfg.EARLY_STOP_PATIENCE} 轮未改善)")
            break

    # ── 最终测试 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("最终测试")
    if test_loader is not None:
        # 加载最佳模型
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, test_loader,
                                criterion_scene, criterion_cond, device)
        print(f"测试集 — 场景准确率: {test_metrics['scene_acc']:.4f}")
        print(f"         工况准确率: {test_metrics['cond_acc']:.4f}")
        print(f"\n场景分类报告:\n{test_metrics['scene_report']}")
        print(f"工况分类报告:\n{test_metrics['cond_report']}")

        # 达标检查
        if test_metrics["scene_acc"] >= 0.95:
            print("✓ 场景识别达标 (≥95%)")
        else:
            print(f"✗ 场景识别未达标 (当前 {test_metrics['scene_acc']:.2%}, 目标 ≥95%)")
        if test_metrics["cond_acc"] >= 0.90:
            print("✓ 工况识别达标 (≥90%)")
        else:
            print(f"✗ 工况识别未达标 (当前 {test_metrics['cond_acc']:.2%}, 目标 ≥90%)")

        # 保存日志
        log_path = os.path.join(cfg.LOG_DIR,
                                f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        log_data = {
            "args": vars(args),
            "test_metrics": {k: v for k, v in test_metrics.items()
                             if k not in ("scene_report", "cond_report")},
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"\n日志已保存: {log_path}")
    else:
        print("(无测试集)")

    print(f"\n最佳模型: {best_ckpt_path}")
    print("训练完成!")


if __name__ == "__main__":
    main()
