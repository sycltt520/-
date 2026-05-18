"""
数据准备工具：从标注视频中提取帧并组织为训练目录结构。

用法:
  python prepare_data.py --videos_dir ./raw_videos --labels ./labels.csv --output ./data/train

labels.csv 格式:
  video_filename,scene,condition
  desert_highspeed_01.mp4,desert,high_speed_traversal
  desert_dune_01.mp4,desert,dune_climbing
  city_01.mp4,other,not_applicable
"""
import os
import csv
import argparse
import cv2
from pathlib import Path


def extract_and_save(video_path: str,
                     output_dir: str,
                     frame_interval: int = 10,
                     max_frames_per_video: int = 200):
    """从单个视频提取帧并保存到指定目录。"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved = 0
    video_stem = Path(video_path).stem

    while saved < max_frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"{video_stem}_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="从标注视频提取训练帧")
    parser.add_argument("--videos_dir", required=True, help="原始视频目录")
    parser.add_argument("--labels", required=True, help="标注 CSV 文件路径")
    parser.add_argument("--output", default="./data/train", help="输出目录 (默认 ./data/train)")
    parser.add_argument("--frame_interval", type=int, default=10,
                        help="每隔多少帧提取一帧 (默认 10)")
    parser.add_argument("--max_per_video", type=int, default=200,
                        help="每个视频最多提取帧数 (默认 200)")
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"[错误] 标注文件不存在: {args.labels}")
        return

    videos_dir = args.videos_dir
    output_base = args.output

    with open(args.labels, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total_saved = 0
        for row in reader:
            fname = row["video_filename"].strip()
            scene = row["scene"].strip()
            condition = row["condition"].strip()

            video_path = os.path.join(videos_dir, fname)
            if not os.path.exists(video_path):
                print(f"[跳过] 视频不存在: {video_path}")
                continue

            # 确定输出子目录
            if scene == "desert":
                out_dir = os.path.join(output_base, "desert", condition)
            else:
                out_dir = os.path.join(output_base, "other")

            n = extract_and_save(video_path, out_dir,
                                 frame_interval=args.frame_interval,
                                 max_frames_per_video=args.max_per_video)
            print(f"  {fname} → {out_dir} ({n} 帧)")
            total_saved += n

    print(f"\n完成! 共提取 {total_saved} 帧 → {output_base}")
    print("目录结构:")
    print(f"  {output_base}/")
    print(f"    desert/")
    for cond in ["high_speed_traversal", "dune_climbing", "knife_edge_tilt", "sand_trap_escape"]:
        cond_dir = os.path.join(output_base, "desert", cond)
        if os.path.isdir(cond_dir):
            n = len([f for f in os.listdir(cond_dir) if f.endswith(".jpg")])
            print(f"      {cond}/  ({n} 张)")
    other_dir = os.path.join(output_base, "other")
    if os.path.isdir(other_dir):
        n = len([f for f in os.listdir(other_dir) if f.endswith(".jpg")])
        print(f"    other/  ({n} 张)")


if __name__ == "__main__":
    main()
