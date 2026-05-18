"""
视频处理工具：帧提取、结果叠加、视频生成
"""
import os
import cv2
import numpy as np
from typing import List, Tuple, Generator
from PIL import Image


def extract_frames(video_path: str,
                   frame_skip: int = 1,
                   max_frames: int = 0) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    从视频中逐帧提取。
    Yields: (frame_index, BGR image)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    kept_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            yield frame_idx, frame
            kept_count += 1
            if max_frames > 0 and kept_count >= max_frames:
                break
        frame_idx += 1

    cap.release()


def video_info(video_path: str) -> dict:
    """获取视频基本信息"""
    cap = cv2.VideoCapture(video_path)
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": 0,
    }
    if info["fps"] > 0:
        info["duration_sec"] = info["total_frames"] / info["fps"]
    cap.release()
    return info


def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """BGR ndarray → RGB PIL Image"""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def draw_overlay(frame: np.ndarray,
                 scene_label: str,
                 condition_label: str,
                 scene_conf: float,
                 cond_conf: float,
                 frame_idx: int) -> np.ndarray:
    """
    在帧上叠加分类结果。
    返回带标注的 BGR 图像。
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 半透明顶栏
    bar_h = min(80, h // 6)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    overlay = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, w / 1200)
    thickness = max(2, int(w / 500))

    # 左上：场景
    scene_text = f"Scene: {scene_label} ({scene_conf:.1%})"
    cv2.putText(overlay, scene_text, (15, int(bar_h * 0.45)),
                font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # 右上：工况
    cond_text = f"Condition: {condition_label} ({cond_conf:.1%})"
    (tw, th), _ = cv2.getTextSize(cond_text, font, font_scale, thickness)
    cv2.putText(overlay, cond_text, (w - tw - 15, int(bar_h * 0.45)),
                font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    # 底部：帧号
    idx_text = f"Frame: {frame_idx}"
    cv2.putText(overlay, idx_text, (15, h - 15),
                font, font_scale * 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    return overlay


def write_annotated_video(input_path: str,
                          output_path: str,
                          results: List[dict],
                          fps: float = 30.0) -> str:
    """
    将推理结果写入带标注的视频文件。
    results: [{"frame_idx": int, "scene_label": str, "condition_label": str,
               "scene_conf": float, "cond_conf": float}, ...]
    """
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    result_map = {r["frame_idx"]: r for r in results}

    for frame_idx, frame in extract_frames(input_path, frame_skip=1):
        if frame_idx in result_map:
            r = result_map[frame_idx]
            frame = draw_overlay(frame, r["scene_label"], r["condition_label"],
                                 r["scene_conf"], r["cond_conf"], frame_idx)
        writer.write(frame)

    writer.release()
    return output_path
