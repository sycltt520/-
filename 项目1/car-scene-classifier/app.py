"""
Gradio 可视化界面 —— 上传视频，实时推理，交互式展示结果。

启动:
  python app.py
  python app.py --ckpt checkpoints/best_model.pt --port 7860
"""
import os
import sys
import argparse
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import config as cfg
from inference import InferenceEngine, aggregate_results
from utils.video_utils import video_info, write_annotated_video

# ── 全局引擎（延迟加载） ──
_engine: InferenceEngine = None


def get_engine(ckpt_path: str) -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine(ckpt_path)
    return _engine


# ── 中文字体设置 ──
def _setup_chinese_font():
    """尝试设置 matplotlib 中文字体。"""
    for name in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
        for fpath in font_manager.findSystemFonts():
            if name.lower() in os.path.basename(fpath).lower():
                font_manager.fontManager.addfont(fpath)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=fpath).get_name()
                return
    plt.rcParams["font.family"] = "sans-serif"


_setup_chinese_font()


def process_video(video_path: str, ckpt_path: str, frame_skip: int, progress=gr.Progress()):
    """处理上传的视频并返回分析结果。"""
    if not video_path or not ckpt_path:
        return None, "请上传视频并选择模型文件。", None

    try:
        engine = get_engine(ckpt_path)
    except Exception as e:
        return None, f"模型加载失败: {e}", None

    # 视频信息
    info = video_info(video_path)
    info_text = (
        f"**视频信息**\n"
        f"- 总帧数: {info['total_frames']}\n"
        f"- FPS: {info['fps']:.1f}\n"
        f"- 分辨率: {info['width']} × {info['height']}\n"
        f"- 时长: {info['duration_sec']:.1f} 秒"
    )

    progress(0, desc="正在推理...")
    results = engine.predict_video(video_path, frame_skip=frame_skip)
    agg = aggregate_results(results)

    # ── 生成结果文本 ──
    result_text = (
        f"### 📊 推理结果\n"
        f"- **处理帧数**: {agg['total_frames']}\n"
        f"- **主导场景**: {agg['dominant_scene']} (置信度 {agg['scene_confidence']:.1%})\n"
        f"- **沙漠帧占比**: {agg['desert_frame_ratio']:.1%}\n"
        f"- **主导工况**: {agg['dominant_condition']} (置信度 {agg['condition_confidence']:.1%})\n"
        f"- **工况分布**:\n"
    )
    for cond, count in agg.get("condition_distribution", {}).items():
        pct = count / len(results) * 100 if results else 0
        result_text += f"  - {cond}: {count} 帧 ({pct:.1f}%)\n"

    # 达标提示
    if agg.get("dominant_scene") == "沙漠场景" and agg["scene_confidence"] >= 0.95:
        result_text += "\n✅ 场景识别置信度达标 (≥95%)"
    if agg.get("dominant_condition") != "不适用" and agg["condition_confidence"] >= 0.90:
        result_text += "\n✅ 工况识别置信度达标 (≥90%)"

    # ── 工况时序图 ──
    progress(0.8, desc="生成图表...")
    chart_path = _generate_timeline_chart(results, agg)

    # ── 标注视频 ──
    progress(0.9, desc="生成标注视频...")
    output_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    write_annotated_video(video_path, output_video, results,
                          fps=info.get("fps", 30))

    progress(1.0, desc="完成!")
    return output_video, result_text, chart_path


def _generate_timeline_chart(results: list, agg: dict) -> str:
    """生成工况随时间变化的堆叠柱状图。"""
    if not results:
        return None

    # 按时间窗口统计
    window_size = max(1, len(results) // 20)  # 最多 20 个窗口
    windows = []
    for i in range(0, len(results), window_size):
        chunk = results[i : i + window_size]
        windows.append(chunk)

    # 统计每个窗口的工况分布
    all_conditions = cfg.CONDITION_LABELS_ZH
    data = {c: [] for c in all_conditions}

    for chunk in windows:
        total = len(chunk)
        for c in all_conditions:
            cnt = sum(1 for r in chunk if r["condition_label"] == c)
            data[c].append(cnt / total * 100 if total else 0)

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(windows))
    bottom = np.zeros(len(windows))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9E9E9E"]

    for i, (cond, vals) in enumerate(data.items()):
        ax.bar(x, vals, bottom=bottom, label=cond, color=colors[i % len(colors)])
        bottom += np.array(vals)

    ax.set_xlabel("时间窗口")
    ax.set_ylabel("占比 (%)")
    ax.set_title(f"工况时序分布 — 主导场景: {agg['dominant_scene']} | 主导工况: {agg['dominant_condition']}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    chart_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig.savefig(chart_path, dpi=100)
    plt.close(fig)
    return chart_path


def build_interface(default_ckpt: str = ""):
    with gr.Blocks(title="汽车测试场景工况分类", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🚗 汽车测试场景工况分类系统
        上传一段视频，系统将逐帧分析**场景**（沙漠/其他）和**工况**（高速穿越/沙丘冲坡/刀锋侧倾/陷沙脱困）。
        """)

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="上传视频")
                ckpt_input = gr.File(
                    label="模型权重文件 (.pt)",
                    file_types=[".pt"],
                    value=default_ckpt if os.path.exists(default_ckpt) else None,
                )
                frame_skip = gr.Slider(
                    minimum=1, maximum=30, value=cfg.FRAME_SKIP, step=1,
                    label="帧采样间隔 (每隔 N 帧取 1 帧)",
                )
                btn = gr.Button("开始分析", variant="primary", size="lg")

                gr.Markdown("---")
                info_box = gr.Markdown("")

            with gr.Column(scale=2):
                result_box = gr.Markdown("### 等待分析...")
                chart_out = gr.Image(label="工况时序分布", type="filepath")
                video_out = gr.Video(label="标注视频 (带叠加信息)")

        btn.click(
            fn=process_video,
            inputs=[video_input, ckpt_input, frame_skip],
            outputs=[video_out, result_box, chart_out],
        )

        gr.Markdown("""
        ---
        ### 使用说明
        1. **准备模型**: 先运行 `python train.py` 训练模型，得到 `checkpoints/best_model.pt`
        2. **上传视频**: 支持 mp4 / avi / mov 等常见格式
        3. **选择模型**: 选择训练好的 `.pt` 权重文件
        4. **开始分析**: 系统将逐帧推理并展示结果

        ### 精度要求
        - 场景识别准确率 ≥ 95%
        - 工况识别准确率 ≥ 90%
        """)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化界面")
    parser.add_argument("--ckpt", type=str, default="", help="默认模型路径")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    args = parser.parse_args()

    default_ckpt = args.ckpt
    if not default_ckpt:
        ckpt_dir = cfg.CHECKPOINT_DIR
        if os.path.isdir(ckpt_dir):
            pts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            if pts:
                default_ckpt = os.path.join(ckpt_dir, sorted(pts)[-1])

    app = build_interface(default_ckpt)
    app.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
