"""Gradio demo for real-time anomaly detection."""

import argparse
import logging
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from torchvision import transforms

from data.dataset import IMAGENET_MEAN, IMAGENET_STD
from inference.base import BaseEngine

logger = logging.getLogger(__name__)

# Global engine cache
_engines: dict[str, BaseEngine] = {}

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _get_engine(
    backend: str,
    model_dir: Path,
    memory_bank_path: Path | None,
    backbone: str,
) -> BaseEngine | None:
    """Get or create a cached engine instance.

    Args:
        backend: Backend identifier string.
        model_dir: Directory containing model files.
        memory_bank_path: Path to PatchCore memory bank.
        backbone: Backbone name.

    Returns:
        Engine instance or None if unavailable.
    """
    if backend in _engines:
        return _engines[backend]

    try:
        if backend == "PyTorch":
            from inference.engine_pytorch import PyTorchEngine
            engine = PyTorchEngine(model_path=memory_bank_path, backbone_name=backbone)
        elif backend == "ONNX":
            from inference.engine_onnx import ONNXEngine
            onnx_path = model_dir / f"{backbone}.onnx"
            engine = ONNXEngine(model_path=onnx_path, memory_bank_path=memory_bank_path)
        elif backend == "TensorRT-FP16":
            from inference.engine_tensorrt import TensorRTEngine
            trt_path = model_dir / f"{backbone}_fp16.trt"
            engine = TensorRTEngine(model_path=trt_path, memory_bank_path=memory_bank_path)
        elif backend == "TensorRT-INT8":
            from inference.engine_tensorrt import TensorRTEngine
            trt_path = model_dir / f"{backbone}_int8.trt"
            engine = TensorRTEngine(model_path=trt_path, memory_bank_path=memory_bank_path)
        else:
            logger.error("Unknown backend: %s", backend)
            return None

        engine.warmup(n=5)
        _engines[backend] = engine
        return engine

    except Exception:
        logger.exception("Failed to create %s engine", backend)
        return None


def process_image(
    image: np.ndarray,
    backend: str,
    threshold: float,
    model_dir: str,
    memory_bank: str,
    backbone: str,
) -> tuple[np.ndarray, str]:
    """Process a single image through the anomaly detection pipeline.

    Args:
        image: Input RGB image.
        backend: Selected inference backend.
        threshold: Anomaly score threshold.
        model_dir: Model directory path string.
        memory_bank: Memory bank path string.
        backbone: Backbone name.

    Returns:
        Tuple of (overlay_image, metrics_text).
    """
    if image is None:
        return np.zeros((224, 224, 3), dtype=np.uint8), "No image provided"

    model_dir_path = Path(model_dir)
    memory_bank_path = Path(memory_bank) if memory_bank else None

    engine = _get_engine(backend, model_dir_path, memory_bank_path, backbone)
    if engine is None:
        return image, f"Backend '{backend}' unavailable"

    # Preprocess
    if image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]

    tensor = _transform(image)

    # Infer
    start = time.perf_counter()
    score, heatmap = engine.infer(tensor)
    inference_ms = (time.perf_counter() - start) * 1000

    # Create overlay
    h, w = image.shape[:2]
    heatmap_norm = np.clip(heatmap / max(heatmap.max(), 1e-6), 0, 1)
    heatmap_resized = cv2.resize(heatmap_norm, (w, h))
    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    # Convert from BGR to RGB for Gradio
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    alpha = 0.4
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    # Build metrics text
    status = "ANOMALY DETECTED" if score > threshold else "Normal"
    metrics = (
        f"Status: {status}\n"
        f"Anomaly Score: {score:.4f}\n"
        f"Threshold: {threshold:.2f}\n"
        f"Inference Time: {inference_ms:.1f} ms\n"
        f"FPS: {1000 / max(inference_ms, 0.1):.1f}\n"
        f"Backend: {backend}"
    )

    return overlay, metrics


def create_app(
    model_dir: Path,
    memory_bank: Path | None,
    backbone: str,
) -> gr.Blocks:
    """Create the Gradio demo interface.

    Args:
        model_dir: Directory containing model files.
        memory_bank: Path to PatchCore memory bank.
        backbone: Backbone name.

    Returns:
        Gradio Blocks application.
    """
    model_dir_str = str(model_dir)
    memory_bank_str = str(memory_bank) if memory_bank else ""

    with gr.Blocks(title="Vigil - Anomaly Detection") as app:
        gr.Markdown("# Vigil - Real-Time Anomaly Detection")
        gr.Markdown("Upload an image or use webcam to detect anomalies using PatchCore.")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input", type="numpy")
                backend_selector = gr.Dropdown(
                    choices=["PyTorch", "ONNX", "TensorRT-FP16", "TensorRT-INT8"],
                    value="PyTorch",
                    label="Inference Backend",
                )
                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=50.0,
                    value=15.0,
                    step=0.5,
                    label="Anomaly Threshold (raw distance score)",
                )
                run_button = gr.Button("Detect Anomalies", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Anomaly Heatmap Overlay")
                metrics_output = gr.Textbox(label="Metrics", lines=6)

        # Hidden state for config
        model_dir_state = gr.State(model_dir_str)
        memory_bank_state = gr.State(memory_bank_str)
        backbone_state = gr.State(backbone)

        run_button.click(
            fn=process_image,
            inputs=[
                input_image,
                backend_selector,
                threshold_slider,
                model_dir_state,
                memory_bank_state,
                backbone_state,
            ],
            outputs=[output_image, metrics_output],
        )

    return app


def main() -> None:
    """CLI entry point for the Gradio demo."""
    parser = argparse.ArgumentParser(description="Vigil anomaly detection demo")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("exports"),
        help="Directory containing exported models",
    )
    parser.add_argument(
        "--memory-bank",
        type=Path,
        default=None,
        help="Path to PatchCore memory bank (.npz)",
    )
    parser.add_argument("--backbone", type=str, default="efficientnet")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    app = create_app(
        model_dir=args.model_dir,
        memory_bank=args.memory_bank,
        backbone=args.backbone,
    )
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
