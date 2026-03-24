"""Real-time video stream anomaly detection with OpenCV overlay."""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from data.dataset import IMAGENET_MEAN, IMAGENET_STD
from inference.base import BaseEngine

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Real-time anomaly detection on video streams.

    Captures frames from webcam or RTSP, runs inference through a pluggable
    engine, and overlays anomaly heatmaps with FPS counter.

    Args:
        engine: Any BaseEngine subclass for inference.
        source: Camera index (int) or RTSP URL (str).
        image_size: Model input size.
        threshold: Anomaly score threshold for alerting.
    """

    def __init__(
        self,
        engine: BaseEngine,
        source: int | str = 0,
        image_size: int = 224,
        threshold: float = 0.5,
    ) -> None:
        self.engine = engine
        self.source = source
        self.image_size = image_size
        self.threshold = threshold

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _open_capture(self) -> cv2.VideoCapture:
        """Open video capture with appropriate backend.

        Returns:
            OpenCV VideoCapture object.
        """
        if isinstance(self.source, int):
            # Use DirectShow backend on Windows for webcam
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        logger.info("Opened video source: %s", self.source)
        return cap

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a raw BGR frame for model input.

        Args:
            frame: BGR frame from OpenCV [H, W, 3].

        Returns:
            Preprocessed tensor [1, 3, image_size, image_size].
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(rgb)

    def _create_overlay(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        score: float,
        fps: float,
    ) -> np.ndarray:
        """Create visualization with heatmap overlay and metrics.

        Args:
            frame: Original BGR frame.
            heatmap: Anomaly heatmap [H, W].
            score: Anomaly score.
            fps: Current FPS.

        Returns:
            Annotated BGR frame.
        """
        display = frame.copy()
        h, w = display.shape[:2]

        # Normalize and colorize heatmap
        heatmap_norm = np.clip(heatmap / max(heatmap.max(), 1e-6), 0, 1)
        heatmap_resized = cv2.resize(heatmap_norm, (w, h))
        heatmap_color = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )

        # Blend heatmap with original frame
        alpha = 0.4
        display = cv2.addWeighted(display, 1 - alpha, heatmap_color, alpha, 0)

        # Draw metrics
        color = (0, 0, 255) if score > self.threshold else (0, 255, 0)
        cv2.putText(display, f"Score: {score:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if score > self.threshold:
            cv2.putText(display, "ANOMALY", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return display

    def run(self, save_dir: Path | None = None) -> None:
        """Start the streaming detection loop.

        Args:
            save_dir: Directory to save frames when 's' is pressed.
        """
        cap = self._open_capture()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        fps = 0.0
        prev_time = time.perf_counter()

        logger.info("Starting stream. Press 'q' to quit, 's' to save frame.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame, retrying...")
                    continue

                # Preprocess and infer
                tensor = self._preprocess_frame(frame)
                score, heatmap = self.engine.infer(tensor)

                # Compute FPS
                current_time = time.perf_counter()
                fps = 1.0 / max(current_time - prev_time, 1e-6)
                prev_time = current_time

                # Create overlay
                display = self._create_overlay(frame, heatmap, score, fps)
                cv2.imshow("Vigil - Anomaly Detection", display)

                frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested.")
                    break
                elif key == ord("s") and save_dir is not None:
                    save_path = save_dir / f"frame_{frame_count:06d}.png"
                    try:
                        cv2.imwrite(str(save_path), display)
                        logger.info("Frame saved to %s", save_path)
                    except Exception:
                        logger.exception("Failed to save frame")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Stream stopped after %d frames.", frame_count)


def main() -> None:
    """CLI entry point for streaming detection."""
    parser = argparse.ArgumentParser(description="Real-time anomaly detection stream")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: camera index (int) or RTSP URL",
    )
    parser.add_argument("--engine", type=str, default="pytorch", choices=["pytorch", "onnx", "tensorrt"])
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model file")
    parser.add_argument("--memory-bank", type=Path, default=None, help="Path to memory bank (.npz)")
    parser.add_argument("--backbone", type=str, default="efficientnet")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly score threshold")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory to save frames")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Parse source
    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    # Create engine
    engine: BaseEngine
    if args.engine == "pytorch":
        from inference.engine_pytorch import PyTorchEngine
        engine = PyTorchEngine(args.model_path, backbone_name=args.backbone)
    elif args.engine == "onnx":
        from inference.engine_onnx import ONNXEngine
        engine = ONNXEngine(args.model_path, memory_bank_path=args.memory_bank)
    elif args.engine == "tensorrt":
        from inference.engine_tensorrt import TensorRTEngine
        engine = TensorRTEngine(args.model_path, memory_bank_path=args.memory_bank)

    engine.warmup()

    processor = StreamProcessor(
        engine=engine,
        source=source,
        threshold=args.threshold,
    )
    processor.run(save_dir=args.save_dir)


if __name__ == "__main__":
    main()
