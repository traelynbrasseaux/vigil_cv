"""PyTorch inference engine for PatchCore anomaly detection."""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference.base import BaseEngine
from models.patchcore import PatchCore

logger = logging.getLogger(__name__)


class PyTorchEngine(BaseEngine):
    """PyTorch-based inference engine.

    Args:
        model_path: Path to the PatchCore checkpoint (.npz file).
        device: Torch device string.
        backbone_name: Backbone name used during training.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        backbone_name: str = "efficientnet",
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.backbone_name = backbone_name

        self.model = PatchCore(
            backbone_name=backbone_name,
            device=device,
        )

        try:
            self.model.load(self.model_path)
        except Exception:
            logger.exception("Failed to load model from %s", self.model_path)
            raise

    def warmup(self, n: int = 10) -> None:
        """Run warmup passes to initialize CUDA context.

        Args:
            n: Number of warmup iterations.
        """
        dummy = torch.randn(1, 3, 224, 224)
        for _ in range(n):
            self.model.predict(dummy)
        logger.info("PyTorch engine warmed up with %d iterations", n)

    def infer(self, image_tensor: torch.Tensor) -> tuple[float, np.ndarray]:
        """Run inference on a single image.

        Args:
            image_tensor: Input tensor of shape [3, 224, 224] or [1, 3, 224, 224].

        Returns:
            Tuple of (anomaly_score, heatmap).
        """
        return self.model.predict(image_tensor)

    def benchmark(
        self,
        dataloader: DataLoader,
        n_runs: int = 200,
    ) -> dict[str, float]:
        """Benchmark inference performance.

        Args:
            dataloader: Test dataloader.
            n_runs: Number of inference runs to measure.

        Returns:
            Dict with mean/p50/p95/p99 latency, FPS, and AUROC.
        """
        latencies = []
        all_scores = []
        all_labels = []
        count = 0

        for images, labels, _, _ in tqdm(dataloader, desc="Benchmarking PyTorch"):
            for i in range(images.shape[0]):
                if count >= n_runs:
                    break

                start = time.perf_counter()
                score, _ = self.infer(images[i])
                elapsed = (time.perf_counter() - start) * 1000  # ms

                latencies.append(elapsed)
                all_scores.append(score)
                all_labels.append(labels[i].item())
                count += 1

            if count >= n_runs:
                break

        latencies_arr = np.array(latencies)
        auroc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else float("nan")

        return {
            "mean_latency_ms": float(np.mean(latencies_arr)),
            "p50_latency_ms": float(np.percentile(latencies_arr, 50)),
            "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
            "p99_latency_ms": float(np.percentile(latencies_arr, 99)),
            "fps": float(1000.0 / np.mean(latencies_arr)),
            "auroc": auroc,
        }
