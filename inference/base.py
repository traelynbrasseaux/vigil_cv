"""Base engine interface for all inference backends."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


class BaseEngine(ABC):
    """Abstract base class for inference engines.

    All inference backends (PyTorch, ONNX, TensorRT) implement this interface
    to enable uniform benchmarking and interchangeable use in the streaming pipeline.
    """

    @abstractmethod
    def __init__(self, model_path: Path, device: str = "cuda") -> None:
        """Initialize the engine.

        Args:
            model_path: Path to the model file.
            device: Device string (e.g. 'cuda', 'cpu').
        """

    @abstractmethod
    def warmup(self, n: int = 10) -> None:
        """Run warmup passes to initialize CUDA context and caches.

        Args:
            n: Number of warmup iterations.
        """

    @abstractmethod
    def infer(self, image_tensor: torch.Tensor) -> tuple[float, np.ndarray]:
        """Run inference on a single image.

        Args:
            image_tensor: Preprocessed input tensor.

        Returns:
            Tuple of (anomaly_score, heatmap_array).
        """

    @abstractmethod
    def benchmark(
        self,
        dataloader: DataLoader,
        n_runs: int = 200,
    ) -> dict[str, float]:
        """Benchmark inference performance.

        Args:
            dataloader: Test dataloader.
            n_runs: Number of inference runs.

        Returns:
            Dict with latency metrics, FPS, and AUROC.
        """
