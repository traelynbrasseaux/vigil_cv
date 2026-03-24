"""ONNX Runtime inference engine for anomaly detection."""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference.base import BaseEngine

logger = logging.getLogger(__name__)


class ONNXEngine(BaseEngine):
    """ONNX Runtime inference engine.

    Runs the backbone feature extractor via ONNX Runtime and performs
    nearest-neighbor scoring against a PatchCore memory bank.

    Args:
        model_path: Path to the ONNX model file.
        device: Device string ('cuda' or 'cpu').
        memory_bank_path: Path to PatchCore memory bank (.npz).
        image_size: Input image spatial dimension.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        memory_bank_path: Path | None = None,
        image_size: int = 224,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime-gpu is required. Install with: pip install onnxruntime-gpu")

        self.model_path = Path(model_path)
        self.device = device
        self.image_size = image_size

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Configure session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception:
            logger.exception("Failed to create ONNX Runtime session")
            raise

        self.input_name = self.session.get_inputs()[0].name
        logger.info("ONNX session created with providers: %s", self.session.get_providers())

        # Load memory bank for scoring
        self.memory_bank: np.ndarray | None = None
        if memory_bank_path is not None:
            self._load_memory_bank(Path(memory_bank_path))

    def _load_memory_bank(self, path: Path) -> None:
        """Load PatchCore memory bank from disk.

        Args:
            path: Path to the .npz checkpoint.
        """
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"Memory bank not found: {npz_path}")

        try:
            data = np.load(npz_path, allow_pickle=True)
            self.memory_bank = data["memory_bank"].astype(np.float32)
            self._build_faiss_index()
            logger.info("Loaded memory bank: %s", self.memory_bank.shape)
        except Exception:
            logger.exception("Failed to load memory bank")
            raise

    def _build_faiss_index(self) -> None:
        """Build a FAISS index for fast kNN on the memory bank."""
        self.faiss_index = None
        try:
            import faiss
            d = self.memory_bank.shape[1]
            self.faiss_index = faiss.IndexFlatL2(d)
            self.faiss_index.add(self.memory_bank)
            logger.info("FAISS index built with %d vectors", self.memory_bank.shape[0])
        except ImportError:
            logger.warning("faiss not installed, falling back to scipy cdist (slow)")

    def warmup(self, n: int = 10) -> None:
        """Run warmup passes.

        Args:
            n: Number of warmup iterations.
        """
        dummy = np.random.randn(1, 3, self.image_size, self.image_size).astype(np.float32)
        for _ in range(n):
            self.session.run(None, {self.input_name: dummy})
        logger.info("ONNX engine warmed up with %d iterations", n)

    def infer(self, image_tensor: torch.Tensor) -> tuple[float, np.ndarray]:
        """Run inference on a single image.

        Args:
            image_tensor: Input tensor [3, 224, 224] or [1, 3, 224, 224].

        Returns:
            Tuple of (anomaly_score, heatmap).
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        input_np = image_tensor.numpy().astype(np.float32)
        features = self.session.run(None, {self.input_name: input_np})[0]  # [1, C, H, W]

        if self.memory_bank is None:
            # Without memory bank, return feature magnitude as anomaly proxy
            heatmap = np.mean(features[0] ** 2, axis=0)
            heatmap = gaussian_filter(heatmap, sigma=4)
            heatmap_up = F.interpolate(
                torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
            return float(np.max(heatmap_up)), heatmap_up

        # PatchCore scoring with memory bank
        b, c, h, w = features.shape
        patches = features.transpose(0, 2, 3, 1).reshape(-1, c).astype(np.float32)

        # kNN search (FAISS if available, scipy fallback)
        if self.faiss_index is not None:
            distances, _ = self.faiss_index.search(patches, 1)
            min_dists = np.sqrt(distances[:, 0])
        else:
            from scipy.spatial.distance import cdist
            dists = cdist(patches, self.memory_bank, metric="euclidean")
            min_dists = np.min(dists, axis=1)

        score_map = min_dists.reshape(h, w)
        score_map = gaussian_filter(score_map, sigma=4)

        heatmap_up = F.interpolate(
            torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0).float(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        return float(np.max(heatmap_up)), heatmap_up

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
        latencies = []
        all_scores = []
        all_labels = []
        count = 0

        for images, labels, _, _ in tqdm(dataloader, desc="Benchmarking ONNX"):
            for i in range(images.shape[0]):
                if count >= n_runs:
                    break

                start = time.perf_counter()
                score, _ = self.infer(images[i])
                elapsed = (time.perf_counter() - start) * 1000

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
