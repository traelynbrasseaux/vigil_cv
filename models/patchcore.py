"""PatchCore anomaly detection (Roth et al., 2022).

Memory-bank-based anomaly detection using nearest-neighbor patch feature matching.
"""

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

from models.backbone import FeatureExtractor

logger = logging.getLogger(__name__)


def _try_import_faiss():
    """Import faiss with graceful fallback."""
    try:
        import faiss
        return faiss
    except ImportError:
        logger.warning(
            "faiss not installed. Install faiss-cpu or faiss-gpu for fast kNN. "
            "Falling back to scipy spatial distance."
        )
        return None


class PatchCore:
    """PatchCore anomaly detection model.

    Builds a memory bank of normal patch features from training data,
    then scores test images by nearest-neighbor distance in feature space.

    Args:
        backbone_name: Feature extractor backbone ('efficientnet' or 'mobilenet').
        device: Torch device string.
        coreset_ratio: Fraction of patches to keep via greedy coreset subsampling.
        image_size: Input image spatial dimension (assumes square).
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet",
        device: str = "cuda",
        coreset_ratio: float = 0.1,
        image_size: int = 224,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.coreset_ratio = coreset_ratio
        self.image_size = image_size

        self.backbone = FeatureExtractor(backbone_name=backbone_name).to(self.device)
        self.backbone.eval()

        self.memory_bank: np.ndarray | None = None
        self.faiss_index = None

    @torch.no_grad()
    def _extract_features(self, dataloader: DataLoader) -> np.ndarray:
        """Extract patch-level features from all images in a dataloader.

        Args:
            dataloader: DataLoader yielding (image, label, mask, category) tuples.

        Returns:
            Array of shape [N_patches, C] with all patch features.
        """
        all_features = []

        for images, _, _, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(self.device)
            features = self.backbone(images)  # [B, C, H, W]
            b, c, h, w = features.shape
            # Reshape to patch-level: [B*H*W, C]
            patches = features.permute(0, 2, 3, 1).reshape(-1, c)
            all_features.append(patches.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def _greedy_coreset(self, features: np.ndarray) -> np.ndarray:
        """Greedy coreset subsampling to reduce memory bank size.

        Iteratively selects the point farthest from the current coreset,
        producing a representative subset of the full feature set.

        Args:
            features: Full feature array of shape [N, C].

        Returns:
            Subsampled feature array of shape [M, C] where M = N * coreset_ratio.
        """
        n_samples = features.shape[0]
        n_coreset = max(1, int(n_samples * self.coreset_ratio))

        if n_coreset >= n_samples:
            return features

        logger.info("Coreset subsampling: %d -> %d patches", n_samples, n_coreset)
        start = time.time()

        # Initialize with a random point
        rng = np.random.default_rng(42)
        selected_indices = [rng.integers(n_samples)]
        min_distances = np.full(n_samples, np.inf, dtype=np.float32)

        for _ in tqdm(range(n_coreset - 1), desc="Coreset subsampling"):
            last_selected = features[selected_indices[-1]]
            # Compute distances from all points to the last selected point
            dists = np.linalg.norm(features - last_selected, axis=1).astype(np.float32)
            min_distances = np.minimum(min_distances, dists)
            # Select the point with maximum minimum distance
            next_idx = np.argmax(min_distances)
            selected_indices.append(int(next_idx))

        elapsed = time.time() - start
        logger.info("Coreset subsampling completed in %.1fs", elapsed)

        return features[selected_indices]

    def _build_faiss_index(self, features: np.ndarray) -> None:
        """Build a FAISS index for fast nearest-neighbor search.

        Args:
            features: Memory bank features of shape [M, C].
        """
        faiss = _try_import_faiss()
        if faiss is None:
            self.faiss_index = None
            return

        d = features.shape[1]
        # Try GPU index first, fall back to CPU
        try:
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.GpuIndexFlatL2(res, d)
        except Exception:
            logger.info("FAISS GPU unavailable, using CPU index.")
            self.faiss_index = faiss.IndexFlatL2(d)

        self.faiss_index.add(features.astype(np.float32))
        logger.info("FAISS index built with %d vectors of dimension %d", features.shape[0], d)

    def fit(self, train_loader: DataLoader) -> None:
        """Build the memory bank from training data.

        Args:
            train_loader: DataLoader of normal training images.
        """
        logger.info("Building PatchCore memory bank...")
        features = self._extract_features(train_loader)
        logger.info("Extracted %d patch features with dimension %d", features.shape[0], features.shape[1])

        self.memory_bank = self._greedy_coreset(features)
        self._build_faiss_index(self.memory_bank)

        logger.info("Memory bank ready: %d patches", self.memory_bank.shape[0])

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> tuple[float, np.ndarray]:
        """Compute anomaly score and heatmap for a single image or batch.

        Args:
            image: Input tensor of shape [B, 3, 224, 224] or [3, 224, 224].

        Returns:
            Tuple of (anomaly_score, heatmap).
            - anomaly_score: Maximum patch distance (float).
            - heatmap: Per-pixel anomaly map of shape [H, W] upsampled to image_size.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        features = self.backbone(image)  # [B, C, H, W]
        b, c, h, w = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy().astype(np.float32)

        # Nearest neighbor search
        distances = self._knn_search(patches, k=1)  # [B*H*W]

        # Reshape to spatial map
        score_map = distances.reshape(b, h, w)

        # For batch, take the first image
        heatmap = score_map[0]

        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=4)

        # Upsample to original image size
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
        heatmap_upsampled = F.interpolate(
            heatmap_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        anomaly_score = float(np.max(heatmap_upsampled))
        return anomaly_score, heatmap_upsampled

    def _knn_search(self, queries: np.ndarray, k: int = 1) -> np.ndarray:
        """Find k-nearest neighbor distances.

        Args:
            queries: Query features of shape [N, C].
            k: Number of neighbors.

        Returns:
            Distances of shape [N].
        """
        if self.faiss_index is not None:
            distances, _ = self.faiss_index.search(queries, k)
            return np.sqrt(distances[:, 0])

        # Fallback: brute-force with numpy (slow)
        from scipy.spatial.distance import cdist
        dists = cdist(queries, self.memory_bank, metric="euclidean")
        return np.min(dists, axis=1)

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluate anomaly detection performance on test set.

        Args:
            test_loader: DataLoader of test images with anomaly labels.

        Returns:
            Dictionary with 'image_auroc' and 'pixel_auroc' metrics.
        """
        all_scores = []
        all_labels = []
        all_pixel_scores = []
        all_pixel_labels = []

        for images, labels, masks, _ in tqdm(test_loader, desc="Evaluating"):
            for i in range(images.shape[0]):
                score, heatmap = self.predict(images[i])
                all_scores.append(score)
                all_labels.append(labels[i].item())

                mask_np = masks[i].squeeze().numpy()
                all_pixel_scores.append(heatmap.flatten())
                all_pixel_labels.append(mask_np.flatten())

        image_auroc = roc_auc_score(all_labels, all_scores)

        pixel_scores = np.concatenate(all_pixel_scores)
        pixel_labels = np.concatenate(all_pixel_labels)
        if pixel_labels.max() > 0:
            pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)
        else:
            pixel_auroc = float("nan")

        results = {"image_auroc": image_auroc, "pixel_auroc": pixel_auroc}
        logger.info("Image AUROC: %.4f | Pixel AUROC: %.4f", image_auroc, pixel_auroc)
        return results

    def save(self, path: Path) -> None:
        """Save memory bank and config to disk.

        Args:
            path: Path to save the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "memory_bank": self.memory_bank,
            "backbone_name": self.backbone.backbone_name,
            "coreset_ratio": self.coreset_ratio,
            "image_size": self.image_size,
            "output_channels": self.backbone.output_channels,
        }

        try:
            np.savez_compressed(path.with_suffix(".npz"), **checkpoint)
            logger.info("PatchCore model saved to %s", path.with_suffix(".npz"))
        except Exception:
            logger.exception("Failed to save model to %s", path)
            raise

    def load(self, path: Path) -> None:
        """Load memory bank from disk.

        Args:
            path: Path to the saved checkpoint.
        """
        path = Path(path)
        if not path.with_suffix(".npz").exists():
            raise FileNotFoundError(f"Checkpoint not found: {path.with_suffix('.npz')}")

        try:
            data = np.load(path.with_suffix(".npz"), allow_pickle=True)
            self.memory_bank = data["memory_bank"]
            self._build_faiss_index(self.memory_bank)
            logger.info("Loaded memory bank with %d patches from %s", self.memory_bank.shape[0], path)
        except Exception:
            logger.exception("Failed to load model from %s", path)
            raise
