"""TensorRT INT8 calibration using MVTec AD training images."""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _import_tensorrt():
    """Import TensorRT with a helpful error message."""
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        raise ImportError(
            "TensorRT is not installed. See export/export_tensorrt.py docstring "
            "or README for Windows installation instructions."
        )


def _make_calibrator_class(trt):
    """Create MVTecCalibrator with proper TensorRT base class inheritance.

    Defined inside a function so the base class is resolved after import.
    """

    class MVTecCalibratorImpl(trt.IInt8MinMaxCalibrator):
        def __init__(
            self,
            data_root: Path,
            category: str = "leather",
            batch_size: int = 8,
            n_samples: int = 100,
            image_size: int = 224,
            cache_file: Path | None = None,
        ) -> None:
            super().__init__()

            self.batch_size = batch_size
            self.n_samples = n_samples
            self.image_size = image_size
            self.cache_file = Path(cache_file) if cache_file else Path("exports/calibration.cache")
            self.current_index = 0

            self._load_calibration_data(Path(data_root), category)

            self.device_tensor = torch.zeros(
                self.batch_size, 3, image_size, image_size,
                dtype=torch.float32, device="cuda",
            ).contiguous()

        def _load_calibration_data(self, data_root: Path, category: str) -> None:
            from data.dataset import MVTecDataset

            dataset = MVTecDataset(
                root=data_root,
                category=category,
                split="train",
                resize=256,
                cropsize=self.image_size,
            )

            n_use = min(self.n_samples, len(dataset))
            self.calibration_data = []

            for i in range(n_use):
                image, _, _, _ = dataset[i]
                self.calibration_data.append(image.numpy())

            self.calibration_data = np.array(self.calibration_data, dtype=np.float32)
            logger.info("Loaded %d calibration images from %s/%s", n_use, data_root, category)

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names):
            if self.current_index >= len(self.calibration_data):
                return None

            end_index = min(self.current_index + self.batch_size, len(self.calibration_data))
            batch = self.calibration_data[self.current_index:end_index]

            if batch.shape[0] < self.batch_size:
                padding = np.zeros(
                    (self.batch_size - batch.shape[0], *batch.shape[1:]),
                    dtype=np.float32,
                )
                batch = np.concatenate([batch, padding], axis=0)

            self.device_tensor.copy_(torch.from_numpy(batch))
            self.current_index = end_index

            return [self.device_tensor.data_ptr()]

        def read_calibration_cache(self):
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, "rb") as f:
                        logger.info("Reading calibration cache from %s", self.cache_file)
                        return f.read()
                except Exception:
                    logger.exception("Failed to read calibration cache")
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.cache_file, "wb") as f:
                    f.write(cache)
                logger.info("Calibration cache written to %s", self.cache_file)
            except Exception:
                logger.exception("Failed to write calibration cache")

    return MVTecCalibratorImpl


def MVTecCalibrator(
    data_root: Path,
    category: str = "leather",
    batch_size: int = 8,
    n_samples: int = 100,
    image_size: int = 224,
    cache_file: Path | None = None,
):
    """Factory that returns a properly-subclassed TensorRT INT8 calibrator instance."""
    trt = _import_tensorrt()
    cls = _make_calibrator_class(trt)
    return cls(
        data_root=data_root,
        category=category,
        batch_size=batch_size,
        n_samples=n_samples,
        image_size=image_size,
        cache_file=cache_file,
    )
