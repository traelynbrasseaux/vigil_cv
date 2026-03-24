"""TensorRT INT8 calibration using MVTec AD training images."""

import logging
from pathlib import Path

import numpy as np

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


class MVTecCalibrator:
    """TensorRT INT8 calibrator using MVTec AD training images.

    Feeds batches of normal training images to TensorRT's calibration process
    to determine optimal INT8 quantization ranges.

    Args:
        data_root: Root directory of MVTec AD dataset.
        category: Product category for calibration images.
        batch_size: Calibration batch size.
        n_samples: Number of calibration samples to use.
        image_size: Input image spatial dimension.
        cache_file: Path to save/load calibration cache.
    """

    def __init__(
        self,
        data_root: Path,
        category: str = "leather",
        batch_size: int = 8,
        n_samples: int = 100,
        image_size: int = 224,
        cache_file: Path | None = None,
    ) -> None:
        trt = _import_tensorrt()
        # Inherit from IInt8MinMaxCalibrator at runtime
        self._trt = trt
        self.__class__ = type(
            "MVTecCalibrator",
            (trt.IInt8MinMaxCalibrator, ),
            dict(self.__class__.__dict__),
        )
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.image_size = image_size
        self.cache_file = Path(cache_file) if cache_file else Path("exports/calibration.cache")
        self.current_index = 0

        self._load_calibration_data(Path(data_root), category)

        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        self.device_input = cuda.mem_alloc(
            self.batch_size * 3 * image_size * image_size * 4  # float32
        )

    def _load_calibration_data(self, data_root: Path, category: str) -> None:
        """Load and preprocess calibration images.

        Args:
            data_root: Root directory of MVTec AD dataset.
            category: Product category.
        """
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
        """Return the calibration batch size."""
        return self.batch_size

    def get_batch(self, names: list[str]) -> list | None:
        """Get the next batch of calibration data.

        Args:
            names: List of input tensor names.

        Returns:
            List of device pointers, or None if no more batches.
        """
        if self.current_index >= len(self.calibration_data):
            return None

        import pycuda.driver as cuda

        end_index = min(self.current_index + self.batch_size, len(self.calibration_data))
        batch = self.calibration_data[self.current_index:end_index]

        # Pad batch if needed
        if batch.shape[0] < self.batch_size:
            padding = np.zeros(
                (self.batch_size - batch.shape[0], *batch.shape[1:]),
                dtype=np.float32,
            )
            batch = np.concatenate([batch, padding], axis=0)

        cuda.memcpy_htod(self.device_input, batch.ravel())
        self.current_index = end_index

        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        """Read calibration cache from disk if available.

        Returns:
            Cache bytes or None.
        """
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    logger.info("Reading calibration cache from %s", self.cache_file)
                    return f.read()
            except Exception:
                logger.exception("Failed to read calibration cache")
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache to disk.

        Args:
            cache: Calibration cache bytes.
        """
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "wb") as f:
                f.write(cache)
            logger.info("Calibration cache written to %s", self.cache_file)
        except Exception:
            logger.exception("Failed to write calibration cache")
