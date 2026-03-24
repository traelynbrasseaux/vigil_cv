"""Export ONNX model to TensorRT engine.

TensorRT 10.x on Windows Setup
==============================
TensorRT for Windows must be installed from the NVIDIA zip package, NOT pip.

Steps:
1. Download TensorRT 10.x GA zip from NVIDIA Developer:
   https://developer.nvidia.com/tensorrt
2. Extract the zip to a directory, e.g. C:\\TensorRT-10.x.x.x
3. Install the Python wheel:
   pip install C:\\TensorRT-10.x.x.x\\python\\tensorrt-10.x.x.x-cp311-none-win_amd64.whl
4. Add the lib directory to your PATH:
   $env:PATH = "C:\\TensorRT-10.x.x.x\\lib;" + $env:PATH
   Or set it permanently in System Environment Variables.
5. Verify installation:
   python -c "import tensorrt; print(tensorrt.__version__)"
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def _import_tensorrt():
    """Import TensorRT with a helpful error message."""
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        raise ImportError(
            "TensorRT is not installed. On Windows, install from the NVIDIA zip package:\n"
            "  1. Download TensorRT 10.x from https://developer.nvidia.com/tensorrt\n"
            "  2. pip install <TRT_ROOT>\\python\\tensorrt-10.x.x.x-cp311-none-win_amd64.whl\n"
            "  3. Add <TRT_ROOT>\\lib to PATH\n"
            "See module docstring or README for detailed instructions."
        )


def build_engine(
    onnx_path: Path,
    output_path: Path,
    precision: Literal["fp32", "fp16", "int8"] = "fp16",
    workspace_gb: int = 4,
    calibrator=None,
    image_size: int = 224,
    max_batch_size: int = 8,
) -> Path:
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to the ONNX model file.
        output_path: Path to save the serialized TensorRT engine.
        precision: Engine precision mode ('fp32', 'fp16', or 'int8').
        workspace_gb: Maximum GPU workspace in GB.
        calibrator: INT8 calibrator instance (required for int8 precision).
        image_size: Input spatial dimension.
        max_batch_size: Maximum batch size for the engine.

    Returns:
        Path to the serialized engine file.
    """
    trt = _import_tensorrt()

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing ONNX model: %s", onnx_path)
    try:
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("ONNX parse error: %s", parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")
    except Exception:
        logger.exception("Failed to read/parse ONNX model")
        raise

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # Set optimization profile for dynamic batch
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 3, image_size, image_size),
        opt=(4, 3, image_size, image_size),
        max=(max_batch_size, 3, image_size, image_size),
    )
    config.add_optimization_profile(profile)

    # Set precision
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            logger.warning("FP16 not supported on this platform, falling back to FP32")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Building with FP16 precision")
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            logger.warning("INT8 not supported on this platform, falling back to FP32")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            if calibrator is not None:
                config.int8_calibrator = calibrator
                logger.info("Building with INT8 precision (with calibration)")
            else:
                logger.warning("INT8 requested but no calibrator provided. Results may be poor.")
    else:
        logger.info("Building with FP32 precision")

    logger.info("Building TensorRT engine (this may take several minutes)...")
    start_time = time.time()

    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT engine build failed")
    except Exception:
        logger.exception("TensorRT engine build failed")
        raise

    build_time = time.time() - start_time

    try:
        with open(output_path, "wb") as f:
            f.write(serialized_engine)
    except Exception:
        logger.exception("Failed to save TensorRT engine")
        raise

    engine_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Engine built in %.1fs", build_time)
    logger.info("Engine saved to %s (%.1f MB)", output_path, engine_size_mb)

    print(f"\nTensorRT Engine Build Summary:")
    print(f"  Precision: {precision.upper()}")
    print(f"  Build time: {build_time:.1f}s")
    print(f"  Engine size: {engine_size_mb:.1f} MB")
    print(f"  Output: {output_path}")

    return output_path


def main() -> None:
    """CLI entry point for TensorRT export."""
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output engine path (default: exports/{onnx_stem}_{precision}.trt)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Engine precision",
    )
    parser.add_argument(
        "--workspace-gb",
        type=int,
        default=4,
        help="Max workspace size in GB (default: 4 for RTX 5070)",
    )
    parser.add_argument(
        "--calibration-data",
        type=Path,
        default=None,
        help="Path to calibration data root (required for INT8)",
    )
    parser.add_argument(
        "--calibration-category",
        type=str,
        default="leather",
        help="Category for calibration data",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max batch size")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output = args.output or Path("exports") / f"{args.onnx.stem}_{args.precision}.trt"

    calibrator = None
    if args.precision == "int8" and args.calibration_data is not None:
        from export.calibration import MVTecCalibrator
        calibrator = MVTecCalibrator(
            data_root=args.calibration_data,
            category=args.calibration_category,
            batch_size=8,
            n_samples=100,
            image_size=args.image_size,
        )

    build_engine(
        onnx_path=args.onnx,
        output_path=output,
        precision=args.precision,
        workspace_gb=args.workspace_gb,
        calibrator=calibrator,
        image_size=args.image_size,
        max_batch_size=args.max_batch_size,
    )


if __name__ == "__main__":
    main()
