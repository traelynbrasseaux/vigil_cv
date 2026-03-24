"""Export PatchCore backbone to ONNX format."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def export_backbone_onnx(
    backbone_name: str,
    output_path: Path,
    image_size: int = 224,
    opset_version: int = 17,
) -> Path:
    """Export feature extractor backbone to ONNX.

    Args:
        backbone_name: Backbone name ('efficientnet' or 'mobilenet').
        output_path: Path for the exported ONNX model.
        image_size: Input spatial dimension.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    from models.backbone import FeatureExtractor

    logger.info("Exporting %s backbone to ONNX...", backbone_name)

    model = FeatureExtractor(backbone_name=backbone_name)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["features"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "features": {0: "batch_size"},
            },
        )
        logger.info("ONNX model exported to %s", output_path)
    except Exception:
        logger.exception("Failed to export ONNX model")
        raise

    # Run shape inference and simplification
    _simplify_onnx(output_path)

    # Verify output matches PyTorch
    _verify_onnx(model, output_path, image_size)

    return output_path


def _simplify_onnx(model_path: Path) -> None:
    """Run onnxsim simplification on the exported model.

    Args:
        model_path: Path to the ONNX model file.
    """
    try:
        import onnx
        from onnxsim import simplify

        logger.info("Running ONNX simplification...")
        model = onnx.load(str(model_path))
        model_simplified, check = simplify(model)
        if check:
            onnx.save(model_simplified, str(model_path))
            logger.info("ONNX model simplified successfully.")
        else:
            logger.warning("ONNX simplification check failed, keeping original model.")
    except ImportError:
        logger.warning("onnxsim not installed. Skipping simplification.")
    except Exception:
        logger.exception("ONNX simplification failed")


def _verify_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    image_size: int,
    tolerance: float = 5e-4,
) -> None:
    """Verify ONNX output matches PyTorch within tolerance.

    Args:
        pytorch_model: Original PyTorch model.
        onnx_path: Path to the exported ONNX model.
        image_size: Input spatial dimension.
        tolerance: Maximum allowed difference.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed. Skipping verification.")
        return

    logger.info("Verifying ONNX output against PyTorch...")

    test_input = torch.randn(1, 3, image_size, image_size)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()

    # ONNX inference
    try:
        session = ort.InferenceSession(str(onnx_path))
        ort_output = session.run(None, {"input": test_input.numpy()})[0]
    except Exception:
        logger.exception("ONNX Runtime inference failed")
        raise

    max_diff = np.max(np.abs(pytorch_output - ort_output))
    logger.info("Max difference between PyTorch and ONNX: %.6e", max_diff)

    if max_diff > tolerance:
        logger.error(
            "ONNX verification FAILED: max diff %.6e exceeds tolerance %.6e",
            max_diff,
            tolerance,
        )
        raise ValueError(f"ONNX output mismatch: max diff {max_diff:.6e} > {tolerance:.6e}")

    logger.info("ONNX verification PASSED (max diff: %.6e)", max_diff)


def main() -> None:
    """CLI entry point for ONNX export."""
    parser = argparse.ArgumentParser(description="Export backbone to ONNX")
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "mobilenet"],
        help="Backbone to export",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path (default: exports/{backbone}.onnx)",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output = args.output or Path("exports") / f"{args.backbone}.onnx"

    export_backbone_onnx(
        backbone_name=args.backbone,
        output_path=output,
        image_size=args.image_size,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
