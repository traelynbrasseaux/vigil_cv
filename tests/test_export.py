"""Tests for ONNX export functionality."""

import numpy as np
import pytest
import torch
from pathlib import Path


@pytest.fixture
def tmp_export_dir(tmp_path: Path) -> Path:
    """Create a temporary export directory."""
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    return export_dir


class TestONNXExport:
    """Test ONNX export and verification."""

    def test_export_efficientnet(self, tmp_export_dir: Path) -> None:
        """Test that EfficientNet backbone exports to ONNX successfully."""
        from export.export_onnx import export_backbone_onnx

        output_path = tmp_export_dir / "efficientnet.onnx"
        result = export_backbone_onnx(
            backbone_name="efficientnet",
            output_path=output_path,
            image_size=224,
        )

        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_mobilenet(self, tmp_export_dir: Path) -> None:
        """Test that MobileNet backbone exports to ONNX successfully."""
        from export.export_onnx import export_backbone_onnx

        output_path = tmp_export_dir / "mobilenet.onnx"
        result = export_backbone_onnx(
            backbone_name="mobilenet",
            output_path=output_path,
            image_size=224,
        )

        assert result.exists()
        assert result.stat().st_size > 0

    def test_onnx_output_matches_pytorch(self, tmp_export_dir: Path) -> None:
        """Verify ONNX output matches PyTorch within tolerance."""
        from models.backbone import FeatureExtractor

        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        from export.export_onnx import export_backbone_onnx

        backbone_name = "efficientnet"
        output_path = tmp_export_dir / f"{backbone_name}.onnx"
        export_backbone_onnx(backbone_name=backbone_name, output_path=output_path)

        # PyTorch inference
        model = FeatureExtractor(backbone_name=backbone_name)
        model.eval()
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            pytorch_output = model(test_input).numpy()

        # ONNX inference
        session = ort.InferenceSession(str(output_path))
        onnx_output = session.run(None, {"input": test_input.numpy()})[0]

        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        assert max_diff < 1e-4, f"Output mismatch: max diff = {max_diff}"

    def test_onnx_dynamic_batch(self, tmp_export_dir: Path) -> None:
        """Test that ONNX model supports dynamic batch sizes."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        from export.export_onnx import export_backbone_onnx

        output_path = tmp_export_dir / "efficientnet.onnx"
        export_backbone_onnx(backbone_name="efficientnet", output_path=output_path)

        session = ort.InferenceSession(str(output_path))

        for batch_size in [1, 2, 4]:
            test_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            outputs = session.run(None, {"input": test_input})
            assert outputs[0].shape[0] == batch_size
