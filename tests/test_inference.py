"""Tests for inference engines."""

import numpy as np
import pytest
import torch
from pathlib import Path


class TestBackbone:
    """Test feature extractor backbone."""

    def test_efficientnet_output_shape(self) -> None:
        """Test EfficientNet backbone output dimensions."""
        from models.backbone import FeatureExtractor

        model = FeatureExtractor(backbone_name="efficientnet")
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 1
        assert out.shape[1] == model.output_channels  # 40 + 80 = 120
        assert out.shape[2] == 28  # 224 / 8
        assert out.shape[3] == 28

    def test_mobilenet_output_shape(self) -> None:
        """Test MobileNet backbone output dimensions."""
        from models.backbone import FeatureExtractor

        model = FeatureExtractor(backbone_name="mobilenet")
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 1
        assert out.shape[1] == model.output_channels  # 24 + 40 = 64
        assert out.shape[2] == 28
        assert out.shape[3] == 28

    def test_backbone_is_frozen(self) -> None:
        """Test that backbone parameters are frozen."""
        from models.backbone import FeatureExtractor

        model = FeatureExtractor(backbone_name="efficientnet")
        for param in model.parameters():
            assert not param.requires_grad


class TestAutoencoder:
    """Test convolutional autoencoder."""

    def test_reconstruction_shape(self) -> None:
        """Test that autoencoder output matches input shape."""
        from models.autoencoder import ConvAutoencoder

        model = ConvAutoencoder()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        assert out.shape == x.shape

    def test_anomaly_score_shape(self) -> None:
        """Test anomaly score output shapes."""
        from models.autoencoder import ConvAutoencoder

        model = ConvAutoencoder()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            score, heatmap = model.anomaly_score(x)

        assert score.shape == (2,)
        assert heatmap.shape == (2, 224, 224)


class TestEngineOutputConsistency:
    """Test that all engines return consistent output shapes."""

    def test_pytorch_engine_output_shape(self, tmp_path: Path) -> None:
        """Test PyTorch engine output format."""
        from models.patchcore import PatchCore

        # Create a minimal PatchCore with synthetic memory bank
        model = PatchCore(backbone_name="efficientnet", device="cpu")

        # Build a tiny memory bank
        dummy_features = np.random.randn(100, 120).astype(np.float32)
        model.memory_bank = dummy_features
        model._build_faiss_index(model.memory_bank)

        # Save and reload
        save_path = tmp_path / "test_model"
        model.save(save_path)

        from inference.engine_pytorch import PyTorchEngine
        engine = PyTorchEngine(
            model_path=save_path,
            device="cpu",
            backbone_name="efficientnet",
        )

        test_input = torch.randn(3, 224, 224)
        score, heatmap = engine.infer(test_input)

        assert isinstance(score, float)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)

    def test_onnx_engine_output_shape(self, tmp_path: Path) -> None:
        """Test ONNX engine output format."""
        try:
            import onnxruntime
        except ImportError:
            pytest.skip("onnxruntime not installed")

        from export.export_onnx import export_backbone_onnx

        onnx_path = tmp_path / "test.onnx"
        export_backbone_onnx(backbone_name="efficientnet", output_path=onnx_path)

        from inference.engine_onnx import ONNXEngine
        engine = ONNXEngine(model_path=onnx_path, device="cpu")

        test_input = torch.randn(3, 224, 224)
        score, heatmap = engine.infer(test_input)

        assert isinstance(score, float)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)
