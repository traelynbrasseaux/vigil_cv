"""Tests for benchmark functionality."""

import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock

from torch.utils.data import DataLoader, TensorDataset


class TestBenchmarkMechanics:
    """Test benchmark infrastructure without requiring real models."""

    def _make_dummy_dataloader(self, n_samples: int = 20) -> DataLoader:
        """Create a dummy dataloader mimicking MVTec format."""
        images = torch.randn(n_samples, 3, 224, 224)
        labels = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)]).long()
        masks = torch.zeros(n_samples, 1, 224, 224)
        categories = ["test"] * n_samples

        class DummyDataset:
            def __len__(self):
                return n_samples

            def __getitem__(self, idx):
                return images[idx], labels[idx], masks[idx], categories[idx]

        return DataLoader(DummyDataset(), batch_size=4)

    def test_benchmark_csv_output(self, tmp_path: Path) -> None:
        """Test that benchmark produces valid CSV output."""
        from benchmark.run_benchmark import save_results
        import pandas as pd

        results = pd.DataFrame([
            {
                "backend": "pytorch",
                "precision": "fp32",
                "mean_latency_ms": 15.2,
                "p50_latency_ms": 14.8,
                "p95_latency_ms": 18.1,
                "p99_latency_ms": 22.3,
                "fps": 65.8,
                "auroc": 0.9523,
            },
        ])

        save_results(results, tmp_path)

        csv_files = list(tmp_path.glob("benchmark_*.csv"))
        assert len(csv_files) == 1

        loaded = pd.read_csv(csv_files[0])
        assert len(loaded) == 1
        assert "backend" in loaded.columns
        assert "auroc" in loaded.columns

    def test_benchmark_plot_output(self, tmp_path: Path) -> None:
        """Test that benchmark produces PNG plot."""
        from benchmark.run_benchmark import save_results
        import pandas as pd

        results = pd.DataFrame([
            {
                "backend": "pytorch",
                "precision": "fp32",
                "mean_latency_ms": 15.2,
                "p50_latency_ms": 14.8,
                "p95_latency_ms": 18.1,
                "p99_latency_ms": 22.3,
                "fps": 65.8,
                "auroc": 0.9523,
            },
            {
                "backend": "onnx",
                "precision": "fp32",
                "mean_latency_ms": 8.4,
                "p50_latency_ms": 8.1,
                "p95_latency_ms": 10.2,
                "p99_latency_ms": 12.1,
                "fps": 119.0,
                "auroc": 0.9518,
            },
        ])

        save_results(results, tmp_path)

        png_files = list(tmp_path.glob("benchmark_*.png"))
        assert len(png_files) == 1
        assert png_files[0].stat().st_size > 0

    def test_engine_benchmark_method(self, tmp_path: Path) -> None:
        """Test that PyTorch engine benchmark runs on small sample."""
        from models.patchcore import PatchCore

        model = PatchCore(backbone_name="efficientnet", device="cpu")
        dummy_features = np.random.randn(50, 120).astype(np.float32)
        model.memory_bank = dummy_features
        model._build_faiss_index(model.memory_bank)

        save_path = tmp_path / "test_model"
        model.save(save_path)

        from inference.engine_pytorch import PyTorchEngine
        engine = PyTorchEngine(
            model_path=save_path,
            device="cpu",
            backbone_name="efficientnet",
        )

        dataloader = self._make_dummy_dataloader(n_samples=10)
        results = engine.benchmark(dataloader, n_runs=10)

        assert "mean_latency_ms" in results
        assert "fps" in results
        assert "auroc" in results
        assert results["mean_latency_ms"] > 0
        assert results["fps"] > 0
