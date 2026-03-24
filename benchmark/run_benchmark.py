"""Multi-backend inference benchmarking suite."""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

BACKENDS = ["pytorch", "onnx", "tensorrt"]
PRECISIONS = ["fp32", "fp16", "int8"]
BATCH_SIZES = [1, 4, 8]


def _create_engine(
    backend: str,
    precision: str,
    model_dir: Path,
    memory_bank_path: Path | None,
    backbone: str,
):
    """Create an inference engine for the given backend and precision.

    Args:
        backend: Inference backend name.
        precision: Precision mode.
        model_dir: Directory containing model files.
        memory_bank_path: Path to PatchCore memory bank.
        backbone: Backbone name.

    Returns:
        Engine instance or None if backend is unavailable.
    """
    try:
        if backend == "pytorch":
            if precision != "fp32":
                return None
            from inference.engine_pytorch import PyTorchEngine
            return PyTorchEngine(
                model_path=memory_bank_path,
                backbone_name=backbone,
            )
        elif backend == "onnx":
            if precision != "fp32":
                return None
            from inference.engine_onnx import ONNXEngine
            onnx_path = model_dir / f"{backbone}.onnx"
            if not onnx_path.exists():
                logger.warning("ONNX model not found: %s", onnx_path)
                return None
            return ONNXEngine(
                model_path=onnx_path,
                memory_bank_path=memory_bank_path,
            )
        elif backend == "tensorrt":
            from inference.engine_tensorrt import TensorRTEngine
            trt_path = model_dir / f"{backbone}_{precision}.trt"
            if not trt_path.exists():
                logger.warning("TensorRT engine not found: %s", trt_path)
                return None
            return TensorRTEngine(
                model_path=trt_path,
                memory_bank_path=memory_bank_path,
            )
    except ImportError as e:
        logger.warning("Backend '%s' unavailable: %s", backend, e)
        return None
    except Exception:
        logger.exception("Failed to create engine for %s/%s", backend, precision)
        return None

    return None


def run_benchmark(
    category: str,
    data_root: Path,
    model_dir: Path,
    memory_bank_path: Path,
    backbone: str,
    output_dir: Path,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> pd.DataFrame:
    """Run the full benchmark sweep across backends and precisions.

    Args:
        category: MVTec AD category for evaluation.
        data_root: Root directory of MVTec AD dataset.
        model_dir: Directory containing exported models.
        memory_bank_path: Path to PatchCore memory bank.
        backbone: Backbone name.
        output_dir: Directory for benchmark results.
        n_warmup: Number of warmup runs.
        n_runs: Number of timed runs.

    Returns:
        DataFrame with benchmark results.
    """
    from data.dataset import get_dataloaders

    _, test_loader = get_dataloaders(
        root=data_root,
        category=category,
        batch_size=1,
        num_workers=2,
    )

    results = []

    for backend in BACKENDS:
        for precision in PRECISIONS:
            # Skip invalid combinations
            if backend in ("pytorch", "onnx") and precision != "fp32":
                continue

            logger.info("Benchmarking %s/%s...", backend, precision)

            engine = _create_engine(
                backend=backend,
                precision=precision,
                model_dir=model_dir,
                memory_bank_path=memory_bank_path,
                backbone=backbone,
            )

            if engine is None:
                logger.warning("Skipping %s/%s (engine unavailable)", backend, precision)
                continue

            try:
                engine.warmup(n=n_warmup)
                metrics = engine.benchmark(test_loader, n_runs=n_runs)

                result = {
                    "backend": backend,
                    "precision": precision,
                    "mean_latency_ms": metrics["mean_latency_ms"],
                    "p50_latency_ms": metrics["p50_latency_ms"],
                    "p95_latency_ms": metrics["p95_latency_ms"],
                    "p99_latency_ms": metrics["p99_latency_ms"],
                    "fps": metrics["fps"],
                    "auroc": metrics["auroc"],
                }
                results.append(result)

                logger.info(
                    "%s/%s: mean=%.2fms p95=%.2fms FPS=%.1f AUROC=%.4f",
                    backend,
                    precision,
                    metrics["mean_latency_ms"],
                    metrics["p95_latency_ms"],
                    metrics["fps"],
                    metrics["auroc"],
                )
            except Exception:
                logger.exception("Benchmark failed for %s/%s", backend, precision)

    df = pd.DataFrame(results)
    return df


def save_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Save benchmark results as CSV and plots.

    Args:
        df: Benchmark results DataFrame.
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save CSV
    csv_path = output_dir / f"benchmark_{timestamp}.csv"
    try:
        df.to_csv(csv_path, index=False)
        logger.info("Results saved to %s", csv_path)
    except Exception:
        logger.exception("Failed to save CSV")
        raise

    if df.empty:
        logger.warning("No results to plot.")
        return

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # FPS comparison
    labels = [f"{row['backend']}\n{row['precision']}" for _, row in df.iterrows()]
    fps_values = df["fps"].values
    colors = ["#2196F3" if b == "pytorch" else "#FF9800" if b == "onnx" else "#4CAF50"
              for b in df["backend"]]

    axes[0].bar(range(len(labels)), fps_values, color=colors)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("FPS")
    axes[0].set_title("Inference Throughput by Backend/Precision")
    axes[0].grid(axis="y", alpha=0.3)

    # AUROC comparison
    auroc_values = df["auroc"].values
    axes[1].bar(range(len(labels)), auroc_values, color=colors)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Detection Accuracy by Backend/Precision")
    axes[1].set_ylim(0.8, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plot_path = output_dir / f"benchmark_{timestamp}.png"
    try:
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", plot_path)
    except Exception:
        logger.exception("Failed to save plot")
    finally:
        plt.close(fig)

    # Print markdown table
    print("\n## Benchmark Results\n")
    print("| Backend | Precision | Mean Latency (ms) | P95 Latency (ms) | FPS | AUROC |")
    print("|---------|-----------|-------------------|-------------------|-----|-------|")
    for _, row in df.iterrows():
        print(
            f"| {row['backend']:<7} | {row['precision']:<9} | "
            f"{row['mean_latency_ms']:>17.2f} | {row['p95_latency_ms']:>17.2f} | "
            f"{row['fps']:>3.0f} | {row['auroc']:.4f} |"
        )
    print()


def main() -> None:
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Run multi-backend inference benchmarks")
    parser.add_argument("--category", type=str, default="leather", help="MVTec AD category")
    parser.add_argument("--backbone", type=str, default="efficientnet")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/mvtec_anomaly_detection"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("exports"),
        help="Directory containing exported models",
    )
    parser.add_argument(
        "--memory-bank",
        type=Path,
        required=True,
        help="Path to PatchCore memory bank (.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results"),
    )
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = run_benchmark(
        category=args.category,
        data_root=args.data_root,
        model_dir=args.model_dir,
        memory_bank_path=args.memory_bank,
        backbone=args.backbone,
        output_dir=args.output_dir,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
    )

    save_results(df, args.output_dir)


if __name__ == "__main__":
    main()
