"""Training entry point for PatchCore and autoencoder models."""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.autoencoder import ConvAutoencoder
from models.patchcore import PatchCore
from training.loss import ReconstructionLoss

logger = logging.getLogger(__name__)


def train_patchcore(
    category: str,
    backbone: str,
    data_root: Path,
    checkpoint_dir: Path,
    coreset_ratio: float,
    device: str,
) -> dict[str, float]:
    """Train PatchCore model (single forward pass, no gradients).

    Args:
        category: MVTec AD category.
        backbone: Backbone name ('efficientnet' or 'mobilenet').
        data_root: Root directory of MVTec AD dataset.
        checkpoint_dir: Directory to save checkpoints.
        coreset_ratio: Fraction of patches for coreset subsampling.
        device: Torch device.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("Training PatchCore on '%s' with %s backbone", category, backbone)

    train_loader, test_loader = get_dataloaders(
        root=data_root, category=category, batch_size=32, num_workers=4,
    )

    model = PatchCore(
        backbone_name=backbone,
        device=device,
        coreset_ratio=coreset_ratio,
    )

    start_time = time.time()
    model.fit(train_loader)
    fit_time = time.time() - start_time
    logger.info("Memory bank built in %.1fs", fit_time)

    results = model.evaluate(test_loader)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"{category}_{backbone}_{timestamp}"
    model.save(checkpoint_path)

    print(f"\nPatchCore Results ({category}, {backbone}):")
    print(f"  Image AUROC: {results['image_auroc']:.4f}")
    print(f"  Pixel AUROC: {results['pixel_auroc']:.4f}")
    print(f"  Training time: {fit_time:.1f}s")
    print(f"  Checkpoint: {checkpoint_path}.npz")

    return results


def train_autoencoder(
    category: str,
    data_root: Path,
    checkpoint_dir: Path,
    epochs: int,
    lr: float,
    device: str,
    patience: int,
) -> dict[str, float]:
    """Train convolutional autoencoder with early stopping.

    Args:
        category: MVTec AD category.
        data_root: Root directory of MVTec AD dataset.
        checkpoint_dir: Directory to save checkpoints.
        epochs: Maximum number of training epochs.
        lr: Learning rate.
        device: Torch device.
        patience: Early stopping patience (epochs without improvement).

    Returns:
        Dictionary of training metrics.
    """
    logger.info("Training autoencoder on '%s' for up to %d epochs", category, epochs)

    train_loader, test_loader = get_dataloaders(
        root=data_root, category=category, batch_size=32, num_workers=4,
    )

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device_obj)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = ReconstructionLoss()

    best_loss = float("inf")
    epochs_without_improvement = 0
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"{category}_autoencoder_{timestamp}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, _, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device_obj)
            # Denormalize to [0, 1] for reconstruction target
            target = _denormalize(images)

            reconstruction = model(images)
            loss = criterion(reconstruction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d - Loss: %.6f - LR: %.2e", epoch, epochs, avg_loss, scheduler.get_last_lr()[0])

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            try:
                torch.save(model.state_dict(), checkpoint_path)
            except Exception:
                logger.exception("Failed to save checkpoint")
                raise
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
            break

    # Evaluate
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device_obj)
            scores, _ = model.anomaly_score(images)
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.tolist())

    from sklearn.metrics import roc_auc_score
    image_auroc = roc_auc_score(all_labels, all_scores)

    results = {"image_auroc": image_auroc, "best_loss": best_loss}
    print(f"\nAutoencoder Results ({category}):")
    print(f"  Image AUROC: {image_auroc:.4f}")
    print(f"  Best Loss: {best_loss:.6f}")
    print(f"  Checkpoint: {checkpoint_path}")

    return results


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization to [0, 1] range.

    Args:
        tensor: Normalized tensor [B, 3, H, W].

    Returns:
        Denormalized tensor clamped to [0, 1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument("--category", type=str, default="leather", help="MVTec AD category")
    parser.add_argument(
        "--model",
        type=str,
        default="patchcore",
        choices=["patchcore", "autoencoder"],
        help="Model to train",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "mobilenet"],
        help="Feature extractor backbone (PatchCore only)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (autoencoder only)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (autoencoder only)")
    parser.add_argument("--coreset-ratio", type=float, default=0.1, help="Coreset ratio (PatchCore only)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (autoencoder only)")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/mvtec_anomaly_detection"),
        help="Root directory of MVTec AD dataset",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Torch device: 'auto' detects GPU, 'cuda' forces GPU, 'cpu' forces CPU",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        logger.warning("For GPU support, install PyTorch with CUDA:")
        logger.warning("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        args.device = "cpu"
    logger.info("Using device: %s", args.device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "patchcore":
        train_patchcore(
            category=args.category,
            backbone=args.backbone,
            data_root=args.data_root,
            checkpoint_dir=args.checkpoint_dir,
            coreset_ratio=args.coreset_ratio,
            device=args.device,
        )
    else:
        train_autoencoder(
            category=args.category,
            data_root=args.data_root,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
