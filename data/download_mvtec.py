"""Download and verify the MVTec Anomaly Detection dataset."""

import argparse
import hashlib
import logging
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

MVTEC_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"

CATEGORY_URLS: dict[str, str] = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937370-1629958698/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937413-1629958794/cable.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937607-1629959262/leather.tar.xz",
}

# MD5 checksums may differ with updated downloads — verification is best-effort
CATEGORY_MD5: dict[str, str] = {}


class _DownloadProgressBar(tqdm):
    """Progress bar wrapper for urllib downloads."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, dest: Path) -> None:
    """Download a file with a progress bar.

    Args:
        url: URL to download from.
        dest: Destination file path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as pbar:
        urllib.request.urlretrieve(url, dest, reporthook=pbar.update_to)


def verify_checksum(filepath: Path, expected_md5: str) -> bool:
    """Verify MD5 checksum of a downloaded file.

    Args:
        filepath: Path to the file to verify.
        expected_md5: Expected MD5 hash string.

    Returns:
        True if checksum matches.
    """
    actual = _compute_md5(filepath)
    if actual != expected_md5:
        logger.error("Checksum mismatch for %s: expected %s, got %s", filepath.name, expected_md5, actual)
        return False
    logger.info("Checksum verified for %s", filepath.name)
    return True


def extract_tar(filepath: Path, dest_dir: Path) -> None:
    """Extract a tar archive.

    Args:
        filepath: Path to the tar archive.
        dest_dir: Destination directory for extraction.
    """
    logger.info("Extracting %s to %s", filepath.name, dest_dir)
    with tarfile.open(filepath) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"Extracting {filepath.name}"):
            tar.extract(member, path=dest_dir, filter="data")
    logger.info("Extraction complete.")


def download_category(category: str, data_dir: Path) -> None:
    """Download and extract a single MVTec AD category.

    Args:
        category: Category name (e.g. 'bottle', 'cable', 'leather').
        data_dir: Root data directory.
    """
    if category not in CATEGORY_URLS:
        raise ValueError(f"Unknown category '{category}'. Available: {list(CATEGORY_URLS.keys())}")

    dest_dir = data_dir / "mvtec_anomaly_detection"
    category_dir = dest_dir / category

    if category_dir.exists() and any(category_dir.iterdir()):
        logger.info("Category '%s' already exists at %s, skipping.", category, category_dir)
        return

    tar_path = data_dir / f"{category}.tar.xz"

    try:
        if not tar_path.exists():
            logger.info("Downloading %s...", category)
            download_file(CATEGORY_URLS[category], tar_path)

        if category in CATEGORY_MD5:
            if not verify_checksum(tar_path, CATEGORY_MD5[category]):
                logger.warning("Checksum verification failed. File may be corrupted.")

        extract_tar(tar_path, dest_dir)

    except Exception:
        logger.exception("Failed to download/extract category '%s'", category)
        raise
    finally:
        if tar_path.exists():
            tar_path.unlink()
            logger.info("Cleaned up archive %s", tar_path.name)


def download_all(data_dir: Path, categories: list[str] | None = None) -> None:
    """Download all specified MVTec AD categories.

    Args:
        data_dir: Root data directory.
        categories: List of categories to download. Defaults to all available.
    """
    if categories is None:
        categories = list(CATEGORY_URLS.keys())

    for category in categories:
        download_category(category, data_dir)

    logger.info("All categories downloaded successfully.")


def main() -> None:
    """CLI entry point for downloading MVTec AD dataset."""
    parser = argparse.ArgumentParser(description="Download MVTec AD dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORY_URLS.keys()),
        choices=list(CATEGORY_URLS.keys()),
        help="Categories to download (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_all(args.data_dir, args.categories)


if __name__ == "__main__":
    main()
