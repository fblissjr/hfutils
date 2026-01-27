import argparse
import logging
import os
import sys
import time
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import get_collection, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# Attempt to import tqdm for progress visualization
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# --- Configuration & Constants ---
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Configure standard python logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("hfutils")


def enable_performance_optimizations():
    """
    Enables high-performance download settings.
    
    1. Checks for 'hf_transfer' (Rust-based downloader) and enables it via env vars.
    2. Checks for 'hf_xet' for Xet-backed repositories.
    """
    # Enable hf_transfer if available (fastest option for general files)
    if importlib.util.find_spec("hf_transfer"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        logger.info("Performance: hf_transfer enabled (Rust accelerator active).")
    else:
        logger.debug("Performance: hf_transfer not found. Install for faster speeds.")

    # Check for Xet support
    if importlib.util.find_spec("hf_xet"):
        logger.info("Performance: hf_xet detected (Xet protocol active).")


def parse_collection_slug(input_str: str) -> str:
    """
    Parses a HuggingFace collection URL or slug into a 'namespace/slug' format.
    
    Args:
        input_str: The full URL or raw slug provided by the user.
        
    Returns:
        The clean slug string (e.g., 'Qwen/qwen3-tts').
    """
    if "huggingface.co/collections/" in input_str:
        # Strip query parameters and trailing slashes
        clean_url = input_str.split("?")[0].rstrip("/")
        parts = clean_url.split("/")
        # Expected format: .../collections/User/slug
        if len(parts) >= 3:
            return f"{parts[-2]}/{parts[-1]}"
    return input_str


class CollectionDownloader:
    """
    Manages the retrieval and downloading of HuggingFace collections.
    """

    def __init__(
        self,
        base_dir: str,
        token: Optional[str] = None,
        max_workers: int = 4,
        max_retries: int = 3,
        force_download: bool = False
    ):
        """
        Initialize the downloader.

        Args:
            base_dir: Local path to save downloads.
            token: HF API token.
            max_workers: Number of parallel download threads.
            max_retries: Number of retry attempts for failed downloads.
            force_download: If True, ignores local cache and redownloads.
        """
        self.base_dir = Path(base_dir)
        self.token = token
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.force_download = force_download

    def fetch_model_list(self, slug: str) -> List[str]:
        """
        Retrieves the list of model IDs associated with a collection.
        """
        try:
            logger.info(f"Fetching metadata for collection: {slug}")
            collection = get_collection(slug, token=self.token)
            
            # Filter only for items that are actual models (skips datasets/spaces)
            models = [
                item.item_id 
                for item in collection.items 
                if item.item_type == "model"
            ]
            
            if not models:
                logger.warning(f"Collection '{slug}' exists but contains no models.")
                return []
                
            logger.info(f"Found {len(models)} models in '{collection.title}'.")
            return models

        except Exception as e:
            logger.error(f"Failed to fetch collection: {e}")
            sys.exit(1)

    def _download_single_repo(
        self, 
        model_id: str, 
        allow_patterns: Optional[List[str]], 
        ignore_patterns: Optional[List[str]]
    ) -> Tuple[bool, str]:
        """
        Internal worker to download a single repository with retry logic.
        """
        # Create a specific folder for this model to avoid file collisions
        target_dir = self.base_dir / model_id
        
        attempt = 0
        while attempt <= self.max_retries:
            try:
                snapshot_download(
                    repo_id=model_id,
                    repo_type="model",
                    local_dir=target_dir,
                    local_dir_use_symlinks=False, # We want actual files, not symlinks
                    token=self.token,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    force_download=self.force_download,
                    resume_download=(not self.force_download)
                )
                return True, model_id
            
            except (HfHubHTTPError, Exception) as e:
                attempt += 1
                if attempt <= self.max_retries:
                    sleep_time = 2 * attempt # Linear backoff
                    # Log to debug so we don't spam main logs unless verbose
                    logger.debug(f"Retry {attempt}/{self.max_retries} for {model_id} after error: {e}")
                    time.sleep(sleep_time)
                else:
                    return False, f"{model_id}: {str(e)}"
        
        return False, f"{model_id}: Max retries exceeded"

    def download_all(
        self, 
        models: List[str], 
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ):
        """
        Executes the parallel download of all models.
        """
        logger.info(f"Starting download of {len(models)} models to '{self.base_dir}'")
        logger.info(f"Configuration: Workers={self.max_workers}, Retries={self.max_retries}")

        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        failed_models = []
        
        # TQDM configuration for clean progress bars
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map futures to model IDs for tracking
            future_to_model = {
                executor.submit(
                    self._download_single_repo, 
                    model, 
                    allow_patterns, 
                    ignore_patterns
                ): model 
                for model in models
            }
            
            # Use TQDM if available, otherwise just iterate
            iterator = as_completed(future_to_model)
            if HAS_TQDM:
                iterator = tqdm(
                    iterator, 
                    total=len(models), 
                    unit="repo", 
                    bar_format=bar_format,
                    desc="Progress"
                )

            for future in iterator:
                success, result = future.result()
                if not success:
                    # In failure case, result is the error message
                    msg = f"Failed: {result}"
                    if HAS_TQDM:
                        tqdm.write(msg)
                    else:
                        logger.error(msg)
                    failed_models.append(result)
                else:
                    # In success case, result is the model_id
                    if not HAS_TQDM:
                        logger.info(f"Completed: {result}")

        # Final Summary
        print("-" * 40)
        if failed_models:
            logger.error(f"Completed with errors. {len(failed_models)} downloads failed.")
            sys.exit(1)
        else:
            logger.info(f"Successfully downloaded all {len(models)} models.")


def main():
    parser = argparse.ArgumentParser(
        description="High-performance HuggingFace Collection Downloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required Arguments
    parser.add_argument("collection", help="Collection URL or slug (e.g. 'Qwen/qwen3-tts')")
    
    # Path & Auth
    parser.add_argument("-d", "--dir", default=".", help="Base directory for downloads")
    parser.add_argument("--token", help="HuggingFace API Token (optional)")
    
    # Filtering
    parser.add_argument("-f", "--filter", nargs="*", help="Include patterns (e.g. '*.safetensors')")
    parser.add_argument("-e", "--exclude", nargs="*", help="Exclude patterns (e.g. '*.bin')")
    
    # Performance & Behavior
    parser.add_argument("-j", "--workers", type=int, default=4, help="Number of parallel download threads")
    parser.add_argument("--retries", type=int, default=3, help="Max retry attempts per model")
    parser.add_argument("--force", action="store_true", help="Force re-download (ignore local cache)")
    parser.add_argument("--dry-run", action="store_true", help="List models without downloading")
    
    args = parser.parse_args()
    
    # 1. Setup Environment
    enable_performance_optimizations()
    
    # 2. Initialize Downloader
    downloader = CollectionDownloader(
        base_dir=args.dir,
        token=args.token,
        max_workers=args.workers,
        max_retries=args.retries,
        force_download=args.force
    )

    # 3. Resolve Collection
    slug = parse_collection_slug(args.collection)
    models = downloader.fetch_model_list(slug)
    
    if not models:
        return

    # 4. Dry Run Check
    if args.dry_run:
        print(f"\n[DRY RUN] The following {len(models)} models would be downloaded to '{args.dir}':")
        for m in models:
            print(f" - {m}")
        return

    # 5. Execute
    downloader.download_all(
        models=models, 
        allow_patterns=args.filter,
        ignore_patterns=args.exclude
    )


if __name__ == "__main__":
    main()
