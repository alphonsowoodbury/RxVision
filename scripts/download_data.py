#!/usr/bin/env python3
"""
RxVision25 Data Acquisition Script

Downloads NIH RxImage dataset from the National Library of Medicine FTP server.
Modernized version with progress tracking, robust error handling, and efficient processing.

Usage:
    python scripts/download_data.py --help
    python scripts/download_data.py --sample  # Download sample dataset
    python scripts/download_data.py --full    # Download full dataset
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from ftplib import FTP, error_perm
from pathlib import Path
import shutil
import time
from typing import List, Dict, Optional, Tuple
import json
from tqdm import tqdm
import hashlib
from PIL import Image, ImageFile
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import rawpy
import imageio
from datetime import datetime

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RxImageDownloader:
    """NIH RxImage dataset downloader with modern features"""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "val"
        self.test_dir = self.base_dir / "test"

        # NIH FTP configuration
        self.ftp_host = "lhcftp.nlm.nih.gov"
        self.ftp_path = "Open-Access-Datasets/Pills/"

        # Create directories
        self._create_directories()

        # Download statistics
        self.stats = {
            'total_files': 0,
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'converted': 0,
            'start_time': None,
            'end_time': None
        }

    def _create_directories(self):
        """Create all necessary directories"""
        for directory in [self.raw_dir, self.processed_dir, self.train_dir,
                         self.val_dir, self.test_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory structure in {self.base_dir}")

    def download_directory_file(self) -> pd.DataFrame:
        """Download and parse the directory file from NIH FTP"""
        logger.info("Downloading directory file from NIH FTP server...")

        directory_file = self.base_dir / "directory_of_images.txt"

        try:
            with FTP(self.ftp_host) as ftp:
                ftp.login()
                ftp.cwd(self.ftp_path)

                with open(directory_file, 'wb') as f:
                    ftp.retrbinary('RETR directory_of_images.txt', f.write)

                logger.info(f"Directory file downloaded to {directory_file}")

        except Exception as e:
            logger.error(f"Failed to download directory file: {e}")
            raise

        # Parse directory file
        try:
            df = pd.read_csv(
                directory_file,
                sep='|',
                names=['NDC', 'PART_NUM', 'FILE', 'TYPE', 'DRUG'],
                dtype={'NDC': str, 'PART_NUM': str}
            )

            # Clean and process data
            df = df.dropna()
            df['DRUG'] = df['DRUG'].str.upper().str.strip()
            df[['ORIG_FOLDER', 'IMAGES', 'FILENAME']] = df['FILE'].str.split('/', expand=True)
            df['FILETYPE'] = df['FILENAME'].str[-4:].str.upper()

            # Filter out video files
            df = df[df['FILETYPE'] != '.WMV']

            logger.info(f"Loaded directory with {len(df):,} image entries")
            logger.info(f"Found {df['NDC'].nunique():,} unique NDCs")
            logger.info(f"Found {df['DRUG'].nunique():,} unique drugs")

            return df

        except Exception as e:
            logger.error(f"Failed to parse directory file: {e}")
            raise

    def get_sample_dataset(self, target_classes: int = 15, min_images_per_class: int = 30) -> pd.DataFrame:
        """Get a balanced sample dataset for development/testing"""
        logger.info(f"Creating sample dataset with {target_classes} classes, {min_images_per_class}+ images each")

        df = self.download_directory_file()

        # Get NDCs with sufficient images
        ndc_counts = df['NDC'].value_counts()
        suitable_ndcs = ndc_counts[ndc_counts >= min_images_per_class].head(target_classes)

        # Filter dataset
        sample_df = df[df['NDC'].isin(suitable_ndcs.index)].copy()

        # Limit to min_images_per_class per NDC for balanced dataset
        balanced_dfs = []
        for ndc in suitable_ndcs.index:
            ndc_data = sample_df[sample_df['NDC'] == ndc].head(min_images_per_class)
            balanced_dfs.append(ndc_data)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)

        logger.info(f"Sample dataset created:")
        logger.info(f"  Total images: {len(balanced_df):,}")
        logger.info(f"  Classes (NDCs): {balanced_df['NDC'].nunique()}")
        logger.info(f"  Images per class: {len(balanced_df) // balanced_df['NDC'].nunique()}")

        return balanced_df

    def get_full_dataset(self, max_classes: Optional[int] = None,
                        max_images_per_class: Optional[int] = None) -> pd.DataFrame:
        """Get the full dataset or a large subset"""
        logger.info("Preparing full dataset...")

        df = self.download_directory_file()

        if max_classes:
            # Get most represented classes
            ndc_counts = df['NDC'].value_counts().head(max_classes)
            df = df[df['NDC'].isin(ndc_counts.index)]
            logger.info(f"Limited to top {max_classes} classes")

        if max_images_per_class:
            # Limit images per class
            limited_dfs = []
            for ndc in df['NDC'].unique():
                ndc_data = df[df['NDC'] == ndc].head(max_images_per_class)
                limited_dfs.append(ndc_data)
            df = pd.concat(limited_dfs, ignore_index=True)
            logger.info(f"Limited to {max_images_per_class} images per class")

        logger.info(f"Full dataset prepared:")
        logger.info(f"  Total images: {len(df):,}")
        logger.info(f"  Classes (NDCs): {df['NDC'].nunique()}")

        return df

    def _create_ftp_directory_map(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Create mapping of FTP directories to files for efficient downloading"""
        ftp_map = {}

        for _, row in df.iterrows():
            ftp_path_parts = row['FILE'].split('/')
            ftp_dir = '/'.join(ftp_path_parts[:-1]) + '/'
            filename = ftp_path_parts[-1]

            if ftp_dir not in ftp_map:
                ftp_map[ftp_dir] = []
            ftp_map[ftp_dir].append(filename)

        return ftp_map

    def download_images(self, df: pd.DataFrame, max_workers: int = 4) -> None:
        """Download images with parallel processing and progress tracking"""
        logger.info(f"Starting download of {len(df):,} images...")

        self.stats['total_files'] = len(df)
        self.stats['start_time'] = datetime.now()

        # Create FTP directory mapping
        ftp_map = self._create_ftp_directory_map(df)

        # Get already downloaded files
        existing_files = set(f.name for f in self.raw_dir.glob('*') if f.is_file())

        # Download with progress bar
        with tqdm(total=len(df), desc="Downloading") as pbar:
            for ftp_dir, filenames in ftp_map.items():
                self._download_from_directory(ftp_dir, filenames, existing_files, pbar)

        self.stats['end_time'] = datetime.now()
        self._log_download_stats()

    def _download_from_directory(self, ftp_dir: str, filenames: List[str],
                                existing_files: set, pbar: tqdm) -> None:
        """Download files from a single FTP directory"""
        try:
            with FTP(self.ftp_host) as ftp:
                ftp.login()
                ftp.cwd(self.ftp_path)
                ftp.cwd(ftp_dir)

                for filename in filenames:
                    if filename in existing_files:
                        self.stats['skipped'] += 1
                        pbar.update(1)
                        continue

                    try:
                        local_path = self.raw_dir / filename
                        with open(local_path, 'wb') as f:
                            ftp.retrbinary(f'RETR {filename}', f.write)

                        self.stats['downloaded'] += 1
                        pbar.update(1)

                    except error_perm as e:
                        logger.warning(f"Could not download {filename}: {e}")
                        self.stats['failed'] += 1
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error downloading {filename}: {e}")
                        self.stats['failed'] += 1
                        pbar.update(1)

        except Exception as e:
            logger.error(f"Error accessing FTP directory {ftp_dir}: {e}")
            self.stats['failed'] += len(filenames)
            pbar.update(len(filenames))

    def _log_download_stats(self):
        """Log download statistics"""
        duration = self.stats['end_time'] - self.stats['start_time']

        logger.info("Download completed!")
        logger.info(f"  Total files: {self.stats['total_files']:,}")
        logger.info(f"  Downloaded: {self.stats['downloaded']:,}")
        logger.info(f"  Skipped (existing): {self.stats['skipped']:,}")
        logger.info(f"  Failed: {self.stats['failed']:,}")
        logger.info(f"  Duration: {duration}")

        if self.stats['downloaded'] > 0:
            rate = self.stats['downloaded'] / duration.total_seconds()
            logger.info(f"  Download rate: {rate:.2f} files/second")

    def convert_images(self, target_format: str = 'JPG',
                      target_size: Optional[Tuple[int, int]] = None,
                      quality: int = 95) -> None:
        """Convert raw images to standard format"""
        logger.info(f"Converting images to {target_format} format...")

        raw_files = list(self.raw_dir.glob('*'))

        with tqdm(total=len(raw_files), desc="Converting") as pbar:
            for raw_file in raw_files:
                if raw_file.is_file():
                    self._convert_single_image(raw_file, target_format, target_size, quality)
                pbar.update(1)

        logger.info(f"Converted {self.stats['converted']:,} images")

    def _convert_single_image(self, raw_path: Path, target_format: str,
                            target_size: Optional[Tuple[int, int]], quality: int) -> None:
        """Convert a single image file"""
        try:
            processed_path = self.processed_dir / f"{raw_path.stem}.{target_format.lower()}"

            # Skip if already processed
            if processed_path.exists():
                return

            file_ext = raw_path.suffix.upper()

            if file_ext in ['.PNG', '.JPG', '.JPEG']:
                # Standard image formats
                with Image.open(raw_path) as img:
                    rgb_img = img.convert('RGB')

                    if target_size:
                        rgb_img = rgb_img.resize(target_size, Image.Resampling.LANCZOS)

                    rgb_img.save(processed_path, target_format, quality=quality, optimize=True)

            elif file_ext == '.CR2':
                # Canon RAW format
                if raw_path.stat().st_size > 0:  # Check for corrupted files
                    raw = rawpy.imread(str(raw_path))
                    rgb = raw.postprocess()

                    if target_size:
                        rgb = cv2.resize(rgb, target_size)

                    imageio.imsave(str(processed_path), rgb)
                else:
                    logger.warning(f"Skipping corrupted CR2 file: {raw_path}")
                    return
            else:
                logger.warning(f"Unsupported format: {file_ext}")
                return

            self.stats['converted'] += 1

        except Exception as e:
            logger.error(f"Failed to convert {raw_path}: {e}")

    def create_splits(self, df: pd.DataFrame, train_ratio: float = 0.7,
                     val_ratio: float = 0.15, test_ratio: float = 0.15) -> None:
        """Create train/validation/test splits"""
        logger.info("Creating train/validation/test splits...")

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        # Group by NDC for stratified splitting
        split_stats = {'train': 0, 'val': 0, 'test': 0}

        for ndc in df['NDC'].unique():
            ndc_data = df[df['NDC'] == ndc]
            n_total = len(ndc_data)

            # Calculate split sizes
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val

            # Create class directories
            for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
                class_dir = split_dir / ndc
                class_dir.mkdir(exist_ok=True)

            # Copy files to appropriate splits
            filenames = ndc_data['FILENAME'].tolist()

            # Train set
            for i in range(n_train):
                self._copy_to_split(filenames[i], self.train_dir / ndc)
            split_stats['train'] += n_train

            # Validation set
            for i in range(n_train, n_train + n_val):
                self._copy_to_split(filenames[i], self.val_dir / ndc)
            split_stats['val'] += n_val

            # Test set
            for i in range(n_train + n_val, n_total):
                self._copy_to_split(filenames[i], self.test_dir / ndc)
            split_stats['test'] += n_test

        logger.info("Dataset splits created:")
        logger.info(f"  Train: {split_stats['train']:,} images")
        logger.info(f"  Validation: {split_stats['val']:,} images")
        logger.info(f"  Test: {split_stats['test']:,} images")

        # Save split information
        self._save_split_info(df, split_stats)

    def _copy_to_split(self, filename: str, destination_dir: Path) -> None:
        """Copy processed image to split directory"""
        # Try different extensions
        for ext in ['jpg', 'jpeg', 'png']:
            source_path = self.processed_dir / f"{filename.split('.')[0]}.{ext}"
            if source_path.exists():
                destination_path = destination_dir / source_path.name
                shutil.copy2(source_path, destination_path)
                break

    def _save_split_info(self, df: pd.DataFrame, split_stats: Dict[str, int]) -> None:
        """Save dataset and split information"""
        info = {
            'dataset_info': {
                'total_images': len(df),
                'num_classes': df['NDC'].nunique(),
                'creation_date': datetime.now().isoformat(),
                'source': 'NIH RxImage Dataset'
            },
            'class_info': {
                str(ndc): {
                    'drug_name': df[df['NDC'] == ndc]['DRUG'].iloc[0],
                    'image_count': int((df['NDC'] == ndc).sum())
                }
                for ndc in df['NDC'].unique()
            },
            'split_stats': split_stats,
            'download_stats': self.stats
        }

        info_path = self.base_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        logger.info(f"Dataset information saved to {info_path}")

    def cleanup_raw_files(self) -> None:
        """Remove raw files to save space (optional)"""
        logger.info("Cleaning up raw files...")

        raw_files = list(self.raw_dir.glob('*'))
        for raw_file in raw_files:
            if raw_file.is_file():
                raw_file.unlink()

        logger.info(f"Removed {len(raw_files)} raw files")

def main():
    parser = argparse.ArgumentParser(description='Download NIH RxImage dataset')
    parser.add_argument('--sample', action='store_true',
                       help='Download sample dataset (15 classes, 30 images each)')
    parser.add_argument('--full', action='store_true',
                       help='Download full dataset')
    parser.add_argument('--classes', type=int, default=15,
                       help='Number of classes for sample dataset')
    parser.add_argument('--images-per-class', type=int, default=30,
                       help='Images per class for sample dataset')
    parser.add_argument('--max-classes', type=int,
                       help='Maximum classes for full dataset')
    parser.add_argument('--max-images-per-class', type=int,
                       help='Maximum images per class for full dataset')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Base directory for dataset')
    parser.add_argument('--target-size', type=str,
                       help='Target image size (e.g., "224,224")')
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove raw files after processing')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of download workers')

    args = parser.parse_args()

    if not (args.sample or args.full):
        parser.error("Must specify either --sample or --full")

    # Parse target size
    target_size = None
    if args.target_size:
        try:
            w, h = map(int, args.target_size.split(','))
            target_size = (w, h)
        except:
            parser.error("Invalid target size format. Use 'width,height'")

    # Initialize downloader
    downloader = RxImageDownloader(args.data_dir)

    try:
        # Get dataset
        if args.sample:
            df = downloader.get_sample_dataset(args.classes, args.images_per_class)
        else:
            df = downloader.get_full_dataset(args.max_classes, args.max_images_per_class)

        # Download images
        downloader.download_images(df, args.workers)

        # Convert images
        downloader.convert_images(target_size=target_size)

        # Create splits
        downloader.create_splits(df)

        # Optional cleanup
        if args.cleanup:
            downloader.cleanup_raw_files()

        logger.info("Data acquisition completed successfully!")

    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Data acquisition failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()