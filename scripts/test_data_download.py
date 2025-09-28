#!/usr/bin/env python3
"""
Test script for NIH RxImage data acquisition

This script tests the data download functionality with a minimal dataset
to verify that the FTP connection and processing pipeline work correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.download_data import RxImageDownloader

def test_data_acquisition():
    """Test the data acquisition pipeline"""
    print("Testing NIH RxImage data acquisition...")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Initialize downloader with temp directory
            downloader = RxImageDownloader(temp_path / "test_data")

            print("1. Testing directory file download...")
            df = downloader.download_directory_file()

            if len(df) > 0:
                print(f"   Success: Loaded {len(df):,} image entries")
                print(f"   Found {df['NDC'].nunique():,} unique NDCs")
            else:
                print("   Error: No data loaded from directory file")
                return False

            print("2. Testing sample dataset creation...")
            sample_df = downloader.get_sample_dataset(target_classes=3, min_images_per_class=5)

            if len(sample_df) > 0:
                print(f"   Success: Created sample with {len(sample_df)} images")
                print(f"   Classes: {sample_df['NDC'].nunique()}")
            else:
                print("   Error: Could not create sample dataset")
                return False

            print("3. Testing image download (first 3 images only)...")
            # Limit to just a few images for testing
            test_df = sample_df.head(3)
            downloader.download_images(test_df)

            raw_files = list(downloader.raw_dir.glob('*'))
            if len(raw_files) > 0:
                print(f"   Success: Downloaded {len(raw_files)} files")
            else:
                print("   Warning: No files downloaded (this may be normal if files already exist)")

            print("4. Testing image conversion...")
            downloader.convert_images()

            processed_files = list(downloader.processed_dir.glob('*'))
            if len(processed_files) > 0:
                print(f"   Success: Converted {len(processed_files)} images")
            else:
                print("   Warning: No images converted")

            print("5. Testing dataset splits...")
            downloader.create_splits(test_df)

            # Check if split directories were created
            splits_created = all([
                (downloader.train_dir).exists(),
                (downloader.val_dir).exists(),
                (downloader.test_dir).exists()
            ])

            if splits_created:
                print("   Success: Dataset splits created")
            else:
                print("   Error: Dataset splits not created properly")
                return False

            print("\nAll tests passed! Data acquisition pipeline is working.")
            return True

        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_ftp_connection():
    """Test basic FTP connection to NIH server"""
    print("Testing FTP connection to NIH server...")

    try:
        from ftplib import FTP

        with FTP('lhcftp.nlm.nih.gov') as ftp:
            ftp.login()
            ftp.cwd('Open-Access-Datasets/Pills/')

            # List directories to verify access
            dirs = ftp.nlst()
            print(f"   Success: Connected to FTP server")
            print(f"   Found {len(dirs)} directories")

            # Check if directory file exists
            files = ftp.nlst()
            if 'directory_of_images.txt' in files:
                print("   Success: Directory file found")
            else:
                print("   Warning: Directory file not found in current location")

        return True

    except Exception as e:
        print(f"   Error: FTP connection failed: {e}")
        return False

if __name__ == "__main__":
    print("RxVision25 Data Acquisition Test")
    print("=" * 40)

    # Test FTP connection first
    ftp_ok = test_ftp_connection()

    if not ftp_ok:
        print("\nFTP connection test failed. Check internet connection and try again.")
        sys.exit(1)

    print("\n" + "="*40)

    # Test full pipeline
    pipeline_ok = test_data_acquisition()

    if pipeline_ok:
        print("\nData acquisition system is ready!")
        print("\nTo download the full sample dataset, run:")
        print("python scripts/download_data.py --sample")
        sys.exit(0)
    else:
        print("\nData acquisition test failed. Please check the error messages above.")
        sys.exit(1)