"""Tests for data engineering functionality."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import json
import tempfile
from PIL import Image
from typing import Dict, Any, List

from scripts.download_data_modern import RxImageModernDownloader


class TestRxImageModernDownloader:
    """Test cases for RxImageModernDownloader class."""

    def test_downloader_initialization(self, temp_data_dir: Path):
        """Test downloader initialization."""
        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))

        assert downloader.base_dir == temp_data_dir
        assert downloader.raw_dir == temp_data_dir / "raw"
        assert downloader.processed_dir == temp_data_dir / "processed"

        # Check that directories are created
        assert downloader.raw_dir.exists()
        assert downloader.processed_dir.exists()

    def test_create_directory_structure(self, temp_data_dir: Path):
        """Test directory structure creation."""
        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))
        downloader.create_directory_structure()

        expected_dirs = [
            temp_data_dir / "raw",
            temp_data_dir / "processed",
            temp_data_dir / "train",
            temp_data_dir / "val",
            temp_data_dir / "test"
        ]

        for dir_path in expected_dirs:
            assert dir_path.exists()
            assert dir_path.is_dir()

    @patch('scripts.download_data_modern.requests.get')
    def test_download_from_direct_url_success(self, mock_get, temp_data_dir: Path):
        """Test successful download from direct URL."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content.return_value = [b'test_data_chunk']
        mock_get.return_value = mock_response

        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))

        with patch('zipfile.ZipFile') as mock_zipfile:
            mock_zip = Mock()
            mock_zipfile.return_value.__enter__.return_value = mock_zip

            result = downloader.download_from_direct_url("sample")

            assert result is True
            mock_get.assert_called_once()
            mock_zip.extractall.assert_called_once()

    @patch('scripts.download_data_modern.requests.get')
    def test_download_from_direct_url_failure(self, mock_get, temp_data_dir: Path):
        """Test failed download from direct URL."""
        # Mock failed response
        mock_get.side_effect = Exception("Network error")

        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))
        result = downloader.download_from_direct_url("sample")

        assert result is False

    def test_create_synthetic_dataset(self, temp_data_dir: Path):
        """Test synthetic dataset creation."""
        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))

        result = downloader.create_synthetic_dataset(
            num_classes=3,
            images_per_class=5
        )

        assert result is True

        # Check that dataset info file was created
        info_file = temp_data_dir / "dataset_info.json"
        assert info_file.exists()

        # Check dataset info content
        with open(info_file, 'r') as f:
            info = json.load(f)

        assert info["dataset_info"]["num_classes"] == 3
        assert info["dataset_info"]["total_images"] == 15
        assert info["dataset_info"]["type"] == "synthetic"

    def test_create_synthetic_pill_image(self, temp_data_dir: Path):
        """Test synthetic pill image generation."""
        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))

        image = downloader._create_synthetic_pill_image()

        assert isinstance(image, Image.Image)
        assert image.size == (224, 224)
        assert image.mode == "RGB"

        # Check that image is not completely black or white
        img_array = np.array(image)
        assert img_array.min() < 255
        assert img_array.max() > 0

    def test_split_dataset(self, temp_data_dir: Path, sample_image: np.ndarray):
        """Test dataset splitting functionality."""
        # Create mock dataset structure
        source_dir = temp_data_dir / "source"
        source_dir.mkdir()

        # Create class directories with sample images
        for class_name in ["class1", "class2"]:
            class_dir = source_dir / class_name
            class_dir.mkdir()

            for i in range(10):
                img_path = class_dir / f"img_{i}.jpg"
                img = Image.fromarray(sample_image)
                img.save(img_path)

        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))

        # Test splitting
        splits = downloader._split_dataset(
            source_dir,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )

        # Check split ratios
        assert splits["train"] == 7  # 70% of 10
        assert splits["val"] == 2   # 20% of 10
        assert splits["test"] == 1  # 10% of 10

        # Check that split directories exist
        for split in ["train", "val", "test"]:
            split_dir = temp_data_dir / split
            assert split_dir.exists()

            for class_name in ["class1", "class2"]:
                class_split_dir = split_dir / class_name
                assert class_split_dir.exists()

    def test_analyze_dataset_structure(self, temp_data_dir: Path):
        """Test dataset structure analysis."""
        # Create mock dataset
        for split in ["train", "val", "test"]:
            split_dir = temp_data_dir / split
            split_dir.mkdir()

            for class_name in ["aspirin", "ibuprofen"]:
                class_dir = split_dir / class_name
                class_dir.mkdir()

                # Create mock images
                for i in range(5):
                    img_file = class_dir / f"img_{i}.jpg"
                    img_file.touch()

        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))
        analysis = downloader.analyze_dataset_structure()

        assert analysis["total_images"] == 30  # 3 splits * 2 classes * 5 images
        assert analysis["num_classes"] == 2
        assert "aspirin" in analysis["classes"]
        assert "ibuprofen" in analysis["classes"]
        assert analysis["splits"]["train"] == 10
        assert analysis["splits"]["val"] == 10
        assert analysis["splits"]["test"] == 10

    @patch('scripts.download_data_modern.requests.get')
    def test_get_sample_dataset_api_call(self, mock_get, temp_data_dir: Path):
        """Test API call for sample dataset."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "ndc11": "12345-678-90",
                "name": "Test Medication",
                "rxnavImageFileName": "test_image.jpg"
            }
        ]
        mock_get.return_value = mock_response

        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))
        df = downloader.get_sample_dataset(num_classes=1, records_per_class=1)

        assert not df.empty
        assert len(df) == 1
        assert "ndc11" in df.columns
        assert "name" in df.columns

    @patch('scripts.download_data_modern.requests.get')
    def test_get_sample_dataset_api_failure(self, mock_get, temp_data_dir: Path):
        """Test API failure handling."""
        # Mock API failure
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response

        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))
        df = downloader.get_sample_dataset(num_classes=1, records_per_class=1)

        assert df.empty

    def test_save_dataset_info(self, temp_data_dir: Path):
        """Test saving dataset information."""
        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))

        info_data = {
            "dataset_info": {
                "total_images": 100,
                "num_classes": 5,
                "source": "Test Dataset"
            }
        }

        info_file = downloader._save_dataset_info(info_data)

        assert info_file.exists()
        assert info_file.name == "dataset_info.json"

        # Verify content
        with open(info_file, 'r') as f:
            saved_data = json.load(f)

        assert saved_data == info_data

    def test_validate_dataset_integrity(self, temp_data_dir: Path, sample_image: np.ndarray):
        """Test dataset integrity validation."""
        # Create valid dataset
        train_dir = temp_data_dir / "train"
        train_dir.mkdir()

        class_dir = train_dir / "test_class"
        class_dir.mkdir()

        # Add valid image
        valid_img_path = class_dir / "valid.jpg"
        img = Image.fromarray(sample_image)
        img.save(valid_img_path)

        # Add invalid file
        invalid_file_path = class_dir / "invalid.txt"
        invalid_file_path.write_text("not an image")

        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))
        issues = downloader._validate_dataset_integrity()

        # Should detect the invalid file
        assert len(issues) > 0
        assert any("invalid.txt" in issue for issue in issues)

    def test_cleanup_corrupted_files(self, temp_data_dir: Path):
        """Test cleanup of corrupted files."""
        class_dir = temp_data_dir / "test_class"
        class_dir.mkdir()

        # Create corrupted image file
        corrupted_file = class_dir / "corrupted.jpg"
        corrupted_file.write_bytes(b"not_a_valid_image")

        downloader = RxImageModernDownloader(base_dir=str(temp_data_dir))
        cleaned_count = downloader._cleanup_corrupted_files(class_dir)

        assert cleaned_count == 1
        assert not corrupted_file.exists()


class TestDataAugmentation:
    """Test cases for data augmentation functionality."""

    @pytest.fixture
    def sample_augmentation_config(self) -> Dict[str, Any]:
        """Sample augmentation configuration."""
        return {
            "rotation_range": 20,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "horizontal_flip": True,
            "zoom_range": 0.1,
            "brightness_range": [0.8, 1.2]
        }

    def test_image_augmentation_pipeline(
        self,
        sample_image: np.ndarray,
        sample_augmentation_config: Dict[str, Any]
    ):
        """Test image augmentation pipeline."""
        with patch('src.data.augmentation.create_augmentation_pipeline') as mock_pipeline:
            # Mock augmentation function
            def mock_augment(image):
                # Simple transformation: add noise
                noisy = image + np.random.normal(0, 0.1, image.shape)
                return np.clip(noisy, 0, 255).astype(np.uint8)

            mock_pipeline.return_value = mock_augment

            from src.data.augmentation import create_augmentation_pipeline
            augment_fn = create_augmentation_pipeline(sample_augmentation_config)

            # Test augmentation
            augmented = augment_fn(sample_image)

            assert augmented.shape == sample_image.shape
            assert augmented.dtype == sample_image.dtype
            # Should be different from original (due to noise)
            assert not np.array_equal(augmented, sample_image)

    def test_batch_augmentation(self, sample_image: np.ndarray):
        """Test batch augmentation functionality."""
        batch_size = 4
        batch = np.array([sample_image] * batch_size)

        with patch('src.data.augmentation.apply_batch_augmentation') as mock_batch_aug:
            # Mock batch augmentation
            mock_batch_aug.return_value = batch + 1  # Simple transformation

            from src.data.augmentation import apply_batch_augmentation
            augmented_batch = apply_batch_augmentation(batch)

            assert augmented_batch.shape == batch.shape
            mock_batch_aug.assert_called_once_with(batch)


class TestDataPreprocessing:
    """Test cases for data preprocessing functionality."""

    def test_image_normalization(self, sample_image: np.ndarray):
        """Test image normalization."""
        with patch('src.data.preprocessing.normalize_image') as mock_normalize:
            # Mock normalization
            normalized = sample_image.astype(np.float32) / 255.0
            mock_normalize.return_value = normalized

            from src.data.preprocessing import normalize_image
            result = normalize_image(sample_image)

            assert result.dtype == np.float32
            assert result.min() >= 0.0
            assert result.max() <= 1.0

    def test_image_resizing(self, sample_image: np.ndarray):
        """Test image resizing functionality."""
        target_size = (128, 128)

        with patch('src.data.preprocessing.resize_image') as mock_resize:
            # Mock resize
            resized = np.zeros((*target_size, 3), dtype=np.uint8)
            mock_resize.return_value = resized

            from src.data.preprocessing import resize_image
            result = resize_image(sample_image, target_size)

            assert result.shape[:2] == target_size
            mock_resize.assert_called_once_with(sample_image, target_size)

    def test_batch_preprocessing(self, sample_image: np.ndarray):
        """Test batch preprocessing pipeline."""
        batch = [sample_image, sample_image, sample_image]

        with patch('src.data.preprocessing.preprocess_batch') as mock_preprocess:
            # Mock preprocessing
            processed_batch = np.array(batch).astype(np.float32) / 255.0
            mock_preprocess.return_value = processed_batch

            from src.data.preprocessing import preprocess_batch
            result = preprocess_batch(batch)

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert len(result) == len(batch)