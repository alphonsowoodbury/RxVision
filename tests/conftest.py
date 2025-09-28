"""Test configuration and fixtures for RxVision2025."""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import json
from typing import Generator, Dict, Any


@pytest.fixture(scope="session")
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample pill image for testing."""
    # Create a simple 224x224 RGB image that looks like a pill
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Create a circular pill shape
    center_x, center_y = 112, 112
    radius = 80

    for y in range(224):
        for x in range(224):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance <= radius:
                # White pill with slight color variation
                image[y, x] = [240 + int(15 * np.random.random()),
                              240 + int(15 * np.random.random()),
                              240 + int(15 * np.random.random())]

    return image


@pytest.fixture
def sample_pil_image(sample_image: np.ndarray) -> Image.Image:
    """Convert sample image to PIL Image."""
    return Image.fromarray(sample_image)


@pytest.fixture
def mock_dataset_structure(temp_data_dir: Path) -> Dict[str, Path]:
    """Create a mock dataset structure for testing."""
    structure = {
        "train": temp_data_dir / "train",
        "val": temp_data_dir / "val",
        "test": temp_data_dir / "test"
    }

    # Create directories
    for split_dir in structure.values():
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create class directories
        for class_name in ["aspirin", "ibuprofen", "acetaminophen"]:
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)

    return structure


@pytest.fixture
def sample_dataset_info() -> Dict[str, Any]:
    """Sample dataset information."""
    return {
        "dataset_info": {
            "total_images": 150,
            "num_classes": 3,
            "creation_date": "2024-01-01T00:00:00",
            "source": "Test Dataset",
            "type": "synthetic"
        },
        "class_info": {
            "aspirin": {"drug_name": "Aspirin 325mg", "image_count": 50},
            "ibuprofen": {"drug_name": "Ibuprofen 200mg", "image_count": 50},
            "acetaminophen": {"drug_name": "Acetaminophen 500mg", "image_count": 50}
        },
        "split_stats": {
            "train": 90,
            "val": 30,
            "test": 30
        }
    }


@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Mock model configuration for testing."""
    return {
        "model": {
            "architecture": "efficientnet_b0",
            "num_classes": 3,
            "input_shape": [224, 224, 3],
            "dropout_rate": 0.2
        },
        "training": {
            "batch_size": 16,
            "epochs": 10,
            "learning_rate": 0.001,
            "validation_split": 0.2
        },
        "data": {
            "image_size": 224,
            "augmentation": {
                "rotation_range": 20,
                "width_shift_range": 0.1,
                "height_shift_range": 0.1,
                "horizontal_flip": True
            }
        }
    }


@pytest.fixture
def sample_prediction_response() -> Dict[str, Any]:
    """Sample API prediction response."""
    return {
        "medication_name": "Aspirin 325mg",
        "confidence": 0.95,
        "alternatives": [
            {"name": "Ibuprofen 200mg", "confidence": 0.03},
            {"name": "Acetaminophen 500mg", "confidence": 0.02}
        ],
        "processing_time": 0.45,
        "model_version": "v2.5.0"
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("RXVISION_ENV", "test")
    monkeypatch.setenv("RXVISION_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RXVISION_DATA_DIR", "test_data")


@pytest.fixture
def mock_nih_metadata() -> Dict[str, Any]:
    """Mock NIH dataset metadata."""
    return {
        "_id": {"$oid": "test123"},
        "ndc11": "00093-0311-01",
        "rxcui": 978006,
        "name": "Test Medication 10 MG Oral Tablet",
        "labeler": "Test Pharmaceuticals",
        "mpc": {
            "shape": "ROUND",
            "size": 10,
            "color": "WHITE",
            "imprint": "TEST;123",
            "imprintColor": "BLACK",
            "imprintType": "PRINTED",
            "score": 1
        },
        "ingredients": {
            "active": ["test compound 10 mg"],
            "inactive": ["lactose", "magnesium stearate"]
        }
    }