"""Tests for inference functionality."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import json
from fastapi.testclient import TestClient
from PIL import Image
import io
from typing import Dict, Any

from src.inference.api import app
from src.inference.predictor import PillPredictor
from src.inference.model_loader import ModelLoader


class TestPillPredictor:
    """Test cases for PillPredictor class."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        with patch('src.inference.predictor.tf.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            predictor = PillPredictor(model_path="test_model.h5")
            assert predictor.model == mock_model
            assert predictor.input_size == (224, 224)

    def test_preprocess_image(self, sample_pil_image: Image.Image):
        """Test image preprocessing."""
        with patch('src.inference.predictor.tf.keras.models.load_model'):
            predictor = PillPredictor(model_path="test_model.h5")

            processed = predictor.preprocess_image(sample_pil_image)

            # Check output shape and type
            assert isinstance(processed, np.ndarray)
            assert processed.shape == (1, 224, 224, 3)
            assert processed.dtype == np.float32

            # Check normalization (values should be between 0 and 1)
            assert processed.min() >= 0.0
            assert processed.max() <= 1.0

    def test_predict_single_image(self, sample_pil_image: Image.Image):
        """Test single image prediction."""
        with patch('src.inference.predictor.tf.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_predictions = np.array([[0.1, 0.7, 0.2]])
            mock_model.predict.return_value = mock_predictions
            mock_load.return_value = mock_model

            predictor = PillPredictor(
                model_path="test_model.h5",
                class_names=["aspirin", "ibuprofen", "acetaminophen"]
            )

            result = predictor.predict(sample_pil_image)

            # Check result structure
            assert "class_name" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert "alternatives" in result

            # Check predicted class
            assert result["class_name"] == "ibuprofen"
            assert result["confidence"] == 0.7

            # Check alternatives
            assert len(result["alternatives"]) == 2
            assert result["alternatives"][0]["name"] == "acetaminophen"
            assert result["alternatives"][0]["confidence"] == 0.2

    def test_predict_batch_images(self, sample_image: np.ndarray):
        """Test batch prediction."""
        with patch('src.inference.predictor.tf.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_predictions = np.array([
                [0.1, 0.7, 0.2],
                [0.8, 0.1, 0.1]
            ])
            mock_model.predict.return_value = mock_predictions
            mock_load.return_value = mock_model

            predictor = PillPredictor(
                model_path="test_model.h5",
                class_names=["aspirin", "ibuprofen", "acetaminophen"]
            )

            # Create batch of images
            images = [Image.fromarray(sample_image) for _ in range(2)]
            results = predictor.predict_batch(images)

            assert len(results) == 2
            assert results[0]["class_name"] == "ibuprofen"
            assert results[1]["class_name"] == "aspirin"

    def test_predict_with_confidence_threshold(self, sample_pil_image: Image.Image):
        """Test prediction with confidence threshold."""
        with patch('src.inference.predictor.tf.keras.models.load_model') as mock_load:
            mock_model = Mock()
            # Low confidence predictions
            mock_predictions = np.array([[0.4, 0.3, 0.3]])
            mock_model.predict.return_value = mock_predictions
            mock_load.return_value = mock_model

            predictor = PillPredictor(
                model_path="test_model.h5",
                class_names=["aspirin", "ibuprofen", "acetaminophen"]
            )

            result = predictor.predict(sample_pil_image, confidence_threshold=0.5)

            # Should indicate low confidence
            assert result["confidence"] < 0.5
            assert "low_confidence" in result.get("warnings", [])


class TestModelLoader:
    """Test cases for ModelLoader class."""

    def test_load_tensorflow_model(self, temp_data_dir: Path):
        """Test loading TensorFlow model."""
        model_path = temp_data_dir / "test_model.h5"

        with patch('src.inference.model_loader.tf.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            loader = ModelLoader()
            model = loader.load_model(str(model_path))

            mock_load.assert_called_once_with(str(model_path))
            assert model == mock_model

    def test_load_onnx_model(self, temp_data_dir: Path):
        """Test loading ONNX model."""
        model_path = temp_data_dir / "test_model.onnx"

        with patch('src.inference.model_loader.onnxruntime.InferenceSession') as mock_session:
            mock_session_instance = Mock()
            mock_session.return_value = mock_session_instance

            loader = ModelLoader()
            model = loader.load_onnx_model(str(model_path))

            mock_session.assert_called_once_with(str(model_path))
            assert model == mock_session_instance

    def test_auto_detect_model_type(self, temp_data_dir: Path):
        """Test automatic model type detection."""
        # Test TensorFlow model detection
        tf_model_path = temp_data_dir / "model.h5"
        tf_model_path.touch()

        loader = ModelLoader()
        model_type = loader._detect_model_type(str(tf_model_path))
        assert model_type == "tensorflow"

        # Test ONNX model detection
        onnx_model_path = temp_data_dir / "model.onnx"
        onnx_model_path.touch()

        model_type = loader._detect_model_type(str(onnx_model_path))
        assert model_type == "onnx"

    def test_load_class_names(self, temp_data_dir: Path):
        """Test loading class names from file."""
        class_names = ["aspirin", "ibuprofen", "acetaminophen"]
        class_file = temp_data_dir / "classes.json"

        with open(class_file, 'w') as f:
            json.dump({"classes": class_names}, f)

        loader = ModelLoader()
        loaded_classes = loader.load_class_names(str(class_file))

        assert loaded_classes == class_names


class TestInferenceAPI:
    """Test cases for FastAPI inference endpoints."""

    def test_health_check(self):
        """Test health check endpoint."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_model_info(self):
        """Test model info endpoint."""
        client = TestClient(app)
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "input_shape" in data
        assert "num_classes" in data

    @patch('src.inference.api.predictor')
    def test_predict_endpoint(self, mock_predictor, sample_image: np.ndarray):
        """Test prediction endpoint."""
        # Mock predictor response
        mock_predictor.predict.return_value = {
            "class_name": "aspirin",
            "confidence": 0.95,
            "probabilities": [0.95, 0.03, 0.02],
            "alternatives": [
                {"name": "ibuprofen", "confidence": 0.03},
                {"name": "acetaminophen", "confidence": 0.02}
            ]
        }

        # Create test image file
        img = Image.fromarray(sample_image)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["medication_name"] == "aspirin"
        assert data["confidence"] == 0.95
        assert len(data["alternatives"]) == 2
        assert "processing_time" in data

    def test_predict_endpoint_invalid_file(self):
        """Test prediction endpoint with invalid file."""
        client = TestClient(app)

        # Send non-image file
        response = client.post(
            "/predict",
            files={"file": ("test.txt", "not an image", "text/plain")}
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data["detail"]

    def test_predict_endpoint_no_file(self):
        """Test prediction endpoint without file."""
        client = TestClient(app)
        response = client.post("/predict")

        assert response.status_code == 422  # Validation error

    @patch('src.inference.api.predictor')
    def test_predict_batch_endpoint(self, mock_predictor, sample_image: np.ndarray):
        """Test batch prediction endpoint."""
        # Mock predictor response
        mock_predictor.predict_batch.return_value = [
            {
                "class_name": "aspirin",
                "confidence": 0.95,
                "probabilities": [0.95, 0.03, 0.02]
            },
            {
                "class_name": "ibuprofen",
                "confidence": 0.88,
                "probabilities": [0.05, 0.88, 0.07]
            }
        ]

        # Create test images
        img = Image.fromarray(sample_image)
        img_bytes1 = io.BytesIO()
        img_bytes2 = io.BytesIO()
        img.save(img_bytes1, format='JPEG')
        img.save(img_bytes2, format='JPEG')
        img_bytes1.seek(0)
        img_bytes2.seek(0)

        client = TestClient(app)
        response = client.post(
            "/predict/batch",
            files=[
                ("files", ("test1.jpg", img_bytes1, "image/jpeg")),
                ("files", ("test2.jpg", img_bytes2, "image/jpeg"))
            ]
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["medication_name"] == "aspirin"
        assert data["predictions"][1]["medication_name"] == "ibuprofen"

    def test_cors_headers(self):
        """Test CORS headers are present."""
        client = TestClient(app)
        response = client.options("/health")

        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers

    @pytest.mark.asyncio
    async def test_async_prediction(self, sample_pil_image: Image.Image):
        """Test async prediction functionality."""
        with patch('src.inference.predictor.PillPredictor') as mock_predictor_class:
            mock_predictor = AsyncMock()
            mock_predictor.predict.return_value = {
                "class_name": "aspirin",
                "confidence": 0.95
            }
            mock_predictor_class.return_value = mock_predictor

            # Test async prediction
            result = await mock_predictor.predict(sample_pil_image)
            assert result["class_name"] == "aspirin"