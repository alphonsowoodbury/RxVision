"""Tests for training functionality."""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from typing import Dict, Any

from src.training.train import RxVisionTrainer


class TestRxVisionTrainer:
    """Test cases for RxVisionTrainer class."""

    def test_trainer_initialization(self, mock_model_config: Dict[str, Any]):
        """Test trainer initialization with default parameters."""
        trainer = RxVisionTrainer()

        assert trainer.img_size == 224
        assert trainer.batch_size == 32
        assert trainer.num_classes == 15
        assert trainer.learning_rate == 1e-4

    def test_trainer_custom_initialization(self):
        """Test trainer initialization with custom parameters."""
        trainer = RxVisionTrainer(
            train_dir="custom/train",
            val_dir="custom/val",
            img_size=128,
            batch_size=16,
            num_classes=10,
            learning_rate=1e-3
        )

        assert trainer.train_dir == "custom/train"
        assert trainer.val_dir == "custom/val"
        assert trainer.img_size == 128
        assert trainer.batch_size == 16
        assert trainer.num_classes == 10
        assert trainer.learning_rate == 1e-3

    @patch('src.training.train.tf.keras.utils.image_dataset_from_directory')
    def test_load_datasets(self, mock_dataset_from_directory, mock_dataset_structure):
        """Test dataset loading functionality."""
        # Mock the TensorFlow dataset
        mock_dataset = Mock()
        mock_dataset.class_names = ['aspirin', 'ibuprofen', 'acetaminophen']
        mock_dataset_from_directory.return_value = mock_dataset

        trainer = RxVisionTrainer(
            train_dir=str(mock_dataset_structure["train"]),
            val_dir=str(mock_dataset_structure["val"])
        )

        train_ds, val_ds = trainer.load_datasets()

        # Verify datasets are loaded
        assert train_ds is not None
        assert val_ds is not None

        # Verify correct parameters passed to TensorFlow
        assert mock_dataset_from_directory.call_count == 2
        calls = mock_dataset_from_directory.call_args_list

        # Check train dataset call
        train_call_args = calls[0][1]
        assert train_call_args['batch_size'] == 32
        assert train_call_args['image_size'] == (224, 224)

        # Check validation dataset call
        val_call_args = calls[1][1]
        assert val_call_args['batch_size'] == 32
        assert val_call_args['image_size'] == (224, 224)

    @patch('src.training.train.tf.keras.applications.EfficientNetV2B0')
    def test_build_model(self, mock_efficientnet):
        """Test model building functionality."""
        # Mock EfficientNet
        mock_base_model = Mock()
        mock_efficientnet.return_value = mock_base_model

        trainer = RxVisionTrainer(num_classes=3)

        with patch('src.training.train.tf.keras.Sequential') as mock_sequential:
            model = trainer.build_model()

            # Verify EfficientNet was called with correct parameters
            mock_efficientnet.assert_called_once_with(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )

            # Verify model was built
            mock_sequential.assert_called_once()

    def test_prepare_callbacks(self, temp_data_dir: Path):
        """Test callback preparation."""
        trainer = RxVisionTrainer()
        callbacks = trainer.prepare_callbacks(str(temp_data_dir))

        # Should have at least early stopping, model checkpoint, and reduce LR
        assert len(callbacks) >= 3

        # Check for specific callback types
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'EarlyStopping' in callback_types
        assert 'ModelCheckpoint' in callback_types
        assert 'ReduceLROnPlateau' in callback_types

    @patch('src.training.train.RxVisionTrainer.load_datasets')
    @patch('src.training.train.RxVisionTrainer.build_model')
    def test_train_model(self, mock_build_model, mock_load_datasets, temp_data_dir: Path):
        """Test model training process."""
        # Mock datasets
        mock_train_ds = Mock()
        mock_val_ds = Mock()
        mock_load_datasets.return_value = (mock_train_ds, mock_val_ds)

        # Mock model
        mock_model = Mock()
        mock_history = Mock()
        mock_model.fit.return_value = mock_history
        mock_build_model.return_value = mock_model

        trainer = RxVisionTrainer()
        history = trainer.train(
            epochs=2,
            save_dir=str(temp_data_dir)
        )

        # Verify training was called
        mock_model.compile.assert_called_once()
        mock_model.fit.assert_called_once()

        # Verify model was saved
        mock_model.save.assert_called()

        assert history == mock_history

    def test_create_data_generators_legacy_fallback(self, mock_dataset_structure):
        """Test fallback to ImageDataGenerator when directory-based loading fails."""
        trainer = RxVisionTrainer(
            train_dir=str(mock_dataset_structure["train"]),
            val_dir=str(mock_dataset_structure["val"])
        )

        with patch('src.training.train.tf.keras.utils.image_dataset_from_directory',
                   side_effect=Exception("Dataset loading failed")):
            with patch('src.training.train.tf.keras.preprocessing.image.ImageDataGenerator') as mock_idg:
                mock_generator = Mock()
                mock_idg.return_value.flow_from_directory.return_value = mock_generator

                train_gen, val_gen = trainer.create_data_generators()

                # Should fallback to ImageDataGenerator
                assert train_gen == mock_generator
                assert val_gen == mock_generator

    @pytest.mark.slow
    def test_end_to_end_training_small_model(self, mock_dataset_structure, sample_image):
        """End-to-end test with a very small model for integration testing."""
        # Create sample images in the mock dataset
        for split_name, split_dir in mock_dataset_structure.items():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    # Create a few sample images
                    for i in range(3):
                        img_path = class_dir / f"sample_{i}.jpg"
                        from PIL import Image
                        img = Image.fromarray(sample_image)
                        img.save(img_path)

        trainer = RxVisionTrainer(
            train_dir=str(mock_dataset_structure["train"]),
            val_dir=str(mock_dataset_structure["val"]),
            batch_size=2,
            num_classes=3
        )

        # Override build_model to create a very simple model for testing
        def build_simple_model():
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            return model

        trainer.build_model = build_simple_model

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should complete without errors
            history = trainer.train(epochs=1, save_dir=temp_dir)
            assert history is not None

    def test_model_compilation_parameters(self):
        """Test that model is compiled with correct parameters."""
        trainer = RxVisionTrainer(learning_rate=0.001)

        with patch('src.training.train.tf.keras.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_sequential.return_value = mock_model

            model = trainer.build_model()
            trainer._compile_model(model)

            # Verify compile was called with correct optimizer and loss
            mock_model.compile.assert_called_once()
            compile_args = mock_model.compile.call_args[1]

            assert 'optimizer' in compile_args
            assert 'loss' in compile_args
            assert 'metrics' in compile_args
            assert compile_args['loss'] == 'sparse_categorical_crossentropy'

    def test_save_training_history(self, temp_data_dir: Path):
        """Test saving training history."""
        trainer = RxVisionTrainer()

        # Mock history object
        mock_history = Mock()
        mock_history.history = {
            'loss': [0.5, 0.3, 0.2],
            'accuracy': [0.8, 0.9, 0.95],
            'val_loss': [0.6, 0.4, 0.3],
            'val_accuracy': [0.75, 0.85, 0.9]
        }

        history_path = trainer._save_training_history(mock_history, str(temp_data_dir))

        # Verify file was created
        assert history_path.exists()

        # Verify content is valid JSON
        import json
        with open(history_path, 'r') as f:
            saved_history = json.load(f)

        assert saved_history == mock_history.history