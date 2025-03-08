"""
Model training module for RxVision25.

This module provides training functionality with modern best practices,
including mixed precision training, learning rate scheduling, and comprehensive logging.
"""

import tensorflow as tf
from tensorflow.keras import callbacks
import mlflow
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training with modern best practices."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        experiment_name: str,
        model_dir: str = 'models',
        use_mixed_precision: bool = True
    ):
        """Initialize the trainer.
        
        Args:
            model: Keras model to train
            experiment_name: Name for MLflow experiment
            model_dir: Directory to save model artifacts
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.experiment_name = experiment_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mixed precision if requested
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Enabled mixed precision training")
    
    def _create_callbacks(
        self,
        patience: int = 10,
        min_delta: float = 1e-4
    ) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            
        Returns:
            List of Keras callbacks
        """
        callbacks_list = [
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                mode='min',
                restore_best_weights=True
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_delta=min_delta,
                mode='min'
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=str(self.model_dir / 'logs'),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        return callbacks_list
    
    def compile_model(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4
    ) -> None:
        """Compile the model with optimizer and loss.
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: Weight decay factor
        """
        # Create optimizer with weight decay
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        logger.info("Compiled model with AdamW optimizer")
    
    def train(
        self,
        train_data: tf.keras.preprocessing.image.DataFrameIterator,
        val_data: tf.keras.preprocessing.image.DataFrameIterator,
        epochs: int = 100,
        initial_epoch: int = 0,
        class_weights: Optional[Dict[int, float]] = None
    ) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            train_data: Training data generator
            val_data: Validation data generator
            epochs: Number of epochs to train
            initial_epoch: Epoch to start from
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Training history
        """
        # Start MLflow run
        mlflow.start_run(experiment_name=self.experiment_name)
        
        try:
            # Log parameters
            mlflow.log_params({
                'epochs': epochs,
                'initial_epoch': initial_epoch,
                'batch_size': train_data.batch_size,
                'optimizer': self.model.optimizer.__class__.__name__,
                'learning_rate': float(self.model.optimizer.learning_rate.numpy())
            })
            
            # Create callbacks
            callbacks_list = self._create_callbacks()
            
            # Train model
            history = self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks_list,
                class_weight=class_weights
            )
            
            # Log metrics
            for metric_name, metric_values in history.history.items():
                for epoch, value in enumerate(metric_values, start=initial_epoch):
                    mlflow.log_metric(metric_name, value, step=epoch)
            
            # Save training history
            with open(self.model_dir / 'history.json', 'w') as f:
                json.dump(history.history, f)
            
            logger.info("Completed model training")
            return history
            
        finally:
            mlflow.end_run()
    
    def evaluate(
        self,
        test_data: tf.keras.preprocessing.image.DataFrameIterator
    ) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            test_data: Test data generator
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Evaluate model
        results = self.model.evaluate(
            test_data,
            return_dict=True
        )
        
        # Log results
        logger.info("Test set evaluation results:")
        for metric_name, value in results.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        return results 