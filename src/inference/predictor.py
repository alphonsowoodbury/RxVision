"""
Inference module for RxVision25.

This module handles model inference for medication image classification,
providing both real-time and batch prediction capabilities.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RxPredictor:
    """Handles model inference for medication classification."""
    
    def __init__(
        self,
        model_path: str,
        class_map_path: Optional[str] = None,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32
    ):
        """Initialize the predictor.
        
        Args:
            model_path: Path to saved model
            class_map_path: Path to class mapping JSON file
            target_size: Input image size (height, width)
            batch_size: Batch size for inference
        """
        self.model_path = Path(model_path)
        self.target_size = target_size
        self.batch_size = batch_size
        
        # Load model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Load class mapping if provided
        self.class_map = None
        if class_map_path:
            try:
                with open(class_map_path, 'r') as f:
                    self.class_map = json.load(f)
                logger.info(f"Loaded class mapping from {class_map_path}")
            except Exception as e:
                logger.error(f"Error loading class mapping: {e}")
                raise
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess a single image for inference.
        
        Args:
            image: Image to preprocess (file path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Resize
            image = image.resize(self.target_size)
            
            # Convert to array and add batch dimension
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Normalize
            img_array = img_array.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict_single(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_top_k: int = 1
    ) -> List[Dict[str, Union[str, float]]]:
        """Make prediction on a single image.
        
        Args:
            image: Image to classify
            return_top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries containing class labels and probabilities
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            
            # Get top k predictions
            top_k_indices = np.argsort(predictions[0])[-return_top_k:][::-1]
            
            # Format results
            results = []
            for idx in top_k_indices:
                class_label = (
                    self.class_map[str(idx)]
                    if self.class_map is not None
                    else str(idx)
                )
                results.append({
                    'class': class_label,
                    'probability': float(predictions[0][idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        return_top_k: int = 1
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """Make predictions on a batch of images.
        
        Args:
            images: List of images to classify
            return_top_k: Number of top predictions to return per image
            
        Returns:
            List of prediction results for each image
        """
        try:
            # Preprocess images
            processed_images = []
            for image in images:
                processed_images.append(self.preprocess_image(image))
            
            # Stack images into batch
            batch = np.vstack(processed_images)
            
            # Make predictions
            predictions = self.model.predict(batch)
            
            # Format results for each image
            batch_results = []
            for pred in predictions:
                # Get top k predictions
                top_k_indices = np.argsort(pred)[-return_top_k:][::-1]
                
                # Format results
                results = []
                for idx in top_k_indices:
                    class_label = (
                        self.class_map[str(idx)]
                        if self.class_map is not None
                        else str(idx)
                    )
                    results.append({
                        'class': class_label,
                        'probability': float(pred[idx])
                    })
                batch_results.append(results)
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            raise
    
    @staticmethod
    def explain_prediction(
        image: Union[str, np.ndarray, Image.Image],
        model: tf.keras.Model,
        layer_name: Optional[str] = None,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM visualization for model decision.
        
        Args:
            image: Input image
            model: Model to explain
            layer_name: Name of layer to use for Grad-CAM (defaults to last conv)
            class_idx: Index of class to explain (defaults to predicted class)
            
        Returns:
            Heatmap array
        """
        try:
            # Get last conv layer if not specified
            if layer_name is None:
                for layer in reversed(model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        layer_name = layer.name
                        break
            
            # Get gradient model
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [
                    model.get_layer(layer_name).output,
                    model.output
                ]
            )
            
            # Process image
            if isinstance(image, (str, Image.Image)):
                image = np.array(Image.open(image) if isinstance(image, str) else image)
            image = tf.convert_to_tensor(image)
            image = tf.expand_dims(image, 0)
            
            # Generate gradients
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(image)
                if class_idx is None:
                    class_idx = tf.argmax(predictions[0])
                class_channel = predictions[:, class_idx]
            
            # Calculate gradients
            grads = tape.gradient(class_channel, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Generate heatmap
            conv_output = conv_output[0]
            heatmap = tf.reduce_sum(
                tf.multiply(pooled_grads, conv_output),
                axis=-1
            )
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            raise 