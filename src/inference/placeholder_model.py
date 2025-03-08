"""
Placeholder model generator for testing the inference service.
"""

import tensorflow as tf
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check if MPS (Metal) GPU is available."""
    try:
        if tf.config.list_physical_devices('GPU'):
            logger.info("GPU device found")
            return True
        if tf.config.list_physical_devices('MPS'):
            logger.info("Apple M1 GPU (Metal) device found")
            return True
        logger.warning("No GPU found, using CPU")
        return False
    except Exception as e:
        logger.warning(f"Error checking GPU: {e}")
        return False

def create_placeholder_model(input_shape=(224, 224, 3), num_classes=1):
    """Create a simple placeholder model for testing.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled TensorFlow model
    """
    # Check GPU availability
    has_gpu = check_gpu()
    if has_gpu:
        logger.info("Building model with GPU support")
    else:
        logger.info("Building model for CPU")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_placeholder_model():
    """Create and save a placeholder model for testing."""
    model = create_placeholder_model()
    model.save('models/best_model.h5')
    logger.info("Saved placeholder model to models/best_model.h5") 