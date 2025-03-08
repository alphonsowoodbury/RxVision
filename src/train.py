"""
RxVision25 Training Script
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RxVisionTrainer:
    """Handles model training for RxVision25."""
    
    def __init__(
        self,
        train_dir: str = 'data/train',
        val_dir: str = 'data/val',
        img_size: int = 224,
        batch_size: int = 32,
        num_classes: int = 15,
        learning_rate: float = 1e-4
    ):
        """Initialize trainer with configuration."""
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Validate directories
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
        if not self.val_dir.exists():
            logger.warning(f"Validation directory not found: {self.val_dir}")
            logger.warning("Will use validation split from training data")
            self.val_dir = None
    
    def create_model(self):
        """Create model architecture based on best performing model from v1."""
        model = tf.keras.Sequential([
            # First conv block
            tf.keras.layers.Conv2D(768, (3,3), activation='relu', padding='same',
                                 input_shape=(self.img_size, self.img_size, 3)),
            tf.keras.layers.MaxPooling2D((3, 3)),
            
            # Second conv block
            tf.keras.layers.Conv2D(1024, (3,3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((3, 3)),
            tf.keras.layers.SpatialDropout2D(0.1),
            
            # Third conv block
            tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth conv block
            tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
            tf.keras.layers.GaussianNoise(1.0),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_data_generators(self):
        """Create data generators with augmentation."""
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            validation_split=0.2 if self.val_dir is None else None
        )
        
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training' if self.val_dir is None else None
        )
        
        if self.val_dir is not None:
            val_generator = val_datagen.flow_from_directory(
                self.val_dir,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical'
            )
        else:
            val_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation'
            )
        
        return train_generator, val_generator
    
    def train(self, epochs=100):
        """Train the model."""
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"Available GPUs: {len(gpus)}")
        
        # Create and compile model
        model = self.create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create data generators
        train_generator, val_generator = self.create_data_generators()
        
        # Setup model checkpointing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path('models') / f"model_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Save best model
            tf.keras.callbacks.ModelCheckpoint(
                model_dir / 'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True
            ),
            # Reduce learning rate when stuck
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            # Save training logs
            tf.keras.callbacks.CSVLogger(
                model_dir / 'training_log.csv'
            )
        ]
        
        # Train
        logger.info("Starting training...")
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            workers=8,
            use_multiprocessing=True
        )
        
        # Save final model and training history
        model.save(model_dir / 'final_model.h5')
        np.save(model_dir / 'training_history.npy', history.history)
        logger.info(f"Training completed! Models saved in {model_dir}/")
        
        return history

def main():
    """Main training function."""
    try:
        trainer = RxVisionTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 