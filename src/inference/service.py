"""
FastAPI service for RxVision25 model serving.

This module provides a REST API for medication image classification,
supporting both single image and batch inference requests.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
from PIL import Image
import io
import logging
from pathlib import Path
import json
import time

from .predictor import RxPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RxVision25 API",
    description="API for medication image classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class PredictionResponse(BaseModel):
    """Model for prediction response."""
    predictions: List[Dict[str, Any]]
    inference_time: float
    model_version: str

class BatchPredictionRequest(BaseModel):
    """Model for batch prediction request."""
    image_urls: List[str]
    return_top_k: Optional[int] = 1

class ExplanationResponse(BaseModel):
    """Model for explanation response."""
    heatmap: List[List[float]]
    class_predicted: str
    confidence: float

# Global variables
predictor: Optional[RxPredictor] = None
model_version: str = "1.0.0"

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global predictor
    
    try:
        model_path = Path("models/best_model.h5")
        class_map_path = Path("models/class_map.json")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        predictor = RxPredictor(
            model_path=str(model_path),
            class_map_path=str(class_map_path) if class_map_path.exists() else None
        )
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_version": model_version
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    return_top_k: int = 1
):
    """Make prediction on a single image.
    
    Args:
        file: Uploaded image file
        return_top_k: Number of top predictions to return
        
    Returns:
        Prediction results and metadata
    """
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        start_time = time.time()
        predictions = predictor.predict_single(image, return_top_k=return_top_k)
        inference_time = time.time() - start_time
        
        return PredictionResponse(
            predictions=predictions,
            inference_time=inference_time,
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions on multiple images.
    
    Args:
        request: Batch prediction request
        
    Returns:
        List of prediction results
    """
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Process each image
        results = []
        for url in request.image_urls:
            # Download and process image
            # Note: In production, use async download
            start_time = time.time()
            predictions = predictor.predict_single(url, return_top_k=request.return_top_k)
            inference_time = time.time() - start_time
            
            results.append(
                PredictionResponse(
                    predictions=predictions,
                    inference_time=inference_time,
                    model_version=model_version
                )
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=ExplanationResponse)
async def explain(
    file: UploadFile = File(...),
    class_idx: Optional[int] = None
):
    """Generate explanation for model prediction.
    
    Args:
        file: Uploaded image file
        class_idx: Optional class index to explain
        
    Returns:
        Grad-CAM heatmap and prediction details
    """
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction first
        prediction = predictor.predict_single(image)[0]
        
        # Generate explanation
        heatmap = predictor.explain_prediction(
            image,
            predictor.model,
            class_idx=class_idx
        )
        
        return ExplanationResponse(
            heatmap=heatmap.tolist(),
            class_predicted=prediction['class'],
            confidence=prediction['probability']
        )
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "version": model_version,
        "framework": "tensorflow",
        "input_shape": predictor.target_size,
        "num_classes": len(predictor.class_map) if predictor.class_map else None,
        "class_labels": list(predictor.class_map.values()) if predictor.class_map else None
    } 