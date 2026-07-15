"""Optional FastAPI surface for local research experiments."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from rxvision.inference import Predictor

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """Create an API whose readiness reflects whether a model was supplied."""

    app = FastAPI(
        title="RxVision Research API",
        description="Candidate rankings from an unvalidated research model; not for medical use.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, bool | str]:
        ready = predictor is not None
        return {"status": "ready" if ready else "not_ready", "model_loaded": ready}

    @app.post("/predict")
    async def predict(file: Annotated[UploadFile, File()]) -> dict[str, object]:
        if predictor is None:
            raise HTTPException(status_code=503, detail="no research model is loaded")
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=415, detail="expected JPEG, PNG, or WebP")

        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="image exceeds the 8 MiB limit")
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=400, detail="invalid image") from exc

        result = predictor.predict(image)
        return {
            **result.to_dict(),
            "warning": "Research output only. Do not use this result to identify medication.",
        }

    return app


def _predictor_from_environment() -> Predictor | None:
    path = os.environ.get("RXVISION_CHECKPOINT")
    if not path or not Path(path).is_file():
        return None
    return Predictor.from_checkpoint(path)


app = create_app(_predictor_from_environment())
