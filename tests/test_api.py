from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from rxvision.inference import Prediction
from rxvision.inference.api import create_app


class StubPredictor:
    def predict(self, _image):
        return Prediction(
            candidate_label="candidate-a",
            confidence=0.62,
            accepted=False,
            alternatives=({"label": "candidate-b", "confidence": 0.38},),
        )


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_missing_model():
    response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "not_ready", "model_loaded": False}


def test_prediction_requires_model():
    response = TestClient(create_app()).post(
        "/predict", files={"file": ("image.png", image_bytes(), "image/png")}
    )

    assert response.status_code == 503


def test_prediction_includes_research_warning():
    response = TestClient(create_app(StubPredictor())).post(
        "/predict", files={"file": ("image.png", image_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json()["candidate_label"] == "candidate-a"
    assert response.json()["accepted"] is False
    assert "Do not use" in response.json()["warning"]


def test_prediction_rejects_non_image_content_type():
    response = TestClient(create_app(StubPredictor())).post(
        "/predict", files={"file": ("notes.txt", b"hello", "text/plain")}
    )

    assert response.status_code == 415
