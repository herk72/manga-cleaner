"""
Comic Text Detector — wraps the comic-text-detector ONNX model.
Downloads the model weights on first run from the official release.
"""

import os
import urllib.request
import logging
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/"
    "beta-0.3/comictextdetector.pt.onnx"
)
FALLBACK_MODEL_URL = (
    "https://huggingface.co/dreMaz/AnimeInstanceSegmentation/resolve/main/"
    "comictextdetector.pt.onnx"
)


def get_model_path() -> Path:
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir / "comictextdetector.pt.onnx"


def download_model():
    model_path = get_model_path()
    if model_path.exists():
        return model_path

    logger.info("Downloading comic-text-detector model …")
    for url in [MODEL_URL, FALLBACK_MODEL_URL]:
        try:
            urllib.request.urlretrieve(url, str(model_path))
            logger.info("Model downloaded to %s", model_path)
            return model_path
        except Exception as exc:
            logger.warning("Download from %s failed: %s", url, exc)

    raise RuntimeError("Could not download comic-text-detector model.")


class TextDetector:
    """Detect speech bubbles / text regions using the ONNX model."""

    def __init__(self):
        self._session = None

    def _load(self):
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("onnxruntime is not installed.")

        model_path = download_model()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        logger.info("TextDetector loaded (providers=%s)", self._session.get_providers())

    def _preprocess(self, image: np.ndarray, target_size: int = 1024):
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        pad_h = target_size - new_h
        pad_w = target_size - new_w
        padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return blob, scale, (h, w)

    def detect(self, image: np.ndarray, conf_threshold: float = 0.3) -> list[dict]:
        """
        Returns a list of dicts: {x1, y1, x2, y2, confidence}
        Coordinates are in the original image space.
        """
        self._load()

        blob, scale, (orig_h, orig_w) = self._preprocess(image)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})

        detections = []
        raw = outputs[0]

        if raw.ndim == 3:
            raw = raw[0]

        for row in raw:
            if len(row) < 5:
                continue
            conf = float(row[4])
            if conf < conf_threshold:
                continue
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            x1 = int((cx - bw / 2) / scale)
            y1 = int((cy - bh / 2) / scale)
            x2 = int((cx + bw / 2) / scale)
            y2 = int((cy + bh / 2) / scale)
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

        return detections

    def create_mask(self, image: np.ndarray, detections: list[dict], dilation: int = 8) -> np.ndarray:
        """Returns a binary mask (same HxW as image) — white = text region."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for d in detections:
            cv2.rectangle(mask, (d["x1"], d["y1"]), (d["x2"], d["y2"]), 255, -1)

        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
            mask = cv2.dilate(mask, kernel)

        return mask


_detector_instance: TextDetector | None = None


def get_detector() -> TextDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = TextDetector()
    return _detector_instance
