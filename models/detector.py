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
        self._has_mask_output = None  # determined on first run

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
        output_names = [o.name for o in self._session.get_outputs()]
        logger.info("TextDetector loaded (providers=%s, outputs=%s)",
                    self._session.get_providers(), output_names)

    def _preprocess(self, image: np.ndarray, target_size: int = 1024):
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        pad_h = target_size - new_h
        pad_w = target_size - new_w
        padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return blob, scale, (h, w), (new_h, new_w)

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> list[dict]:
        """
        Returns a list of dicts: {x1, y1, x2, y2, confidence}
        Coordinates are in the original image space.
        """
        self._load()

        blob, scale, (orig_h, orig_w), (new_h, new_w) = self._preprocess(image)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})

        # Determine which output contains the YOLO boxes
        # Model may output: [mask(1,2,H,W), boxes(1,N,5+)] or [boxes(1,N,5+), ...]
        boxes_output = None
        self._seg_mask = None  # store segmentation mask if available

        for i, out in enumerate(outputs):
            arr = out
            if arr.ndim == 4:
                # This is a segmentation mask (1, C, H, W)
                if self._has_mask_output is None:
                    logger.info("Found segmentation output at index %d, shape=%s", i, arr.shape)
                self._has_mask_output = True
                self._seg_mask = arr[0]  # (C, H, W)
                self._seg_new_h = new_h
                self._seg_new_w = new_w
            elif arr.ndim == 3:
                # YOLO detections (1, N, 5+)
                boxes_output = arr[0]

        if boxes_output is None and len(outputs) > 0:
            # Fallback: first output as boxes
            raw = outputs[0]
            if raw.ndim == 3:
                boxes_output = raw[0]
            elif raw.ndim == 2:
                boxes_output = raw

        if self._has_mask_output is None:
            self._has_mask_output = False

        detections = []
        if boxes_output is not None:
            for row in boxes_output:
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
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                if x2 > x1 and y2 > y1:
                    detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

        return detections

    def create_mask(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Returns a binary mask (same HxW as image) — white = region to erase.

        Priority:
        1. Use segmentation mask from model (most accurate)
        2. Use bubble-shape detection within each bbox (color-based)
        """
        h, w = image.shape[:2]

        # ── Option 1: use the model's segmentation output ─────────────────────
        if self._has_mask_output and self._seg_mask is not None:
            return self._mask_from_segmentation(h, w)

        # ── Option 2: precise bubble mask via color thresholding ──────────────
        return self._mask_from_bubbles(image, detections)

    def _mask_from_segmentation(self, orig_h: int, orig_w: int) -> np.ndarray:
        """Resize and threshold the model's segmentation output."""
        # seg_mask shape: (C, H, W)  — channel 0=text, channel 1=bubble region
        # Use the channel with the highest values (bubble region tends to be channel 1)
        seg = self._seg_mask
        if seg.shape[0] >= 2:
            combined = np.maximum(seg[0], seg[1])
        else:
            combined = seg[0]

        # Resize from padded 1024 space → original dims
        # We need only the valid (non-padded) area
        valid = combined[:self._seg_new_h, :self._seg_new_w]
        resized = cv2.resize(valid, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Threshold at 0.5
        mask = (resized > 0.5).astype(np.uint8) * 255

        # Light dilation to cover edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel)
        return mask

    def _mask_from_bubbles(self, image: np.ndarray, detections: list[dict],
                           dilation: int = 4) -> np.ndarray:
        """
        Precise bubble-shape mask using color thresholding inside each bbox.
        Manga/manhwa speech bubbles are white (or near-white) regions.
        This avoids masking non-text background content.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for d in detections:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # ── Find light (white) regions = bubble content ────────────────
            # Use adaptive threshold to handle slight off-white bubbles
            _, light = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

            # Close small gaps so the bubble interior is connected
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            closed = cv2.morphologyEx(light, cv2.MORPH_CLOSE, close_k)

            # Fill the largest contour (the bubble shape)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bubble_roi = np.zeros_like(roi)
            if contours:
                # Keep contours that cover > 10% of the ROI area (ignore tiny artifacts)
                roi_area = roi.shape[0] * roi.shape[1]
                big = [c for c in contours if cv2.contourArea(c) > roi_area * 0.05]
                if not big:
                    big = contours  # fall back to all if none qualify
                cv2.fillPoly(bubble_roi, big, 255)

            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], bubble_roi)

        # Light dilation to cover bubble borders / partial pixels
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
