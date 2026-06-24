"""
Comic Text Detector — wraps the comic-text-detector ONNX model.
Outputs: blk (balloon mask), seg (text-pixel mask), det (bboxes)
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
    """Detect text regions using the ONNX model."""

    def __init__(self):
        self._session = None
        # After first run these hold the last inference results:
        self._seg_mask_full: np.ndarray | None = None  # (H_orig, W_orig) uint8 0/255
        self._blk_mask_full: np.ndarray | None = None  # balloon mask, same shape

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
        names = [o.name for o in self._session.get_outputs()]
        logger.info("TextDetector loaded (providers=%s, outputs=%s)",
                    self._session.get_providers(), names)

    def _preprocess(self, image: np.ndarray, target_size: int = 1024):
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w,
                                    cv2.BORDER_CONSTANT, value=255)
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return blob, scale, (h, w), (new_h, new_w)

    def _seg_to_mask(self, seg_out: np.ndarray, orig_h: int, orig_w: int,
                     new_h: int, new_w: int, threshold: float = 0.3) -> np.ndarray:
        """Convert model segmentation output → binary mask in original image space."""
        # seg_out shape: (C, H, W) or (H, W)
        if seg_out.ndim == 3:
            # Combine all channels
            combined = seg_out.max(axis=0)
        else:
            combined = seg_out

        # Crop to valid (non-padded) area then resize to original
        valid = combined[:new_h, :new_w]
        resized = cv2.resize(valid.astype(np.float32), (orig_w, orig_h),
                             interpolation=cv2.INTER_LINEAR)
        binary = (resized > threshold).astype(np.uint8) * 255
        return binary

    def detect(self, image: np.ndarray, conf_threshold: float = 0.4) -> list[dict]:
        """
        Run inference. Stores seg/blk masks internally for create_mask().
        Returns list of dicts: {x1, y1, x2, y2, confidence}
        """
        self._load()
        blob, scale, (orig_h, orig_w), (new_h, new_w) = self._preprocess(image)
        input_name = self._session.get_inputs()[0].name
        out_names = [o.name for o in self._session.get_outputs()]
        outputs = self._session.run(None, {input_name: blob})

        # Map output names
        out_map = {name: arr for name, arr in zip(out_names, outputs)}

        # ── Segmentation masks ─────────────────────────────────────────────
        # 'seg' = text-pixel mask  (most precise — use for inpainting)
        # 'blk' = balloon/bubble region mask
        seg_raw = out_map.get("seg")
        blk_raw = out_map.get("blk")

        if seg_raw is not None:
            arr = seg_raw[0] if seg_raw.ndim == 4 else seg_raw
            self._seg_mask_full = self._seg_to_mask(arr, orig_h, orig_w, new_h, new_w, threshold=0.3)
        else:
            self._seg_mask_full = None

        if blk_raw is not None:
            arr = blk_raw[0] if blk_raw.ndim == 4 else blk_raw
            self._blk_mask_full = self._seg_to_mask(arr, orig_h, orig_w, new_h, new_w, threshold=0.3)
        else:
            self._blk_mask_full = None

        # ── Detection bboxes ───────────────────────────────────────────────
        det_raw = out_map.get("det")
        detections = []
        if det_raw is not None:
            raw = det_raw[0] if det_raw.ndim == 3 else det_raw
            for row in raw:
                if len(row) < 5:
                    continue
                conf = float(row[4])
                if conf < conf_threshold:
                    continue
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                x1 = max(0, int((cx - bw / 2) / scale))
                y1 = max(0, int((cy - bh / 2) / scale))
                x2 = min(orig_w, int((cx + bw / 2) / scale))
                y2 = min(orig_h, int((cy + bh / 2) / scale))
                if x2 > x1 and y2 > y1:
                    detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                       "confidence": conf})

        # Fallback: if no bbox output but seg is present, derive bboxes from seg
        if not detections and self._seg_mask_full is not None:
            detections = self._bboxes_from_mask(self._seg_mask_full)

        return detections

    def _bboxes_from_mask(self, mask: np.ndarray) -> list[dict]:
        """Derive bounding boxes from a binary mask via connected components."""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 50:
                continue
            detections.append({"x1": x, "y1": y, "x2": x + w, "y2": y + h, "confidence": 0.9})
        return detections

    def create_mask(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Return the best available mask for inpainting.

        Priority:
          1. seg mask from model (precise text pixels) — best for all bubble types
          2. blk mask from model (full balloon)
          3. Fallback: bbox rectangles
        """
        h, w = image.shape[:2]

        if self._seg_mask_full is not None and self._seg_mask_full.max() > 0:
            mask = self._seg_mask_full.copy()
            # Light dilation to cover anti-aliased text edges
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            return cv2.dilate(mask, k)

        if self._blk_mask_full is not None and self._blk_mask_full.max() > 0:
            mask = self._blk_mask_full.copy()
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            return cv2.dilate(mask, k)

        # Fallback
        mask = np.zeros((h, w), dtype=np.uint8)
        for d in detections:
            cv2.rectangle(mask, (d["x1"], d["y1"]), (d["x2"], d["y2"]), 255, -1)
        return mask


_detector_instance: TextDetector | None = None


def get_detector() -> TextDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = TextDetector()
    return _detector_instance
