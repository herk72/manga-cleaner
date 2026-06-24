"""
Manga text cleaning pipeline — final correct approach.

  seg mask (pixel-precise text mask from ONNX)
      ↓ dilate 3px to cover anti-aliasing
      ↓
  IOPaint/LaMa  ← AI fills with surrounding context perfectly
                  works for: white/black/gradient/pattern bubbles,
                             text on artwork, SFX, all types
      ↓
  Fallback: cv2.INPAINT_TELEA if LaMa unreachable

Output: JPEG 95%
"""

import io
import os
import logging
import zipfile
import tempfile
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image

from models.detector import get_detector

logger = logging.getLogger(__name__)

JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]


def _save_jpg(image: np.ndarray, output_path: str) -> str:
    jpg_path = str(Path(output_path).with_suffix(".jpg"))
    cv2.imwrite(jpg_path, image, JPEG_PARAMS)
    return jpg_path


def _ndarray_to_png(arr: np.ndarray) -> bytes:
    pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr.ndim == 3 else arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _lama_inpaint(image: np.ndarray, mask: np.ndarray, iopaint_url: str,
                  timeout: float = 120.0) -> np.ndarray:
    """Call IOPaint/LaMa API. Returns inpainted image or raises on failure."""
    resp = httpx.post(
        f"{iopaint_url.rstrip('/')}/inpaint",
        files={
            "image": ("image.png", _ndarray_to_png(image), "image/png"),
            "mask":  ("mask.png",  _ndarray_to_png(mask),  "image/png"),
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _telea_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """CPU fallback — fast but less accurate on complex backgrounds."""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, binary, inpaintRadius=10, flags=cv2.INPAINT_TELEA)


def _build_mask(detector, det_boxes: list[dict], image_shape: tuple) -> np.ndarray | None:
    """
    Get the best available mask:
    1. seg mask from model (precise text pixels) ← preferred
    2. det bboxes with Otsu threshold (dark ink pixels in each box)
    Returns None if nothing detected.
    """
    # Precise pixel mask from model
    seg = detector._seg_mask_full
    if seg is not None and seg.max() > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.dilate(seg, k)

    # Fallback: find dark text pixels inside each detection box
    if not det_boxes:
        return None

    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # We can't access the image here; caller will build this mask if needed
    return None  # handled in process_single_page


def clean_page(image: np.ndarray, seg_mask: np.ndarray | None,
               det_boxes: list[dict], iopaint_url: str) -> np.ndarray:
    """
    Erase text from one page using the best available method.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── Build mask ────────────────────────────────────────────────────────
    if seg_mask is not None and seg_mask.max() > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(seg_mask, k)
        method = "seg+lama"
    elif det_boxes:
        # Otsu threshold inside each detected bbox → dark ink pixels
        mask = np.zeros((h, w), dtype=np.uint8)
        for d in det_boxes:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            _, dark = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dark = cv2.dilate(dark, k)
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], dark)
        if mask.max() == 0:
            return image
        method = "bbox-otsu+lama"
    else:
        return image

    # ── Try LaMa first, fall back to TELEA ───────────────────────────────
    try:
        result = _lama_inpaint(image, mask, iopaint_url)
        logger.info("Cleaned with LaMa (%s)", method)
        return result
    except Exception as exc:
        logger.warning("LaMa unavailable (%s) → TELEA fallback", exc)
        return _telea_inpaint(image, mask)


def process_single_page(image_path: str, output_path: str,
                         conf_threshold: float = 0.4) -> dict:
    try:
        from app.config import IOPAINT_URL
    except Exception:
        IOPAINT_URL = os.getenv("IOPAINT_URL", "http://iopaint:8080")

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        detector = get_detector()
        detections = detector.detect(img, conf_threshold=conf_threshold)
        logger.info("Detected %d text regions in %s",
                    len(detections), os.path.basename(image_path))

        if detections or (detector._seg_mask_full is not None
                          and detector._seg_mask_full.max() > 0):
            cleaned = clean_page(img, detector._seg_mask_full, detections, IOPAINT_URL)
        else:
            cleaned = img

        saved_path = _save_jpg(cleaned, output_path)
        logger.info("Saved → %s", saved_path)
        return {"output_path": saved_path, "detections_count": len(detections),
                "success": True, "error": None}

    except Exception as exc:
        logger.exception("pipeline failed for %s", image_path)
        return {"output_path": None, "detections_count": 0,
                "success": False, "error": str(exc)}


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def process_chapter_zip(zip_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as extract_dir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        image_files = sorted([
            p for p in Path(extract_dir).rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])
        if not image_files:
            return {"output_zip": None, "total": 0, "succeeded": 0, "failed": 0,
                    "errors": ["No supported images found in zip"]}

        results = []
        for idx, img_path in enumerate(image_files, start=1):
            out_path = os.path.join(output_dir, f"{idx}.jpg")
            results.append(process_single_page(str(img_path), out_path))

        output_zip = zip_path.replace(".zip", "_cleaned.zip")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            for r in results:
                if r["success"] and r["output_path"] and os.path.exists(r["output_path"]):
                    zout.write(r["output_path"], os.path.basename(r["output_path"]))

        succeeded = sum(1 for r in results if r["success"])
        return {
            "output_zip": output_zip, "total": len(image_files),
            "succeeded": succeeded, "failed": len(image_files) - succeeded,
            "errors": [r["error"] for r in results if not r["success"] and r["error"]],
        }


def process_image_list(image_paths: list[str], output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for idx, img_path in enumerate(image_paths, start=1):
        out_path = os.path.join(output_dir, f"{idx}.jpg")
        results.append(process_single_page(img_path, out_path))

    succeeded = sum(1 for r in results if r["success"])
    output_zip = os.path.join(output_dir, "cleaned_pages.zip")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zout:
        for r in results:
            if r["success"] and r["output_path"] and os.path.exists(r["output_path"]):
                zout.write(r["output_path"], os.path.basename(r["output_path"]))

    return {
        "output_zip": output_zip, "total": len(image_paths),
        "succeeded": succeeded, "failed": len(image_paths) - succeeded,
        "errors": [r["error"] for r in results if not r["success"] and r["error"]],
    }
