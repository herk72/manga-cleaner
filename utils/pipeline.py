"""
Manga text cleaning pipeline.

Uses ONLY the seg mask (precise text pixels from the model) + cv2.INPAINT_TELEA.
- seg = exact ink/character pixels detected by the ONNX model
- TELEA propagates surrounding colors inward → white inside bubble → fills white
                                              → art color outside bubble → fills art

This is the same approach used by manga-image-translator internally.
No blk mask used for filling (it over-covers the image).
"""

import os
import logging
import zipfile
import tempfile
from pathlib import Path

import cv2
import numpy as np

from models.detector import get_detector

logger = logging.getLogger(__name__)

JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]


def _save_jpg(image: np.ndarray, output_path: str) -> str:
    jpg_path = str(Path(output_path).with_suffix(".jpg"))
    cv2.imwrite(jpg_path, image, JPEG_PARAMS)
    return jpg_path


def clean_page(image: np.ndarray, seg_mask: np.ndarray | None,
               det_boxes: list[dict]) -> np.ndarray:
    """
    Erase text from manga page.

    1. If we have a seg mask from the model → use it (most accurate).
       TELEA inpaints from surrounding pixels, so:
         - text inside white bubble → fills white ✓
         - text inside black bubble → fills black ✓
         - text on artwork → reconstructs artwork ✓
         - text on gradient/pattern → reconstructs pattern ✓

    2. Fallback: build mask from det bboxes + color-threshold inside each box.
    """
    if seg_mask is not None and seg_mask.max() > 0:
        # Dilate slightly to cover anti-aliased edges of characters
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(seg_mask, k)
        # TELEA with radius 10 fills characters cleanly
        result = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        return result

    # ── Fallback: bbox + threshold ────────────────────────────────────────
    if not det_boxes:
        return image

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for d in det_boxes:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        # Dark pixels in the region = text ink
        _, dark = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], dark)

    if mask.max() == 0:
        return image

    return cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)


def process_single_page(image_path: str, output_path: str,
                         conf_threshold: float = 0.4) -> dict:
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        detector = get_detector()
        detections = detector.detect(img, conf_threshold=conf_threshold)
        logger.info("Detected %d text regions in %s",
                    len(detections), os.path.basename(image_path))

        cleaned = clean_page(img, detector._seg_mask_full, detections)
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
