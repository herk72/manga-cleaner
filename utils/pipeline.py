"""
Core pipeline: detection → white-fill → clean JPG.

فقاعات الكلام في المانغا/المانهوا بيضاء دائماً — نملؤها بأبيض خالص.
هذا أسرع وأدق من LaMa الذي صُمِّم للصور الطبيعية.
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

# JPEG params — quality 95
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]


def _save_jpg(image: np.ndarray, output_path: str) -> str:
    """Save as JPEG regardless of output_path extension."""
    jpg_path = str(Path(output_path).with_suffix(".jpg"))
    cv2.imwrite(jpg_path, image, JPEG_PARAMS)
    return jpg_path


def fill_bubbles_white(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Fill speech bubble interiors with pure white.

    Algorithm per detected bbox:
      1. Threshold the ROI to find white/near-white pixels (the bubble)
      2. Morphological close to bridge text gaps inside the bubble
      3. Find the large white contours (actual bubble, not noise)
      4. Flood-fill those contours → solid white patch

    Works for English, Japanese, Korean, etc. — language-agnostic.
    """
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size == 0:
            continue

        roi_h, roi_w = roi_gray.shape
        roi_area = roi_h * roi_w

        # ── Step 1: find light pixels (bubble background) ────────────────
        # threshold at 180 to catch slightly off-white bubbles
        _, light = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)

        # ── Step 2: close gaps caused by dark text inside the bubble ─────
        close_size = max(21, min(roi_h, roi_w) // 4)
        close_size = close_size if close_size % 2 == 1 else close_size + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )
        closed = cv2.morphologyEx(light, cv2.MORPH_CLOSE, kernel)

        # ── Step 3: find bubble contours ──────────────────────────────────
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bubble_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        for c in contours:
            area = cv2.contourArea(c)
            # keep contours that cover at least 8% of the ROI
            if area >= roi_area * 0.08:
                cv2.fillPoly(bubble_mask, [c], 255)

        # If no large contour found, fall back to filling the whole bbox
        if bubble_mask.max() == 0:
            bubble_mask[:] = 255

        # ── Step 4: paint white ───────────────────────────────────────────
        roi_color = result[y1:y2, x1:x2]
        roi_color[bubble_mask > 0] = [255, 255, 255]
        result[y1:y2, x1:x2] = roi_color

    return result


def process_single_page(
    image_path: str, output_path: str, conf_threshold: float = 0.5
) -> dict:
    """
    Clean one manga page.

    Returns:
        dict with keys: output_path, detections_count, success, error
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        detector = get_detector()
        detections = detector.detect(img, conf_threshold=conf_threshold)
        logger.info(
            "Detected %d text regions in %s",
            len(detections),
            os.path.basename(image_path),
        )

        if detections:
            cleaned = fill_bubbles_white(img, detections)
        else:
            cleaned = img

        saved_path = _save_jpg(cleaned, output_path)
        logger.info("Saved cleaned page → %s", saved_path)

        return {
            "output_path": saved_path,
            "detections_count": len(detections),
            "success": True,
            "error": None,
        }

    except Exception as exc:
        logger.exception("pipeline failed for %s", image_path)
        return {"output_path": None, "detections_count": 0, "success": False, "error": str(exc)}


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def process_chapter_zip(zip_path: str, output_dir: str) -> dict:
    """
    Extract a zip, clean each page, rezip with sequential names: 1.jpg, 2.jpg, …
    """
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as extract_dir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        image_files = sorted(
            [
                p
                for p in Path(extract_dir).rglob("*")
                if p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )

        if not image_files:
            return {
                "output_zip": None,
                "total": 0,
                "succeeded": 0,
                "failed": 0,
                "errors": ["No supported images found in zip"],
            }

        results = []
        for idx, img_path in enumerate(image_files, start=1):
            out_path = os.path.join(output_dir, f"{idx}.jpg")
            result = process_single_page(str(img_path), out_path)
            results.append(result)

        output_zip = zip_path.replace(".zip", "_cleaned.zip")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            for r in results:
                if r["success"] and r["output_path"] and os.path.exists(r["output_path"]):
                    zout.write(r["output_path"], os.path.basename(r["output_path"]))

        succeeded = sum(1 for r in results if r["success"])
        failed = len(results) - succeeded
        errors = [r["error"] for r in results if not r["success"] and r["error"]]

        return {
            "output_zip": output_zip,
            "total": len(image_files),
            "succeeded": succeeded,
            "failed": failed,
            "errors": errors,
        }


def process_image_list(image_paths: list[str], output_dir: str) -> dict:
    """Process individual image files — output named 1.jpg, 2.jpg, …"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for idx, img_path in enumerate(image_paths, start=1):
        out_path = os.path.join(output_dir, f"{idx}.jpg")
        result = process_single_page(img_path, out_path)
        results.append(result)

    succeeded = sum(1 for r in results if r["success"])
    output_zip = os.path.join(output_dir, "cleaned_pages.zip")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zout:
        for r in results:
            if r["success"] and r["output_path"] and os.path.exists(r["output_path"]):
                zout.write(r["output_path"], os.path.basename(r["output_path"]))

    return {
        "output_zip": output_zip,
        "total": len(image_paths),
        "succeeded": succeeded,
        "failed": len(image_paths) - succeeded,
        "errors": [r["error"] for r in results if not r["success"] and r["error"]],
    }
