"""
Core pipeline: detection → inpaint → JPG output.

يستخدم seg mask الدقيق من الموديل + cv2.inpaint (TELEA) للتبييض.
TELEA يعمل لكل أنواع الفقاعات: بيضاء، سوداء، شفافة، منقطة، ملوّنة.
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


def _inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    OpenCV TELEA inpainting — يملأ المنطقة المحجوبة بالألوان المحيطة.
    يعمل لكل أنواع الخلفيات: أبيض، أسود، شفاف، نقاط، ألوان متدرجة.
    لا يحتاج GPU ولا IOPaint.
    """
    # تأكد أن الـ mask ثنائي
    mask_gray = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    if binary.max() == 0:
        return image

    return cv2.inpaint(image, binary, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def process_single_page(
    image_path: str, output_path: str, conf_threshold: float = 0.4
) -> dict:
    """
    Clean one manga page. Returns dict: {output_path, detections_count, success, error}
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        detector = get_detector()
        detections = detector.detect(img, conf_threshold=conf_threshold)
        logger.info("Detected %d text regions in %s",
                    len(detections), os.path.basename(image_path))

        if detections:
            mask = detector.create_mask(img, detections)
            cleaned = _inpaint(img, mask)
        else:
            cleaned = img

        saved_path = _save_jpg(cleaned, output_path)
        logger.info("Saved → %s", saved_path)

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
    Extract ZIP, clean each page, rezip as 1.jpg, 2.jpg, …
    """
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
            result = process_single_page(str(img_path), out_path)
            results.append(result)

        output_zip = zip_path.replace(".zip", "_cleaned.zip")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            for r in results:
                if r["success"] and r["output_path"] and os.path.exists(r["output_path"]):
                    zout.write(r["output_path"], os.path.basename(r["output_path"]))

        succeeded = sum(1 for r in results if r["success"])
        failed = len(results) - succeeded

        return {
            "output_zip": output_zip,
            "total": len(image_files),
            "succeeded": succeeded,
            "failed": failed,
            "errors": [r["error"] for r in results if not r["success"] and r["error"]],
        }


def process_image_list(image_paths: list[str], output_dir: str) -> dict:
    """Process individual images — output named 1.jpg, 2.jpg, …"""
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
