"""
Core pipeline: detection → mask → inpainting → clean image.
"""

import os
import logging
import zipfile
import tempfile
from pathlib import Path

import cv2
import numpy as np

from models.detector import get_detector
from models.inpainter import get_inpainter

logger = logging.getLogger(__name__)


def process_single_page(image_path: str, output_path: str, conf_threshold: float = 0.3) -> dict:
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
        logger.info("Detected %d text regions in %s", len(detections), os.path.basename(image_path))

        if not detections:
            cv2.imwrite(output_path, img)
            return {"output_path": output_path, "detections_count": 0, "success": True, "error": None}

        mask = detector.create_mask(img, detections)

        inpainter = get_inpainter()
        cleaned = inpainter.inpaint(img, mask)

        cv2.imwrite(output_path, cleaned)
        logger.info("Saved cleaned page → %s", output_path)

        return {
            "output_path": output_path,
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
    Extract a zip of manga pages, process each image, and re-zip the results.

    Returns:
        dict with keys: output_zip, total, succeeded, failed, errors
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
        for img_path in image_files:
            out_name = img_path.name
            out_path = os.path.join(output_dir, out_name)
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
    """Process a list of individual image files."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for img_path in image_paths:
        out_path = os.path.join(output_dir, os.path.basename(img_path))
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
