"""
Core pipeline — radical manga text cleaning.

Strategy:
  1. blk mask (balloon regions) → detect actual background color per balloon → fill clean
  2. seg mask outside balloons (text on artwork) → TELEA inpainting
  3. Output: JPEG 95%
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


def _detect_balloon_bg_color(image: np.ndarray, balloon_mask: np.ndarray,
                              text_mask: np.ndarray | None) -> np.ndarray:
    """
    Sample the background color of a balloon region.
    Uses pixels inside the balloon that are NOT text (i.e., the actual background).
    Falls back to the dominant color in the border ring if needed.
    """
    balloon_bool = balloon_mask > 0

    if text_mask is not None:
        bg_pixels_mask = balloon_bool & (text_mask == 0)
    else:
        bg_pixels_mask = balloon_bool

    if bg_pixels_mask.sum() > 20:
        colors = image[bg_pixels_mask]
        # Use median for robustness against remaining text pixels
        return np.median(colors, axis=0).astype(np.uint8)

    # Fallback: sample from a thin ring around the balloon boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded = cv2.erode(balloon_mask, kernel)
    ring = balloon_mask & ~eroded
    if ring.any():
        colors = image[ring > 0]
        return np.median(colors, axis=0).astype(np.uint8)

    return np.array([255, 255, 255], dtype=np.uint8)  # last resort: white


def clean_page(image: np.ndarray,
               blk_mask: np.ndarray | None,
               seg_mask: np.ndarray | None,
               det_mask: np.ndarray | None) -> np.ndarray:
    """
    Two-pass manga text cleaning:

    Pass 1 — Balloons (blk_mask):
        Each connected balloon region gets filled with its own detected background color.
        Works for: white, black, transparent, patterned, gradient bubbles.

    Pass 2 — Text on artwork (seg outside blk):
        TELEA inpainting reconstructs the underlying artwork.
    """
    result = image.copy()

    # ── Pass 1: Balloon regions ──────────────────────────────────────────
    working_blk = blk_mask if (blk_mask is not None and blk_mask.max() > 0) else None

    if working_blk is not None:
        # Dilate blk slightly to cover balloon borders/outlines
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blk_dilated = cv2.dilate(working_blk, dk)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            blk_dilated, connectivity=8
        )
        h, w = image.shape[:2]

        for label_id in range(1, num):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 100:
                continue  # skip noise

            balloon_mask_i = (labels == label_id).astype(np.uint8) * 255

            # Get text pixels inside this balloon
            text_in_balloon = None
            if seg_mask is not None:
                text_in_balloon = (balloon_mask_i > 0) & (seg_mask > 0)

            bg_color = _detect_balloon_bg_color(image, balloon_mask_i, text_in_balloon)
            result[balloon_mask_i > 0] = bg_color

    # ── Pass 2: Text outside balloons ────────────────────────────────────
    if seg_mask is not None and seg_mask.max() > 0:
        if working_blk is not None:
            outside_text = (seg_mask > 127) & (blk_mask == 0)
        else:
            outside_text = seg_mask > 127

        if outside_text.any():
            outside_mask = outside_text.astype(np.uint8) * 255
            # Dilate to cover anti-aliased edges
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            outside_mask = cv2.dilate(outside_mask, ek)
            result = cv2.inpaint(result, outside_mask, 7, cv2.INPAINT_TELEA)

    # ── Fallback: if no masks available, use det bbox rects ─────────────
    elif det_mask is not None and det_mask.max() > 0 and working_blk is None:
        result = cv2.inpaint(result, det_mask, 7, cv2.INPAINT_TELEA)

    return result


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

        if detections or (
            detector._seg_mask_full is not None and detector._seg_mask_full.max() > 0
        ) or (
            detector._blk_mask_full is not None and detector._blk_mask_full.max() > 0
        ):
            # Build a fallback det mask from bboxes (used only if both blk+seg are empty)
            det_mask = None
            if detections and detector._blk_mask_full is None and detector._seg_mask_full is None:
                h, w = img.shape[:2]
                det_mask = np.zeros((h, w), dtype=np.uint8)
                for d in detections:
                    cv2.rectangle(det_mask, (d["x1"], d["y1"]),
                                  (d["x2"], d["y2"]), 255, -1)

            cleaned = clean_page(
                img,
                blk_mask=detector._blk_mask_full,
                seg_mask=detector._seg_mask_full,
                det_mask=det_mask,
            )
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
            result = process_single_page(str(img_path), out_path)
            results.append(result)

        output_zip = zip_path.replace(".zip", "_cleaned.zip")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            for r in results:
                if r["success"] and r["output_path"] and os.path.exists(r["output_path"]):
                    zout.write(r["output_path"], os.path.basename(r["output_path"]))

        succeeded = sum(1 for r in results if r["success"])
        return {
            "output_zip": output_zip,
            "total": len(image_files),
            "succeeded": succeeded,
            "failed": len(image_files) - succeeded,
            "errors": [r["error"] for r in results if not r["success"] and r["error"]],
        }


def process_image_list(image_paths: list[str], output_dir: str) -> dict:
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
