"""
Inpainting via IOPaint (formerly lama-cleaner).
Calls the IOPaint REST API to erase text and redraw background.
"""

import io
import logging
import numpy as np
import cv2
import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class IOPaintClient:
    """Thin client for the IOPaint /inpaint endpoint."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def _ndarray_to_png_bytes(self, arr: np.ndarray) -> bytes:
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr.ndim == 3 else arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        timeout: float = 120.0,
    ) -> np.ndarray:
        """
        Send image + mask to IOPaint, return the cleaned image as ndarray (BGR).
        Falls back to a simple OpenCV inpaint if IOPaint is unreachable.
        """
        img_bytes = self._ndarray_to_png_bytes(image)
        mask_bytes = self._ndarray_to_png_bytes(mask)

        try:
            resp = httpx.post(
                f"{self.base_url}/inpaint",
                files={
                    "image": ("image.png", img_bytes, "image/png"),
                    "mask": ("mask.png", mask_bytes, "image/png"),
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            result_pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
            result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
            return result_bgr

        except Exception as exc:
            logger.warning("IOPaint unavailable (%s) — falling back to OpenCV inpaint", exc)
            return self._opencv_fallback(image, mask)

    def _opencv_fallback(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use OpenCV Navier-Stokes inpaint as a CPU-only fallback."""
        mask_gray = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        result = cv2.inpaint(image, binary_mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
        return result


_client_instance: IOPaintClient | None = None


def get_inpainter(base_url: str | None = None) -> IOPaintClient:
    global _client_instance
    if _client_instance is None:
        from app.config import IOPAINT_URL
        _client_instance = IOPaintClient(base_url or IOPAINT_URL)
    return _client_instance
