"""Image helpers for multimodal chat requests."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


class ImageInputError(ValueError):
    """Raised when an image path cannot be used as model input."""


def image_to_data_url(path: str | Path) -> str:
    """Convert a local image path to a base64 data URL."""

    image_path = Path(path)
    if not image_path.exists():
        raise ImageInputError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise ImageInputError(f"Image path is not a file: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None or not mime_type.startswith("image/"):
        raise ImageInputError(
            f"Could not infer a supported image MIME type for: {image_path}"
        )

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
