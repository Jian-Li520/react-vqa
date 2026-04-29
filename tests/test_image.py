from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from kbvqa.image import ImageInputError, image_to_data_url


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class ImageHelperTests(unittest.TestCase):
    def test_image_to_data_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pixel.png"
            path.write_bytes(PNG_1X1)

            data_url = image_to_data_url(path)

            self.assertTrue(data_url.startswith("data:image/png;base64,"))
            self.assertIn(base64.b64encode(PNG_1X1).decode("ascii"), data_url)

    def test_missing_image_raises(self):
        with self.assertRaises(ImageInputError):
            image_to_data_url("/no/such/image.png")


if __name__ == "__main__":
    unittest.main()
