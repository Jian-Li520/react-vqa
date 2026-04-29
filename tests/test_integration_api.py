from __future__ import annotations

import base64
import os
import tempfile
import unittest
from pathlib import Path

from kbvqa import KBVQAAgent


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@unittest.skipUnless(
    os.getenv("KBVQA_RUN_INTEGRATION") == "1"
    and os.getenv("KBVQA_API_KEY")
    and os.getenv("KBVQA_MODEL"),
    "set KBVQA_RUN_INTEGRATION=1, KBVQA_API_KEY, and KBVQA_MODEL to run",
)
class RealApiIntegrationTests(unittest.TestCase):
    def test_real_api_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "pixel.png"
            kb = root / "kb.json"
            image.write_bytes(PNG_1X1)
            kb.write_text(
                '[{"id":"pixel","title":"Tiny PNG","text":"A one-pixel PNG test image is usually not enough to identify real-world objects."}]',
                encoding="utf-8",
            )

            result = KBVQAAgent.from_env(kb, max_steps=2, top_k=1).answer(
                image,
                "Can you identify a real-world object in this image?",
            )

        self.assertIn(result["confidence"], {"low", "medium", "high"})
        self.assertTrue(result["answer"])


if __name__ == "__main__":
    unittest.main()
