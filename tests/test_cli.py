from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class CliTests(unittest.TestCase):
    def test_missing_api_key_is_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "pixel.png"
            kb = root / "kb.json"
            image.write_bytes(PNG_1X1)
            kb.write_text(
                '[{"id":"x","title":"X","text":"Knowledge text"}]',
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.pop("KBVQA_API_KEY", None)
            env.pop("KBVQA_MODEL", None)
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kbvqa",
                    "answer",
                    "--image",
                    str(image),
                    "--question",
                    "What is shown?",
                    "--kb",
                    str(kb),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(process.returncode, 2)
        self.assertIn("KBVQA_API_KEY", process.stderr)


if __name__ == "__main__":
    unittest.main()
