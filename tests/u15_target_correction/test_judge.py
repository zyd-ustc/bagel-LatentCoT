"""External judge entry: skip completed rows, and do not read the key from disk."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from qwen_latent_cot.u15_target_correction import judge as judge_mod


class _Usage:
    def model_dump(self):
        return {"total_tokens": 3}


class _Response:
    class _Message:
        content = "```json\n" + json.dumps({
            "requirements": [{"id": "q0", "verdict": "yes"}],
            "structural_integrity": "intact",
            "preservation": "kept",
            "edit_instruction": "not_applicable",
            "accepted": True,
        }) + "\n```"

    choices = [type("Choice", (), {"message": _Message()})()]
    usage = _Usage()


class _Client:
    calls = 0

    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

    def create(self, **kwargs):
        type(self).calls += 1
        self.last = kwargs
        return _Response()

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self


class JudgeEntry(unittest.TestCase):
    def setUp(self):
        self._openai = judge_mod.__dict__.get("openai")
        judge_mod_openai = type("openai", (), {"OpenAI": _Client})
        import sys
        self._sys_openai = sys.modules.get("openai")
        sys.modules["openai"] = judge_mod_openai
        _Client.calls = 0

    def tearDown(self):
        import sys
        if self._sys_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = self._sys_openai

    def _request(self, root, index=7):
        image = root / "bird.png"
        Image.new("RGB", (8, 8), (20, 40, 60)).save(image)
        path = root / "requests.jsonl"
        row = {
            "index": index,
            "uid": "case-7",
            "instruction": "Five stone birds.",
            "images": [{"role": "candidate", "path": "bird.png"}],
        }
        path.write_text(json.dumps(row) + "\n")
        return path

    def test_writes_json_and_skips_an_existing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = self._request(root)
            out = root / "out"
            os.environ["TARGET_CORRECTION_TEST_KEY"] = "not-a-real-key"
            judge_mod.judge(requests, out, "https://example.invalid/v1", "glm-5.3-flash",
                            "TARGET_CORRECTION_TEST_KEY")
            written = json.loads((out / "000007.json").read_text())
            self.assertEqual(written["uid"], "case-7")
            self.assertEqual(written["model"], "glm-5.3-flash")
            self.assertIn("accepted", written["response"])
            self.assertEqual(_Client.calls, 1)
            judge_mod.judge(requests, out, "https://example.invalid/v1", "glm-5.3-flash",
                            "TARGET_CORRECTION_TEST_KEY")
            self.assertEqual(_Client.calls, 1)

    def test_missing_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = self._request(root, index=1)
            os.environ.pop("TARGET_CORRECTION_ABSENT_KEY", None)
            with self.assertRaises(KeyError):
                judge_mod.judge(requests, root / "out", "https://example.invalid/v1",
                                "glm-5.3-flash", "TARGET_CORRECTION_ABSENT_KEY")


if __name__ == "__main__":
    unittest.main()
