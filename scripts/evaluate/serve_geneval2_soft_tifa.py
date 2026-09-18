#!/usr/bin/env python3
"""Serve GenEval2 per-atom Soft-TIFA rewards over the Flow-GRPO protocol."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


class SoftTIFAScorer:
    def __init__(self, model_path: str, device: str, atom_batch_size: int) -> None:
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map={"": str(self.device)},
        ).eval()
        self.atom_batch_size = int(atom_batch_size)
        if self.atom_batch_size < 1:
            raise ValueError("atom_batch_size must be positive")
        self.lock = threading.Lock()

    def _answer_ids(self, question: str, answer: str) -> list[int]:
        if question.startswith("How many"):
            values = {answer, answer.capitalize(), " " + answer, " " + answer.capitalize()}
            numeric = NUMBER_WORDS.get(answer.lower())
            if numeric:
                values.update({numeric, " " + numeric})
        else:
            values = {"Yes", "yes", " Yes", " yes"}
        ids = set()
        for value in values:
            tokens = self.processor.tokenizer.encode(value, add_special_tokens=False)
            if tokens:
                ids.add(int(tokens[0]))
        return sorted(ids)

    def _atom_probabilities(
        self,
        image: Image.Image,
        vqa_list: list[list[str]],
    ) -> list[float]:
        probabilities = []
        for start in range(0, len(vqa_list), self.atom_batch_size):
            batch = vqa_list[start : start + self.atom_batch_size]
            conversations = [
                [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{question} Answer in one word."},
                    ],
                }]
                for question, _ in batch
            ]
            inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.inference_mode():
                output = self.model(**inputs).logits.float()
                mask = inputs["attention_mask"]
                positions = mask.shape[1] - 1 - mask.flip(1).argmax(-1)
                logits = output[
                    torch.arange(len(batch), device=self.device), positions
                ].softmax(-1)
            for index, (question, answer) in enumerate(batch):
                probability = logits[
                    index, self._answer_ids(str(question), str(answer))
                ].sum()
                probabilities.append(float(probability.clamp(0.0, 1.0)))
        return probabilities

    def score(self, images: list[bytes], metadata: list[dict]) -> dict:
        if len(images) != len(metadata) or not images:
            raise ValueError("images and metadata must have equal non-zero length")
        prompt_scores, atom_scores = [], []
        with self.lock:
            for image_bytes, row in zip(images, metadata):
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                vqa_list = row.get("vqa_list")
                if not vqa_list:
                    raise ValueError("GenEval2 metadata requires a non-empty vqa_list")
                probabilities = self._atom_probabilities(image, vqa_list)
                # Log-GM preserves GenEval2's all-atoms-must-succeed semantics while
                # retaining useful within-group resolution for GRPO.
                log_gm = sum(math.log(max(value, 1e-8)) for value in probabilities) / len(probabilities)
                prompt_scores.append(log_gm)
                atom_scores.append(probabilities)
        return {"scores": prompt_scores, "atom_scores": atom_scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--atom-batch-size", type=int, default=8)
    args = parser.parse_args()
    scorer = SoftTIFAScorer(args.model_path, args.device, args.atom_batch_size)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"status": "ok", "reward": "geneval2_soft_tifa_log_gm"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                length = int(self.headers["Content-Length"])
                payload = pickle.loads(self.rfile.read(length))
                result = scorer.score(payload["images"], payload["meta_datas"])
                body = pickle.dumps(result)
                status = 200
            except Exception as error:
                body = pickle.dumps({"error": repr(error)})
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, message, *values):
            print(message % values, flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"GenEval2 Soft-TIFA listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
