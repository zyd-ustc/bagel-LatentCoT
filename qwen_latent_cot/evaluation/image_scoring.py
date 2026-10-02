"""CPU preflight and validated Soft-TIFA transport for Phase 1A pNNN/arm.png."""

from io import BytesIO
import json
import math
from pathlib import Path
import pickle

from PIL import Image
import requests

from .geneval2 import load_benchmark, validate_score_lists
from .offline_reader import sha256


PHASE1A_ARMS = ("native", "teacher", "correct", "shuffled", "zero")


def verified_image(path):
    path = Path(path)
    raw = path.read_bytes()
    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG":
            raise ValueError(f"expected PNG image: {path}")
        image.verify()
    return raw


def prepare_phase1a_image_maps(benchmark_path, image_dir, arms):
    benchmark = load_benchmark(benchmark_path)
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(PHASE1A_ARMS):
        raise ValueError("unique supported Phase 1A arms required")
    root = Path(image_dir)
    maps = {arm: {} for arm in arms}
    hashes = {arm: {} for arm in arms}
    dimensions = set()
    for index, item in enumerate(benchmark.prompts):
        folder = root / f"p{index:03d}"
        if (folder / "prompt.txt").read_text().strip() != item.prompt.strip():
            raise ValueError(f"prompt.txt disagrees with benchmark at {folder}")
        for arm in arms:
            image = folder / f"{arm}.png"
            verified_image(image)
            with Image.open(image) as decoded:
                dimensions.add(decoded.size)
            maps[arm][item.prompt] = str(image.resolve())
            hashes[arm][item.prompt] = sha256(image)
    if len(dimensions) != 1:
        raise ValueError("image geometry differs across prompts/arms")
    return dict(schema="bagel-phase1a-image-score-inputs-v1", benchmark_sha256=benchmark.source_sha256,
                prompt_order=[item.prompt for item in benchmark.prompts], num_prompts=benchmark.prompt_count,
                image_maps=maps, image_sha256=hashes, image_shape=list(next(iter(dimensions))),
                verification_scope="prompt/image alignment and geometry; checkpoint/seed/NFE comparability requires generation metadata")


def score_image_map(benchmark_path, image_map_path, *, server_url, batch_size=4,
                    timeout_seconds=600., post=None):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    benchmark = load_benchmark(benchmark_path)
    metadata = [json.loads(line) for line in Path(benchmark_path).read_text().splitlines() if line.strip()]
    if any("vqa_list" not in row for row in metadata):
        raise ValueError("scoring requires VQA questions, not prompt-only reader data")
    paths = json.loads(Path(image_map_path).read_text())
    if not isinstance(paths, dict) or set(paths) != {item.prompt for item in benchmark.prompts}:
        raise ValueError("image map must cover exactly the benchmark prompts")
    resolved = [Path(paths[row["prompt"]]).resolve() for row in metadata]
    if len(set(resolved)) != len(resolved):
        raise ValueError("different prompts cannot reuse the same image path")
    # Validate every image before making any HTTP requests; never silently drop failures.
    images = [verified_image(path) for path in resolved]
    call = requests.post if post is None else post
    atom_scores, log_scores = [], []
    for start in range(0, len(metadata), batch_size):
        batch = metadata[start:start + batch_size]
        response = call(server_url, data=pickle.dumps(dict(images=images[start:start + batch_size],
                       meta_datas=batch, only_strict=True)), timeout=timeout_seconds)
        response.raise_for_status()
        result = pickle.loads(response.content)  # Trusted local Soft-TIFA server only.
        if not isinstance(result, dict):
            raise ValueError("score server must return a dictionary")
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        rows, logs = result.get("atom_scores"), result.get("scores")
        if not isinstance(rows, list) or not isinstance(logs, list) or len(rows) != len(batch) or len(logs) != len(batch):
            raise ValueError("score server batch coverage mismatch")
        if "prompt_order" in result and result["prompt_order"] != [row["prompt"] for row in batch]:
            raise ValueError("score server prompt order mismatch")
        for row, values, log in zip(batch, rows, logs):
            if not isinstance(values, list) or len(values) != len(row["skills"]):
                raise ValueError("score server atom/skill coverage mismatch")
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError("score server invalid probability")
            expected = sum(math.log(max(value, 1e-8)) for value in values) / len(values)
            if isinstance(log, bool) or not isinstance(log, (int, float)) or not math.isfinite(log) or not math.isclose(log, expected, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError("score server log-GM disagrees with atom probabilities")
        atom_scores.extend(rows)
        log_scores.extend(logs)
    validate_score_lists(atom_scores, benchmark)
    return dict(score_lists=atom_scores, log_gm_scores=log_scores,
                benchmark_sha256=benchmark.source_sha256,
                prompt_order=[row["prompt"] for row in metadata],
                provenance=dict(image_map_sha256=sha256(image_map_path), server_url=server_url,
                                image_sha256={row["prompt"]: sha256(path) for row, path in zip(metadata, resolved)}))
