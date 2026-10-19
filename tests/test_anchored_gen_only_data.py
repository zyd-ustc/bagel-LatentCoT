"""GEN-only reads the image-paired prompt from the annotation index."""
import json
from pathlib import Path

import pytest

from qwen_latent_cot.anchored.gen_only_data import (
    GenerationPromptSource,
    open_curriculum_val,
    open_gen_only_source,
)
from qwen_latent_cot.diagnostics.corpus import REF, Corpus, build


def paired_row(sample_id, image, prompt, caption):
    return {
        "sample_id": sample_id,
        "image": image,
        "prompt": prompt,
        "caption": caption,
        "target_caption": caption,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def build_index(tmp_path):
    train_a = tmp_path / "a.jsonl"
    train_b = tmp_path / "b.jsonl"
    val = tmp_path / "val.jsonl"
    write_jsonl(train_a, [
        paired_row("a0", str(tmp_path / "a.png"), "  prompt-a  ", "caption-a"),
    ])
    write_jsonl(train_b, [
        paired_row(f"b{i}", str(tmp_path / f"b{i}.png"), f"prompt-b{i}", f"caption-b{i}")
        for i in range(4)
    ])
    write_jsonl(val, [
        paired_row("v0", str(tmp_path / "v.png"), "prompt-val", "caption-val"),
    ])
    root = tmp_path / "corpus"
    build({
        "sources": [
            {"name": "train-a", "format": "paired", "path": str(train_a), "split": "train"},
            {"name": "train-b", "format": "paired", "path": str(train_b), "split": "train"},
            {"name": "held", "format": "paired", "path": str(val), "split": "val"},
        ],
    }, root)
    for observe in (root / "views" / "observe").rglob("*.idx"):
        observe.unlink()
    return root


def test_generation_train_uses_goal_prompt_and_skips_other_views(tmp_path):
    root = build_index(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["format"] = "annotation_index_v1"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rebuild corpus"):
        Corpus(root)
    source = GenerationPromptSource(root, 17)
    try:
        assert source.text_field == "goal_prompt"
        assert source.pair_count == 5
        assert source.fingerprint == manifest["fingerprint"]
        assert all(path.startswith("generation/") and path.endswith("/train.idx") for path in (
            view.info["path"] for view in source.views
        ))
        rows = source.take(12)
        captions = {row["caption"] for row in rows}
        assert "prompt-a" in captions
        assert captions <= {"prompt-a", *(f"prompt-b{i}" for i in range(4))}
        assert all(row["prompt"] == row["caption"] for row in rows)
        assert all(row["caption"] != "caption-a" and not row["caption"].startswith("caption-") for row in rows)
        assert all(Path(row["image"]).is_absolute() for row in rows)
        state = source.state_dict()
        assert "order" not in state
        assert len(state["cursors"]) == len(source.views)
        expected = source.take(7)
        resumed = GenerationPromptSource(root, 99)
        try:
            resumed.load_state_dict(state)
            assert resumed.take(7) == expected
        finally:
            resumed.close()
    finally:
        source.close()


def test_empty_generation_train_split_is_refused(tmp_path):
    val = tmp_path / "val.jsonl"
    write_jsonl(val, [paired_row("v0", str(tmp_path / "v.png"), "prompt-val", "caption-val")])
    root = tmp_path / "corpus"
    build({
        "sources": [{"name": "held", "format": "paired", "path": str(val), "split": "val"}],
    }, root)
    with pytest.raises(ValueError, match="generation train split is empty"):
        GenerationPromptSource(root, 1)


def test_blank_goal_prompt_and_bad_role_are_refused(tmp_path):
    root = tmp_path / "index"
    records = root / "records"
    views = root / "views" / "generation" / "tiny"
    records.mkdir(parents=True)
    views.mkdir(parents=True)
    row = {
        "goal_prompt": "   ",
        "target_caption": "caption-not-used",
        "target_image": str(tmp_path / "cup.png"),
        "original": {"path": str(tmp_path / "source.jsonl")},
    }
    (records / "000000.jsonl").write_text(json.dumps(row) + "\n")
    (views / "train.idx").write_bytes(REF.pack(0, 0, 2))
    manifest = {
        "format": "annotation_index_v1",
        "fingerprint": "abc",
        "source_sampling": "sqrt_task_records",
        "registry": {"roots": {}},
        "shards": [{"path": "000000.jsonl"}],
        "views": [{
            "path": "generation/tiny/train.idx",
            "task": "generation",
            "split": "train",
            "count": 1,
        }],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    source = GenerationPromptSource(root, 1)
    try:
        with pytest.raises(ValueError, match="goal_prompt"):
            source.take(1)
    finally:
        source.close()
    (views / "train.idx").write_bytes(REF.pack(0, 0, 0))
    source = GenerationPromptSource(root, 1)
    try:
        with pytest.raises(ValueError, match="role 2"):
            source.take(1)
    finally:
        source.close()


def test_generation_val_split_is_separate_from_train(tmp_path):
    root = build_index(tmp_path)
    source = open_curriculum_val(root, None, 19)
    try:
        assert source.split == "val"
        assert source.pair_count == 1
        row = source.take(1)[0]
        assert row["caption"] == "prompt-val"
        assert row["caption"] != "caption-val"
    finally:
        source.close()
    with pytest.raises(ValueError, match="held-out"):
        open_gen_only_source(tmp_path / "unused.jsonl", 1, split="val")


def test_jsonl_path_still_uses_its_caption(tmp_path):
    image = tmp_path / "dog.png"
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({
        "caption": "  a red dog  ",
        "image": str(image),
    }) + "\n")
    source = open_gen_only_source(path, 3)
    try:
        assert source.text_field == "caption"
        assert source.fingerprint is None
        assert source.pair_count == 1
        row = source.take(1)[0]
        assert row["caption"] == "a red dog"
        assert row["image"] == str(image)
        state = source.state_dict()
        again = open_gen_only_source(path, 3)
        again.load_state_dict(state)
        assert again.take(1)[0]["caption"] == source.take(1)[0]["caption"]
    finally:
        source.close()
    with pytest.raises(ValueError, match="missing"):
        open_gen_only_source(tmp_path / "missing.jsonl", 1)
