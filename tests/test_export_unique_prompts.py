import json

import pytest

from scripts.data.export_unique_prompts import export_unique_prompts, sha256_file


def test_preserves_first_id_order_and_exact_prompt_text(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [dict(sample_id="first",prompt=" A",source="one"),
            dict(sample_id="second",prompt=" A",source="two"),
            dict(sample_id="third",prompt="A",source="three")]
    source.write_text("\n".join(map(json.dumps, rows))+"\n")
    before = sha256_file(source)
    summary = export_unique_prompts(source, tmp_path / "export")
    result = [json.loads(line) for line in (tmp_path/"export/prompts.jsonl").read_text().splitlines()]
    assert [r["id"] for r in result] == ["first","third"]
    assert [r["prompt"] for r in result] == [" A","A"]
    assert result[0]["duplicate_count"] == 1 and result[0]["source_line"] == 1
    assert summary["input_records"] == 3 and summary["unique_prompts"] == 2
    assert summary["skipped_duplicate_rows"] == 1
    assert sha256_file(source) == before == summary["source_sha256"]
    with pytest.raises(FileExistsError):
        export_unique_prompts(source, tmp_path/"export")


@pytest.mark.parametrize("rows", [[dict(sample_id="a",prompt=" ")],
    [dict(prompt="a")], [dict(sample_id="a",prompt="a"),dict(sample_id="a",prompt="b")]])
def test_rejects_invalid_inputs_before_writing(tmp_path, rows):
    source = tmp_path/"source.jsonl"
    source.write_text("\n".join(map(json.dumps,rows))+"\n")
    with pytest.raises(ValueError):
        export_unique_prompts(source,tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_heldout_overlap_is_not_silently_filtered(tmp_path):
    source = tmp_path/"source.jsonl"
    source.write_text('{"sample_id":"one","prompt":"same"}\n')
    heldout = tmp_path/"heldout.jsonl"
    heldout.write_text('{"prompt":"same"}\n')
    with pytest.raises(ValueError,match="held-out"):
        export_unique_prompts(source,tmp_path/"output",heldout=heldout)
    assert not (tmp_path/"output").exists()
