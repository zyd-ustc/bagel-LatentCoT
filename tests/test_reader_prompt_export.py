import json
import pytest
from scripts.data.prepare_reader_warmup_prompts import label_prompt,prepare


def test_explicit_tags_do_not_invent_labels():
    assert label_prompt("Two red cubes on a table")[0]=="count"
    assert label_prompt("A blue cup to the left of a red bowl")[0]=="spatial_relation"
    assert label_prompt("A beautiful landscape") is None
    assert label_prompt("Two weeks of sunshine") is None


def test_export_preserves_official_heldout_and_removes_train_overlap(tmp_path):
    train=[dict(sample_id=str(i),prompt=f"Two red cubes beside {i} trees") for i in range(9)]
    val=[dict(sample_id="v0",prompt=train[0]["prompt"]),
         dict(sample_id="v1",prompt="A cat behind a dog")]
    source=tmp_path/"train.jsonl"; heldout=tmp_path/"val.jsonl"
    for path,rows in ((source,train),(heldout,val)):
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    output=tmp_path/"export"
    report=prepare(source,heldout,output,heldout_count=2)
    assert report["normalized_source_prompt_overlaps"]==1
    assert report["train_records"]==8 and report["heldout_records"]==2
    assert not report["images_consumed"]
    rows=[json.loads(line) for line in (output/"reader_train.jsonl").read_text().splitlines()]
    assert train[0]["prompt"] not in {row["prompt"] for row in rows}
    with pytest.raises(FileExistsError):
        prepare(source,heldout,output,heldout_count=2)
