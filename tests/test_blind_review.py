import csv
import json
from PIL import Image
import pytest
from qwen_latent_cot.evaluation.blind_review import make_blind_pack,import_blind_review
from qwen_latent_cot.evaluation.io import sha256


def test_blind_pack_hides_arms_and_preserves_hashes(tmp_path):
    rows=[]
    for arm,color in [('BASE','red'),('MEMORY_DYNAMIC','blue')]:
        path=tmp_path/f'{arm}.png';Image.new('RGB',(16,16),color).save(path)
        rows.append({'arm':arm,'prompt_id':'p0','seed':0,'path':str(path),'prompt':'a cube',
            'semantic_questions':[['Is there a cube?','Yes']]})
    out=tmp_path/'blind';make_blind_pack(rows,out)
    table=list(csv.DictReader((out/'review.csv').open()))
    assert 'MEMORY_DYNAMIC' not in (out/'review.csv').read_text()
    assert (out/'semantic_review.json').exists()
    table[0].update(preference_A_B_tie='A',A_invalid='false',B_invalid='false')
    with (out/'review.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=table[0].keys());writer.writeheader();writer.writerows(table)
    result=import_blind_review(out/'review.csv',tmp_path/'blind_review_private_key.json')
    assert result['complete'] and result['reviewed_pairs']==1
    Image.new('RGB',(16,16),'white').save(out/table[0]['image_A'])
    with pytest.raises(ValueError,match='changed'):import_blind_review(out/'review.csv',tmp_path/'blind_review_private_key.json')
