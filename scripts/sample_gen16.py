from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
import argparse,json,random,re,time,textwrap
parser=argparse.ArgumentParser(description="Stream CoRT manifest and sample 16 candidate images, without decoding the image library")
parser.add_argument("--dataset-root", default="/data/zyd_workspace/bagel-LatentCoT/datasets")
parser.add_argument("--output", required=True)
args=parser.parse_args()
root=Path(args.dataset_root)
source=root/'qwen_latent_cot_v3_adapted_20260816/manifests/cort36k_test.jsonl'
rng=random.Random(20261012); pools={'red':[],'blue':[]}; counts={'red':0,'blue':0}; total=0
with source.open() as stream:
    for line in stream:
        total+=1; row=json.loads(line); prompt=row.get('prompt','')
        colors=set(re.findall(r'\b(?:red|blue)\b',prompt.lower()))
        if len(prompt.split())>25 or len(colors)!=1 or row.get('chain_status')!='terminal' or not row.get('image_paths'):continue
        color=next(iter(colors)); counts[color]+=1
        entry=dict(row,color=color,source_line=total)
        if len(pools[color])<32:pools[color].append(entry)
        else:
            index=rng.randrange(counts[color])
            if index<32:pools[color][index]=entry
out=Path(args.output);out.mkdir();(out/'thumbnails').mkdir()
selected=[]
for color in ('red','blue'):
    rng.shuffle(pools[color]); kept=0
    for row in pools[color]:
        image=Path(row['image_paths'][-1])
        if not image.is_file():
            prefix='/data/bagel-LatentCoT/datasets/'
            image=root/str(image).removeprefix(prefix)
        if not image.is_file():continue
        index=len(selected);sid=row['sample_id']
        with Image.open(image) as original:
            thumb=original.convert('RGB');thumb.thumbnail((512,512));thumb.save(out/'thumbnails'/f'{index:02d}.jpg')
        selected.append(dict(index=index,sample_id=sid,image=str(image),source_prompt=row['prompt'],source_manifest=str(source),source_line=row['source_line'],generator_model=row.get('generator_model'),candidate_color=color,verified=False))
        kept+=1
        if kept==8:break
    if kept!=8:raise RuntimeError(f'Only {kept} accessible images for {color}')
(out/'sources.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected))
sheet=Image.new('RGB',(1200,1440),'white');draw=ImageDraw.Draw(sheet)
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
except OSError:font=ImageFont.load_default()
for row in selected:
    x=(row['index']%4)*300;y=(row['index']//4)*360
    with Image.open(out/'thumbnails'/f'{row["index"]:02d}.jpg') as im:tile=ImageOps.contain(im,(292,280))
    sheet.paste(tile,(x+(300-tile.width)//2,y))
    draw.text((x+5,y+283),f'{row["index"]:02d} {row["sample_id"]}',fill='black',font=font)
    draw.text((x+5,y+305),'\n'.join(textwrap.wrap(row['source_prompt'],34)[:3]),fill='black',font=font)
sheet.save(out/'sources16.jpg')
(out/'selection.json').write_text(json.dumps(dict(seed=20261012,manifest=str(source),rows_scanned=total,eligible_counts=counts,selected=16,policy='reservoir-balanced-red-blue-short-prompts-terminal-last-image; semantic review pending'),indent=2))
print(out)
for row in selected:print(row['index'],row['sample_id'],row['source_prompt'])
