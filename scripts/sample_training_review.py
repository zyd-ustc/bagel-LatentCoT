"""Sample one image/text pair per discovered category using bounded index reads."""
import argparse
import base64
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
import html
import io
import json
import os
from pathlib import Path
import random
import re
import secrets
import struct
import time

from PIL import Image, ImageOps


REFERENCE = struct.Struct('<IQB')
ROLES = {
    0: ('目标图描述', 'target_caption'), 1: ('输入图描述', 'input_caption'),
    2: ('生成目标文字', 'target_caption'), 3: ('当前图与修正指令', 'edit_instruction'),
    4: ('保持指令', None), 5: ('轨迹反馈', None),
    6: ('条件生成指令', 'edit_instruction'), 7: ('状态描述', 'observed_caption'),
}


def relocate(path, annotation, roots):
    if not path:
        return None
    for old, new in sorted(roots.items(), key=lambda item: -len(item[0])):
        if path.startswith(old):
            path = new + path[len(old):]
            break
    return os.path.abspath(os.path.join(os.path.dirname(annotation), path))


def categories(row, image):
    """Use annotation types and named dataset subsets, rather than image filenames."""
    types = row.get('edit_type') or ['other']
    if isinstance(types, str):
        types = [types]
    types = [kind for kind in types if kind != 'other']
    if types:
        return sorted(set(types))
    parts = Path(image or '').parts
    subsets = [part for part in parts if 'Subset' in part]
    if subsets:
        return [subsets[0]]
    if 'lowlevel' in parts:
        tail = parts[parts.index('lowlevel')+1:]
        return ['/'.join(tail[:2])] if len(tail) >= 3 else [row['condition_kind']]
    if 'CapRL-2M' in parts:
        tail = parts[max(i for i, part in enumerate(parts) if part == 'CapRL-2M')+1:]
        return [tail[0]] if len(tail) >= 2 else ['default']
    if 'X2Edit-Dataset' in parts and 'tar' in parts:
        return [parts[parts.index('tar')+1]]
    if 'ImgEdit' in parts:
        task = next((p for p in parts if p.startswith('results_')), None)
        if task:
            return [re.sub(r'_part\d+$', '', task.removeprefix('results_'))]
    seed = next((p for p in parts if p.startswith('SEED-Data-Edit-Part')), None)
    if seed:
        return [seed]
    return [row.get('condition_kind', 'none') if row.get('condition_kind', 'none') != 'none' else 'default']


class IndexReader:
    def __init__(self, root, manifest):
        self.root, self.manifest = root, manifest
        self.shards = OrderedDict()
        self.reads = 0

    def read(self, index, position):
        index.seek(position * REFERENCE.size)
        shard, offset, role = REFERENCE.unpack(index.read(REFERENCE.size))
        if shard not in self.shards:
            self.shards[shard] = (self.root/'records'/self.manifest['shards'][shard]['path']).open('rb')
            if len(self.shards) > 8:
                self.shards.popitem(last=False)[1].close()
        stream = self.shards[shard]
        self.shards.move_to_end(shard)
        stream.seek(offset)
        self.reads += 1
        return json.loads(stream.readline()), role, shard, offset

    def close(self):
        for stream in self.shards.values():
            stream.close()


def sample(root, manifest, seed, probes):
    grouped = defaultdict(list)
    for view in manifest['views']:
        if view['count']:
            grouped[view['source']].append(view)
    rng = random.Random(seed)
    reader = IndexReader(root, manifest)
    chosen, seen = {}, Counter()
    roots = manifest['registry'].get('roots', {})
    try:
        for source, views in sorted(grouped.items()):
            tasks = {v['task'] for v in views}
            if source.startswith('phase1_structured') and 'endpoint' in tasks:
                task = 'endpoint'
            elif 'generation' in tasks and 'conditional' not in tasks:
                task = 'generation'
            else:
                task = next(t for t in ('observe', 'conditional', 'endpoint', 'trajectory', 'generation') if t in tasks)
            candidates = [v for v in views if v['task'] == task]
            split = next(s for s in ('train', 'val', 'test') if any(v['split'] == s for v in candidates))
            view = next(v for v in candidates if v['split'] == split)
            count = view['count']; n = min(probes, count)
            positions = {int((i+rng.random())*count/n) for i in range(n)}
            # Observe input/target roles are adjacent in editing indexes.
            if task == 'observe':
                positions |= {p+1 for p in positions if p+1 < count}
            with (root/'views'/view['path']).open('rb') as index:
                for position in sorted(positions):
                    row, role, shard, offset = reader.read(index, position)
                    image = row['input_images'][0] if role in (1, 3) else row['target_image']
                    image = relocate(image, row['original']['path'], roots)
                    label, field = ROLES[role]
                    if role == 4:
                        text = 'No change is needed.'
                    elif role == 5:
                        feedback = row['feedback']
                        text = 'No change is needed.' if feedback['satisfied'] else feedback['correction']
                    else:
                        text = row.get(field, '') or row.get('goal_prompt', '')
                    for category in categories(row, image):
                        key = (source, role, category)
                        seen[key] += 1
                        if rng.randrange(seen[key]):
                            continue
                        chosen[key] = dict(source=source, category=category, role=role, label=label, text=text,
                            image=image, prompt=row['goal_prompt'], instruction=row['edit_instruction'],
                            input_caption=row['input_caption'], target_caption=row['target_caption'],
                            split=split, record_id=row['record_id'], original_id=row['original']['id'],
                            annotation_path=row['original']['path'], index=view['path'], position=position,
                            shard=shard, offset=offset, tokens=row.get('tokens', {}))
        return [chosen[key] for key in sorted(chosen)], reader.reads
    finally:
        reader.close()


def render(rows, manifest, seed, probes, reads, edge):
    esc = lambda value: html.escape(str(value))
    cards = []; cache = {}; errors = 0
    for number, row in enumerate(rows, 1):
        path = row['image']
        if path not in cache:
            try:
                with Image.open(path) as image:
                    size = image.size
                    image.draft('RGB', (edge, edge))
                    image = ImageOps.exif_transpose(image)
                    image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                    stream = io.BytesIO()
                    image.convert('RGB').save(stream, format='JPEG', quality=86)
                cache[path] = (base64.b64encode(stream.getvalue()).decode(), size, None)
            except (OSError, ValueError, TypeError) as error:
                cache[path] = (None, None, f'{type(error).__name__}: {error}')
        data, size, error = cache[path]
        row['original_size'] = size; row['read_error'] = error
        errors += int(error is not None)
        visual = f'<img loading="lazy" src="data:image/jpeg;base64,{data}" alt="样本 {number}">' if data else f'<div class="error">{esc(error)}</div>'
        metadata = {k: row[k] for k in ('record_id', 'original_id', 'image', 'annotation_path', 'index', 'position', 'tokens')}
        captions = ''.join(f'<dt>{title}</dt><dd>{esc(row[key])}</dd>' for key, title in
                           [('prompt', '目标 prompt'), ('instruction', '编辑指令'), ('input_caption', '输入图描述'), ('target_caption', '目标图描述')]
                           if row[key] and row[key] != row['text'])
        cards.append(f'''<article data-search="{esc(row['source']+' '+row['category']+' '+row['label'])}">
          <div class="heading"><span class="number">{number:03}</span><div><h2>{esc(row['source'])}</h2>
          <p>{esc(row['category'])} · {esc(row['label'])} · {esc(row['split'])}</p></div></div>
          <div class="pair"><div class="visual">{visual}<small>{esc(size or '读取失败')}</small></div>
          <div class="caption"><div class="tag">{esc(row['label'])} · {len(row['text'])} 字符</div><pre>{esc(row['text'])}</pre>
          <details><summary>上下文与原始来源</summary><dl>{captions}</dl><pre>{esc(json.dumps(metadata,ensure_ascii=False,indent=2))}</pre></details></div></div></article>''')
    disabled = [s for s in manifest['registry']['sources'] if s.get('enabled') is False]
    excluded = ''.join(f'<li>{esc(s["name"])}：{esc(s.get("disabled_reason", "已停用"))}</li>' for s in disabled)
    sources = len({r['source'] for r in rows})
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>图文数据审阅 · seed {seed}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f3ef;color:#242820;font:15px/1.65 system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:36px 28px}}h1{{font-size:32px;margin:0 0 10px}}header p{{max-width:1050px;color:#62685b}}.stats{{display:flex;gap:22px;flex-wrap:wrap;padding:16px 0}}.stats b{{font-size:23px}}input{{padding:12px 16px;border:1px solid #c5c9bd;border-radius:8px;width:100%;margin:12px 0 22px;background:white}}article{{background:white;border:1px solid #dce0d5;border-radius:12px;overflow:hidden;margin-bottom:26px}}.heading{{display:flex;gap:15px;padding:18px 22px;border-bottom:1px solid #eceee7}}h2{{font-size:17px;margin:0;overflow-wrap:anywhere}}.heading p{{margin:2px 0;color:#68715e}}.number{{color:#7d885e;font-size:22px}}.pair{{display:grid;grid-template-columns:minmax(320px,1fr) minmax(300px,1fr)}}.visual{{background:#eaece6;padding:18px;text-align:center;align-self:start}}img{{display:block;width:100%;max-height:600px;object-fit:contain;margin:auto;cursor:zoom-in}}small{{color:#68715e}}.caption{{padding:22px;min-width:0}}.tag{{font-size:12px;color:#627745;font-weight:600}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.75 system-ui,sans-serif;max-height:520px;overflow:auto}}details{{margin-top:18px;border-top:1px solid #eee;padding-top:10px}}summary{{cursor:pointer;color:#536640}}dt{{font-size:12px;color:#68715e;margin-top:10px}}dd{{margin:0;white-space:pre-wrap}}.error{{color:#9d362a;background:#fff0ed;padding:25px;overflow-wrap:anywhere}}footer{{color:#68715e;font-size:13px}}dialog{{border:0;padding:12px;max-width:96vw;max-height:96vh;background:#f4f3ef}}dialog img{{width:auto;max-width:92vw;max-height:90vh}}@media(max-width:760px){{main{{padding:20px 12px}}.pair{{grid-template-columns:1fr}}h1{{font-size:26px}}}}
</style><main><header><h1>图文数据审阅</h1><p>每个来源优先抽取图像描述；编辑前后图分开覆盖。补充结构化编辑类型、命名路径子集和条件生成类别，每类只保留一对。条件生成与反馈样本明确标为指令。</p>
<div class="stats"><span><b>{len(rows)}</b> 对</span><span><b>{sources}</b> 个来源</span><span><b>{errors}</b> 个读取错误</span><span>seed <b>{seed}</b></span></div>
<p>每来源最多 {probes} 个分层随机索引探针，Observe 增加相邻角色。本轮共读 {reads} 条标注，只读取最终选中的 {len(cache)} 张图片；无模型推理。图像缩至最长 {edge}px，文字保留全文。点击图片放大，搜索来源、类别或角色。</p>
<input id="search" placeholder="筛选：TextAtlas / 输入图 / rain / replacement …"><p id="visible"></p></header>
{''.join(cards)}<footer><details><summary>已停用来源（{len(disabled)}）</summary><ul>{excluded}</ul></details><p>Corpus fingerprint：{esc(manifest['fingerprint'])} · 当前页保存本轮样本，不会在刷新时重新采样。再次运行脚本会自动更换 seed；指定 --seed 可复现。</p></footer></main>
<dialog id="zoom"><img alt="放大样本"></dialog><script>
const cards=[...document.querySelectorAll('article')],search=document.querySelector('#search');
function filter(){{let n=0;for(const c of cards){{c.hidden=!c.dataset.search.toLowerCase().includes(search.value.toLowerCase());n+=!c.hidden;}}document.querySelector('#visible').textContent=`显示 ${{n}} / ${{cards.length}} 对`;}}search.addEventListener('input',filter);filter();
const zoom=document.querySelector('#zoom');for(const img of document.querySelectorAll('article img'))img.addEventListener('click',()=>{{zoom.querySelector('img').src=img.src;zoom.showModal();}});zoom.addEventListener('click',()=>zoom.close());
</script></html>''', errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='/data/zyd_workspace/bagel-LatentCoT/datasets/observe_feedback_corpus')
    parser.add_argument('--seed', type=int, help='default: fresh random seed on every invocation')
    parser.add_argument('--probes', type=int, default=32)
    parser.add_argument('--image-edge', type=int, default=1024)
    parser.add_argument('--output', help='HTML path; defaults to a fresh file in /root/outputs')
    args = parser.parse_args(); start = time.monotonic()
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    output = Path(args.output) if args.output else Path('/root/outputs')/f'dataset_review_{datetime.now():%Y%m%d_%H%M%S}_{seed}.html'
    root = Path(args.corpus); manifest = json.loads((root/'manifest.json').read_text())
    rows, reads = sample(root, manifest, seed, args.probes)
    page, errors = render(rows, manifest, seed, args.probes, reads, args.image_edge)
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(page)
    report = dict(seed=seed, corpus=str(root), fingerprint=manifest['fingerprint'], probes=args.probes,
                  annotation_reads=reads, sources=len({r['source'] for r in rows}), pairs=len(rows), errors=errors,
                  seconds=round(time.monotonic()-start,2), html=str(output), samples=rows)
    output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
