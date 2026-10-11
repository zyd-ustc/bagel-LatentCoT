"""Four protocol tables. Blind labels are optional and never fabricated."""
import csv
import json
from pathlib import Path
from .data import cluster_interval
from .record import write_json


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize(runs, output):
    text_groups = {}; visual_groups = {}; gen_groups = {}; final_groups = {}
    raw = {'text_bridge': [], 'visual_caption': [], 'gen_local': [], 'final_image': []}
    contracts = []
    for name in runs:
        run = Path(name); manifest = json.loads((run/'manifest.json').read_text())
        contracts.append({k: manifest[k] for k in ('model_files', 'dataset_hash', 'depth', 'slots', 'config', 'source_hashes')})
        for row in read_rows(run/'captions.jsonl'):
            key = (row['topology'], row['stage'], row['arm'], row.get('t'), row['label_type'], row['state_source'], row['initialization_only'])
            target = text_groups if row['stage'] in ('C0', 'C1') else visual_groups
            target.setdefault(key, []).append(row)
        for row in read_rows(run/'gen.jsonl'):
            key = (row['topology'], row['arm'], row['t'], row['state_source'])
            gen_groups.setdefault(key, []).append(row)
        key_file = run/'blind_key.json'; annotation = run/'blind_annotations.csv'
        if key_file.exists() and annotation.exists():
            mapping = {r['blind_file']: r for r in json.loads(key_file.read_text())}
            scored = {}
            with annotation.open() as stream:
                for item in csv.DictReader(stream):
                    if not item['target_facts'].strip() or not item['quality'].strip():
                        continue
                    meta = mapping[item['blind_file']]
                    values = {k: float(item[k]) for k in ('target_facts', 'quality', 'unchanged_facts', 'artifacts') if item.get(k, '').strip()}
                    if not 0 <= values['target_facts'] <= 1 or not 1 <= values['quality'] <= 5:
                        raise ValueError('blind target_facts must be [0,1]; quality must be [1,5]')
                    scored[(meta['sample_id'], meta['t'], meta['seed'], meta['arm'])] = dict(meta, **values)
            for key, item in scored.items():
                if item['arm'] in ('G-base', 'G-layout'):
                    continue
                for baseline in ('G-base', 'G-layout'):
                    reference = scored.get((*key[:3], baseline))
                    if reference is None:
                        continue
                    fact_delta = item['target_facts']-reference['target_facts']
                    quality_delta = item['quality']-reference['quality']
                    entry = dict(scene_id=item['scene_id'], family_id=item.get('family_id', item['scene_id']),
                                 fact_delta=fact_delta, quality_delta=quality_delta,
                                 win=int(fact_delta > 0), tie=int(fact_delta == 0), loss=int(fact_delta < 0))
                    final_groups.setdefault((manifest['topology'], item['arm'], baseline, item['t']), []).append(entry)
    # Do not silently combine runs from different native models or frozen splits.
    if contracts and any(c != contracts[0] for c in contracts):
        raise ValueError('report runs have different model/data/depth/capacity contracts; summarize separately')
    for destination, groups in (('text_bridge', text_groups), ('visual_caption', visual_groups)):
        for key, rows in groups.items():
            topology, stage, arm, t, label, source, init = key
            entry = dict(topology=topology, stage=stage, arm=arm, t=t, label_type=label, source=source, init=init,
                         nll=sum(r['nll'] for r in rows)/len(rows), fact_accuracy=cluster_interval(rows, 'fact_accuracy'),
                         unknown_assertion_rate=cluster_interval(rows, 'unknown_assertion_rate'))
            entry['by_category'] = {}
            for category in ('object', 'attribute', 'count', 'relation'):
                category_rows = []
                for row in rows:
                    facts = [f for f in row['facts'] if f['category'] == category and not f['unknown']]
                    if facts:
                        category_rows.append(dict(row, score=sum(f['correct'] for f in facts)/len(facts)))
                entry['by_category'][category] = cluster_interval(category_rows, 'score')
            raw[destination].append(entry)
    for key, rows in gen_groups.items():
        raw['gen_local'].append(dict(topology=key[0], arm=key[1], t=key[2], source=key[3],
            conditional_difference=sum(r['conditional']['normalized_difference'] for r in rows)/len(rows),
            cfg_difference=sum(r['cfg']['normalized_difference'] for r in rows)/len(rows),
            cfg_cosine=sum(r['cfg']['cosine'] for r in rows)/len(rows),
            cfg_norm_ratio=sum(r['cfg']['norm_ratio'] for r in rows)/len(rows),
            repeat_cfg_error=max(r['repeat_error']['cfg']['normalized_difference'] for r in rows),
            layout_native_shift_parity=max(r['layout_native_shift_parity']['cfg']['normalized_difference'] for r in rows),
            semantics='pending independent factual review'))
    for key, rows in final_groups.items():
        raw['final_image'].append(dict(topology=key[0], arm=key[1], reference=key[2], t=key[3],
            fact_delta=cluster_interval(rows, 'fact_delta'), quality_delta=cluster_interval(rows, 'quality_delta'),
            wins=sum(r['win'] for r in rows), ties=sum(r['tie'] for r in rows), losses=sum(r['loss'] for r in rows)))
    output.mkdir(parents=True, exist_ok=True); write_json(output/'tables.json', raw)
    def fmt(value):
        if value is None: return '未标注/无数据'
        if isinstance(value, dict) and 'mean' in value:
            return f'{value["mean"]:.4f} [{value["low"]:.4f}, {value["high"]:.4f}]'
        if isinstance(value, float): return f'{value:.4f}'
        return str(value).replace('|', '/')
    lines = ['# Memory 拓扑诊断报告', '', '事实分数使用预先固定的短语规则；数值变化不代表语义有效或画质改善。',
             '区间按场景族聚类重采样；t=1单独列出。未完成盲评的最终图像结果保留为空。', '']
    tables = [('文本桥', 'text_bridge', ['topology','stage','arm','init','nll','fact_accuracy']),
              ('视觉 Caption', 'visual_caption', ['topology','arm','t','label_type','source','nll','fact_accuracy','unknown_assertion_rate']),
              ('GEN 局部敏感性', 'gen_local', ['topology','arm','t','source','conditional_difference','cfg_difference','cfg_norm_ratio','repeat_cfg_error']),
              ('最终图像事实与画质', 'final_image', ['topology','arm','reference','t','fact_delta','quality_delta','wins','ties','losses'])]
    for title, key, columns in tables:
        lines += [f'## {title}', '', '| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
        for item in raw[key]: lines.append('| '+' | '.join(fmt(item.get(c)) for c in columns)+' |')
        if not raw[key]: lines.append('| '+' | '.join(['待评测']*len(columns))+' |')
        lines.append('')
    (output/'report.md').write_text('\n'.join(lines))
