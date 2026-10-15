"""Audit fixed-memory regeneration by early window and layout control."""
import argparse
import json
from pathlib import Path
from PIL import Image
from .explicit_feedback import image_diagnostics
from .record import write_json


def audit(run, output):
    run = Path(run).resolve(); output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('audit output must be empty')
    records = []
    paths = sorted(run.rglob('*_bagel_auto.json'))+sorted(run.rglob('*_manual_checked.json'))
    for path in paths:
        result = json.loads(path.read_text())
        all_images = {arm:Image.open(path.parent/file).convert('RGB') for arm,file in result['images'].items()}
        times = result['sampling_times']
        for window in result['early']:
            prefix = f'early{window}_'
            images = {'Base':all_images['Base'], **{
                arm[len(prefix):]:image for arm,image in all_images.items() if arm.startswith(prefix)}}
            intervention_steps = 0 if result['bypass'] else window
            paired = image_diagnostics(images)
            base = {arm:image_diagnostics({'Base':images['Base'], 'Candidate':image})['Candidate']
                    for arm,image in images.items()}
            arms = {arm:trajectory for arm,trajectory in result['arms'].items() if arm.startswith(prefix)}
            records.append(dict(record=str(path), sample_id=result['sample_id'], worker=path.parent.name,
                early=window, prompt=result['prompt'], feedback=result['feedback'], bypass=result['bypass'],
                initial_noise_hash=result['initial_noise_hash'],
                intervention_steps=intervention_steps, native_suffix_steps=len(times)-1-intervention_steps,
                feedback_time_fraction=0 if result['bypass'] else (times[0]-times[window])/times[0],
                paired=paired, versus_base=base, trajectory_records_available=bool(arms),
                generation_arms=arms))
    if not records:
        raise ValueError('no explicit feedback comparison records found')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'image_audit.json', records)
    def average(arm, key):
        values = [r[key][arm]['mae_255'] for r in records if arm in r[key]]
        return sum(values)/len(values) if values else None
    summary = dict(samples=len({r['record'] for r in records}), window_comparisons=len(records),
        bypass_count=sum(r['bypass'] for r in records),
        text_mae_vs_base=average('Text', 'versus_base'), memory_mae_vs_base=average('Text-memory', 'versus_base'),
        text_mae_vs_layout=average('Text','paired'), memory_mae_vs_layout=average('Text-memory','paired'),
        trajectory_window_records=sum(r['trajectory_records_available'] for r in records))
    write_json(output/'summary.json', summary)
    lines = ['# 显式反馈结果审计', '',
        f"读取 {summary['samples']} 个反馈结果，汇总 {len(records)} 个窗口比较。", '',
        '|样本|early 窗口|反馈跳过|Text vs Base MAE|Memory vs Base MAE|Text vs Layout MAE|Memory vs Layout MAE|注入步数|原生步数|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for record in records:
        def value(group,arm):
            return f"{record[group][arm]['mae_255']:.4f}" if arm in record[group] else '—'
        lines.append(f"|{record['worker']} / {record['sample_id']}|{record['early']}|{record['bypass']}|"
            f"{value('versus_base','Text')}|{value('versus_base','Text-memory')}|{value('paired','Text')}|"
            f"{value('paired','Text-memory')}|{record['intervention_steps']}|{record['native_suffix_steps']}|")
    lines += ['', 'MAE 是 RGB 通道绝对差的均值，范围为 0–255。条件内容使用对应窗口的 Layout 作为参照。', '',
        '逐步记录包含条件速度与 CFG 速度的哈希、范数和条件注入次数。终图评价记录目标事实修正、正确事实保持与质量。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
