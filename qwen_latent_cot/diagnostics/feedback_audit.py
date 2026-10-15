"""Read-only audit of single-node or full-restart explicit-feedback results."""
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
        r = json.loads(path.read_text())
        all_images = {arm:Image.open(path.parent/file).convert('RGB') for arm,file in r['images'].items()}
        restart = r.get('protocol') == 'explicit_feedback_restart_v2'
        windows = r['early'] if restart else [None]
        for window in windows:
            if restart:
                prefix = f'early{window}_'
                images = {'Base':all_images['Base'], **{arm[len(prefix):]:image for arm,image in all_images.items() if arm.startswith(prefix)}}
                times = r['sampling_times']; t = times[0]
                intervention_steps = 0 if r['bypass'] else window
                native_steps = len(times)-1-intervention_steps
                time_fraction = 0 if r['bypass'] else (times[0]-times[window])/times[0]
            else:
                images = all_images; times = r['continuation_times']; t = r['t']
                intervention_steps = 0 if r['bypass'] else 1
                native_steps = len(times)-1-intervention_steps
                time_fraction = 0 if r['bypass'] else (times[0]-times[1])/times[0]
            paired = image_diagnostics(images)
            base = {arm:image_diagnostics({'Base':images['Base'], 'Candidate':image})['Candidate']
                    for arm,image in images.items()}
            preview = Image.open(path.parent/r['preview']).convert('RGB')
            records.append(dict(record=str(path), sample_id=r['sample_id'], worker=path.parent.name,
                protocol=r.get('protocol','explicit_feedback_v1'), early=window,
                prompt=r['prompt'], feedback=r['feedback'], t=t, bypass=r['bypass'],
                dt=times[0]-times[1], intervention_steps=intervention_steps, native_suffix_steps=native_steps,
                feedback_time_fraction=time_fraction, paired=paired, versus_base=base,
                preview_vs_base=image_diagnostics({'Base':images['Base'], 'Preview':preview})['Preview'],
                response_available='response_diagnostics' in r, response_diagnostics=r.get('response_diagnostics'),
                trajectory_records_available=bool(restart and r.get('arms'))))
    if not records:
        raise ValueError('no explicit feedback comparison records found')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'image_audit.json', records)
    def avg(arm, key):
        values = [r[key][arm]['mae_255'] for r in records if arm in r[key]]
        return sum(values)/len(values) if values else None
    summary = dict(samples=len({r['record'] for r in records}), window_comparisons=len(records),
        bypass_count=sum(r['bypass'] for r in records),
        text_mae_vs_base=avg('Text', 'versus_base'), memory_mae_vs_base=avg('Text-memory', 'versus_base'),
        text_mae_vs_layout=avg('Text','paired'), memory_mae_vs_layout=avg('Text-memory','paired'),
        velocity_response_records=sum(r['response_available'] for r in records),
        semantic_quality='not scored by pixel differences; requires human final-image annotation')
    write_json(output/'summary.json', summary)
    lines = ['# 显式反馈结果审计', '', f"读取 {summary['samples']} 个反馈结果，汇总 {len(records)} 个窗口比较。未修改原结果。", '',
        '像素差异不能证明对象、数量或关系得到修正。先对照原 prompt 核对反馈，再审查终图纠错与保持。', '',
        '|样本|early 窗口|反馈跳过|Text vs Base MAE|Memory vs Base MAE|Text vs Layout MAE|Memory vs Layout MAE|注入步数|原生步数|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        def val(group,arm):
            return f"{r[group][arm]['mae_255']:.4f}" if arm in r[group] else '—'
        lines.append(f"|{r['worker']} / {r['sample_id']}|{r['early'] if r['early'] is not None else '单节点'}|{r['bypass']}|{val('versus_base','Text')}|{val('versus_base','Text-memory')}|{val('paired','Text')}|{val('paired','Text-memory')}|{r['intervention_steps']}|{r['native_suffix_steps']}|")
    lines += ['', 'MAE 是 RGB 通道绝对差的均值，范围为 0–255。条件臂与对应窗口的 Layout 比较，避免把位置变化当作内容作用。', '',
        '自动反馈的数量、关系和对象保持均须单独核对。非零像素差和速度响应不能替代终图收益。', '',
        '重生成协议的 JSON 保存逐步速度哈希、范数与条件注入记录；单节点协议按其独立响应字段审计。缺少响应数据时不能推断条件未消费。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
