"""Aggregation for the explicit closed-loop pilot: detection and retry gains."""
from math import comb

from .atoms import parse_question


def mcnemar_exact(loop_only, resample_only):
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    n = loop_only + resample_only
    if n == 0:
        return 1.0
    k = min(loop_only, resample_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _preview(entry, index):
    return next(p for p in entry['previews'] if p['index'] == index)


def _retry(entry, index):
    return next((r for r in entry['retries'] if r['index'] == index), None)


def base_stats(rows):
    total = len(rows)
    passed = sum(1 for r in rows if r['base']['passed'])
    by_skill = {}
    for row in rows:
        for atom in row['base']['atoms']:
            skill = parse_question(atom['question'])['kind']
            slot = by_skill.setdefault(skill, dict(atoms=0, correct=0))
            slot['atoms'] += 1
            slot['correct'] += int(atom['correct'])
    for slot in by_skill.values():
        slot['rate'] = slot['correct'] / slot['atoms'] if slot['atoms'] else None
    return dict(images=total, passed=passed,
                pass_rate=passed / total if total else None, by_skill=by_skill)


def detection_stats(rows, index):
    tp = fp = fn = tn = 0
    for row in rows:
        preview_fail = _preview(row, index)['detected_failure']
        base_fail = not row['base']['passed']
        if preview_fail and base_fail:
            tp += 1
        elif preview_fail and not base_fail:
            fp += 1
        elif base_fail:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                preview_fail_rate=(tp + fp) / len(rows) if rows else None,
                precision=precision, recall=recall, f1=f1,
                missed_failure_rate=(fn / (fn + tp)) if (fn + tp) else None)


def retry_stats(rows, index):
    """Compute-matched loop vs resample on the detected final-failure subset."""
    subset = []
    for row in rows:
        retry = _retry(row, index)
        if retry is None:
            continue
        if row['base']['passed']:
            continue
        subset.append((row, retry))
    loop_pass = sum(1 for _, r in subset if r['loop']['passed'])
    resample_pass = sum(1 for _, r in subset if r['resample']['passed'])
    loop_only = sum(1 for _, r in subset if r['loop']['passed'] and not r['resample']['passed'])
    resample_only = sum(1 for _, r in subset if r['resample']['passed'] and not r['loop']['passed'])
    both = sum(1 for _, r in subset if r['loop']['passed'] and r['resample']['passed'])
    neither = len(subset) - loop_only - resample_only - both
    full = [r['full_resample']['passed'] for _, r in subset if 'full_resample' in r]
    return dict(images=len(subset), loop_success=loop_pass, resample_success=resample_pass,
                full_success=sum(full) if full else None,
                full_rate=(sum(full) / len(full)) if full else None,
                loop_rate=loop_pass / len(subset) if subset else None,
                resample_rate=resample_pass / len(subset) if subset else None,
                both=both, loop_only=loop_only, resample_only=resample_only, neither=neither,
                mcnemar_p=mcnemar_exact(loop_only, resample_only))


def atom_fix_stats(rows, index):
    """Per-skill fate of the atoms that failed in the final base image."""
    table = {}
    for row, retry in ((r, _retry(r, index)) for r in rows):
        if retry is None or row['base']['passed']:
            continue
        loop_atoms = {a['question']: a for a in retry['loop']['atoms']}
        resample_atoms = {a['question']: a for a in retry['resample']['atoms']}
        for atom in row['base']['atoms']:
            if atom['correct']:
                continue
            skill = parse_question(atom['question'])['kind']
            slot = table.setdefault(skill, dict(atoms=0, loop_fixed=0, resample_fixed=0,
                                                full_fixed=0, full_atoms=0))
            slot['atoms'] += 1
            slot['loop_fixed'] += int(loop_atoms.get(atom['question'], {}).get('correct', False))
            slot['resample_fixed'] += int(resample_atoms.get(atom['question'], {}).get('correct', False))
            if 'full_resample' in retry:
                full_atoms = {a['question']: a for a in retry['full_resample']['atoms']}
                slot['full_atoms'] += 1
                slot['full_fixed'] += int(full_atoms.get(atom['question'], {}).get('correct', False))
    for slot in table.values():
        slot['loop_rate'] = slot['loop_fixed'] / slot['atoms'] if slot['atoms'] else None
        slot['resample_rate'] = slot['resample_fixed'] / slot['atoms'] if slot['atoms'] else None
        slot['full_rate'] = slot['full_fixed'] / slot['full_atoms'] if slot['full_atoms'] else None
    return table


def evaluate(rows, read_indexes):
    read_indexes = [int(i) for i in read_indexes]
    report = dict(base=base_stats(rows), read_points=[])
    for index in read_indexes:
        preview = _preview(rows[0], index) if rows else None
        report['read_points'].append(dict(
            index=index, t=preview['t'] if preview else None,
            detection=detection_stats(rows, index),
            retry=retry_stats(rows, index),
            atoms=atom_fix_stats(rows, index)))
    return report


def summary_lines(report):
    out = []
    base = report['base']
    out.append(f"base image pass {base['passed']}/{base['images']} = {base['pass_rate']:.3f}")
    for point in report['read_points']:
        d, r = point['detection'], point['retry']
        line = (f"t_read={point['t']:.3f} (idx {point['index']}): "
                f"detect TP/FP/FN={d['tp']}/{d['fp']}/{d['fn']} "
                f"prec={d['precision']} rec={d['recall']}")
        if r['images']:
            line += (f" | retry n={r['images']} loop={r['loop_rate']:.3f} "
                     f"resample={r['resample_rate']:.3f} p={r['mcnemar_p']:.3f}")
        out.append(line)
    return out
