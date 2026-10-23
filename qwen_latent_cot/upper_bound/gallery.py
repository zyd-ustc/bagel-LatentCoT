"""Static HTML gallery for manual review of the closed-loop pilot."""
import html
import json
from pathlib import Path


def _badge(passed):
    color = '#2f7d4f' if passed else '#b64342'
    text = 'PASS' if passed else 'FAIL'
    return f'<span style="color:#fff;background:{color};padding:1px 6px;border-radius:4px">{text}</span>'


def _atoms(atoms, label):
    rows = ''.join(
        f'<tr><td>{html.escape(a["question"])}</td><td>{html.escape(a["expected"])}</td>'
        f'<td>{html.escape(str(a["answer"]))}</td><td>{"✓" if a["correct"] else "✗"}</td></tr>'
        for a in atoms)
    return (f'<details><summary>{label}</summary>'
            '<table border="1" cellpadding="3" style="border-collapse:collapse;font-size:12px">'
            '<tr><th>question</th><th>expected</th><th>answer</th><th></th></tr>'
            f'{rows}</table></details>')


def write_gallery(results, output):
    output = Path(output)
    blocks = []
    for entry in results:
        header = (f'<h3>{html.escape(entry["prompt_id"])} · {html.escape(entry["prompt"])} '
                  f'{_badge(entry["base"]["passed"])}</h3>')
        previews = {p['index']: p for p in entry['previews']}
        retries = {r['index']: r for r in entry['retries']}
        rows = []
        for index, preview in previews.items():
            retry = retries.get(index)
            cells = [f'<td><b>preview t={preview["t"]:.2f}</b><br><img src="images/{preview["image"]}" width="220">'
                     f'<br>{_badge(preview["passed"])}{_atoms(preview["atoms"], "preview atoms")}</td>',
                     f'<td><b>base final</b><br><img src="images/{entry["base"]["image"]}" width="220">'
                     f'<br>{_badge(entry["base"]["passed"])}{_atoms(entry["base"]["atoms"], "base atoms")}</td>']
            if retry:
                cells.append(f'<td><b>loop</b><br><img src="images/{retry["loop"]["image"]}" width="220">'
                             f'<br>{_badge(retry["loop"]["passed"])}'
                             f'<div style="font-size:11px;max-width:220px">{html.escape(retry["correction"])}</div>'
                             f'{_atoms(retry["loop"]["atoms"], "loop atoms")}</td>')
                cells.append(f'<td><b>resample (seed {retry["resample"]["seed"]})</b><br>'
                             f'<img src="images/{retry["resample"]["image"]}" width="220">'
                             f'<br>{_badge(retry["resample"]["passed"])}'
                             f'{_atoms(retry["resample"]["atoms"], "resample atoms")}</td>')
            else:
                cells.append('<td><i>no retry: preview passed</i></td><td></td>')
            rows.append(f'<tr>{"".join(cells)}</tr>')
        blocks.append(header + '<table cellpadding="8"><tr>' + '</tr><tr>'.join(rows) + '</tr></table>')

    document = ('<!doctype html><meta charset="utf-8"><title>closed-loop upper bound</title>'
                '<body style="font-family:Arial;background:#fff">'
                '<h1>Explicit closed-loop upper bound</h1>'
                '<p>base / loop (prompt correction) / resample (new seed) share the same '
                'detection and the same number of retry steps.</p>'
                + '\n'.join(blocks) + '</body>')
    (output / 'index.html').write_text(document)
