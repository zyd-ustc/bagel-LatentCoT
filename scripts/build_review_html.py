"""Build a static human-review gallery for isochronic samples."""

import argparse
import html
import json
from pathlib import Path


def _rows(root):
    rows = []
    for path in sorted(Path(root).glob("*.json")):
        if path.name == "complete.json":
            continue
        data = json.loads(path.read_text())
        index = path.stem
        rows.append(
            dict(
                index=index,
                uid=data.get("uid", index),
                raw_prompt=data.get("raw_prompt", ""),
                seed=data.get("seed"),
                route=data.get("route", ""),
                failed=data.get("failed_questions", []),
                edit=data.get("edit_request", ""),
                preserve=data.get("preserve", ""),
                nfe=data.get("nfe", {}),
                seconds=data.get("seconds"),
                preview=f"{Path(root).name}/{index}_preview.png",
                result=f"{Path(root).name}/{index}_result.png",
            )
        )
    return rows


def _card(row, kind):
    failed = "".join(f"<li>{html.escape(str(q))}</li>" for q in row["failed"])
    nfe = " · ".join(f"{k}={v}" for k, v in row["nfe"].items() if v)
    keep = "" if not row["edit"] else f"<div class='edit'>Edit: {html.escape(row['edit'])}</div>"
    preserve = "" if not row["preserve"] else f"<div class='edit'>Preserve: {html.escape(row['preserve'])}</div>"
    return f"""
<article class="card" data-kind="{kind}" data-uid="{html.escape(row['uid'])}" data-route="{html.escape(row['route'])}">
  <div class="meta">
    <span class="badge {kind}">{kind}</span>
    <span class="badge route">{html.escape(row['route'])}</span>
    <span class="uid">{html.escape(row['uid'])}</span>
  </div>
  <div class="prompt">{html.escape(row['raw_prompt'])}</div>
  {keep}{preserve}
  <div class="images">
    <figure><img loading="lazy" src="{html.escape(row['preview'])}" alt="preview"><figcaption>preview 0.36</figcaption></figure>
    <figure><img loading="lazy" src="{html.escape(row['result'])}" alt="result"><figcaption>result</figcaption></figure>
  </div>
  <details><summary>诊断 / NFE</summary>
    <ul>{failed}</ul>
    <div class="small">{html.escape(nfe)}{' · %.2fs' % row['seconds'] if row['seconds'] else ''}</div>
  </details>
  <div class="review">
    <button data-v="pass">通过</button>
    <button data-v="fail">失败</button>
    <button data-v="unsure">不确定</button>
    <span class="decision"></span>
  </div>
</article>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--keep-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    samples = _rows(args.sample_dir)
    keeps = _rows(args.keep_dir)
    cards = [_card(row, "edit" if row["route"] == "self_correct_state_edit" else "pass") for row in samples]
    cards += [_card(row, "keep") for row in keeps]
    edits = sum(row["route"] == "self_correct_state_edit" for row in samples)
    passes = len(samples) - edits
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Isochronic Stage 1 Review</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;--ok:#3fb950;--bad:#f85149;--warn:#d29922}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{position:sticky;top:0;z-index:5;background:#0d1117ee;border-bottom:1px solid var(--line);padding:14px 20px;backdrop-filter:blur(8px)}}
h1{{font-size:18px;margin:0 0 6px}} .summary{{color:var(--dim);font-size:13px}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}} input,button{{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font:inherit}}
button{{cursor:pointer}} button:hover{{border-color:#58a6ff}} .active{{border-color:#58a6ff;color:#58a6ff}}
main{{padding:18px;display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}} .meta{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}}
.badge{{font-size:11px;border:1px solid var(--line);border-radius:999px;padding:2px 8px;color:var(--dim)}} .badge.edit{{color:var(--warn);border-color:#9e6a03}} .badge.keep{{color:var(--ok);border-color:#238636}}
.prompt{{color:var(--fg);margin:6px 0 8px;word-break:break-word}} .edit{{font-size:12px;color:var(--warn);margin:4px 0;word-break:break-word}}
.images{{display:grid;grid-template-columns:1fr 1fr;gap:8px}} figure{{margin:0}} img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#000;border:1px solid var(--line);cursor:zoom-in}}
figcaption{{font-size:11px;color:var(--dim);text-align:center;margin-top:4px}} details{{margin-top:8px;color:var(--dim)}} details ul{{margin:6px 0;padding-left:18px;font-size:12px}}
.small{{font-size:11px;word-break:break-all}} .review{{display:flex;gap:6px;align-items:center;margin-top:10px}}
.review button[data-v=pass]:hover{{color:var(--ok)}} .review button[data-v=fail]:hover{{color:var(--bad)}} .decision{{font-size:12px;color:var(--dim)}}
.hidden{{display:none}}
</style></head><body>
<header><h1>Isochronic Stage 1 — 人工审查</h1>
<div class="summary">checkpoint step_003607 · closed-loop: {len(samples)}（直通 {passes} / 编辑 {edits}） · keep: {len(keeps)} · 点图放大 · 标记保存在浏览器 localStorage</div>
<div class="controls">
<button class="filter active" data-f="all">全部</button><button class="filter" data-f="edit">编辑</button><button class="filter" data-f="pass">直通</button><button class="filter" data-f="keep">保持</button>
<input id="q" placeholder="搜索 uid / prompt / edit">
<button id="export">导出决策 JSON</button><button id="reset">清空决策</button>
<span id="counts" class="summary"></span>
</div></header><main>{''.join(cards)}</main>
<script>
const KEY='iso-review-v1';
function load(){{try{{return JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{return {{}}}}}}
function save(x){{localStorage.setItem(KEY,JSON.stringify(x))}}
let decisions=load();
function paint(card){{
  const uid=card.dataset.uid, d=decisions[uid]||'';
  card.querySelectorAll('.review button').forEach(b=>b.classList.toggle('active',b.dataset.v===d));
  card.querySelector('.decision').textContent=d?({{pass:'已标记通过',fail:'已标记失败',unsure:'待复核'}}[d]||''):'';
}}
document.querySelectorAll('.card').forEach(paint);
document.querySelectorAll('.card').forEach(card=>{{
  card.querySelectorAll('.review button').forEach(btn=>btn.onclick=()=>{{
    const uid=card.dataset.uid;
    decisions[uid]=decisions[uid]===btn.dataset.v?'':btn.dataset.v; save(decisions); paint(card); updateCounts();
  }});
}});
document.querySelectorAll('.filter').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('.filter').forEach(x=>x.classList.remove('active')); b.classList.add('active');
  const f=b.dataset.f;
  document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hidden', f!=='all' && c.dataset.kind!==f));
}});
document.getElementById('q').oninput=e=>{{
  const q=e.target.value.toLowerCase();
  document.querySelectorAll('.card').forEach(c=>{{
    c.classList.toggle('hidden', q && !c.textContent.toLowerCase().includes(q));
  }});
}};
function updateCounts(){{
  const vals=Object.values(decisions).filter(Boolean);
  const c={{pass:0,fail:0,unsure:0}}; vals.forEach(v=>c[v]=(c[v]||0)+1);
  document.getElementById('counts').textContent=`已审 ${{vals.length}} / 通过 ${{c.pass||0}} / 失败 ${{c.fail||0}} / 待复核 ${{c.unsure||0}}`;
}}
updateCounts();
document.getElementById('export').onclick=()=>{{
  const blob=new Blob([JSON.stringify({{checkpoint:'step_003607',decisions}},null,2)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='iso_review_decisions.json';a.click();
}};
document.getElementById('reset').onclick=()=>{{if(confirm('清空本页决策？')){{decisions={{}};save(decisions);document.querySelectorAll('.card').forEach(paint);updateCounts();}}}};
document.querySelectorAll('img').forEach(img=>img.onclick=()=>window.open(img.src,'_blank'));
</script></body></html>"""
    out = Path(args.out)
    out.write_text(body)
    print(json.dumps(dict(out=str(out), samples=len(samples), keeps=len(keeps), edits=edits, passes=passes)))


if __name__ == "__main__":
    main()
