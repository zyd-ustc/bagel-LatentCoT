"""Build one self-contained HTML page for eyeballing constructed samples.

Whoever reviews needs the two pictures, the instruction, the target the pair was
supposed to reach, and every machine verdict side by side — plus a way to say
what they think. This writes a single file with the images inlined as base64, so
it opens from disk with no server and no network.

Each card carries three votes (keep / reject / unsure). Votes live in
localStorage and can be exported as JSON, which is the file to bring back:
a judgement made by eye is the one signal no model in this pipeline can supply.

Usage::

    python3 scripts/build_review_html.py --rows <jsonl> --limit 60 \
        --out review.html --title "..."
"""

import argparse
import base64
import json
import mimetypes
import random
from pathlib import Path

CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:#14161a; color:#e6e8eb; }
header { position:sticky; top:0; z-index:10; background:#1b1f26; border-bottom:1px solid #2c323c;
         padding:14px 22px; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
h1 { font-size:16px; margin:0; font-weight:600; }
.stat { color:#98a2b3; font-size:13px; }
.stat b { color:#e6e8eb; }
button { font:inherit; cursor:pointer; border-radius:7px; border:1px solid #333a46;
         background:#232833; color:#e6e8eb; padding:6px 13px; }
button:hover { background:#2b313d; }
button.on { border-color:#4f7cff; background:#233152; color:#cfe0ff; }
.wrap { padding:22px; display:grid; gap:20px;
        grid-template-columns:repeat(auto-fill,minmax(520px,1fr)); }
.card { background:#1b1f26; border:1px solid #2c323c; border-radius:12px; overflow:hidden;
        display:flex; flex-direction:column; }
.card.keep { border-color:#2f6b45; } .card.reject { border-color:#7a3030; }
.card.unsure { border-color:#7a6526; }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#2c323c; }
.pane { position:relative; background:#000; }
.pane img { width:100%; display:block; }
.pane .tag { position:absolute; top:7px; left:7px; background:#000a; padding:2px 8px;
             border-radius:5px; font-size:11px; letter-spacing:.06em; text-transform:uppercase; }
.meta { padding:13px 15px; display:flex; flex-direction:column; gap:9px; flex:1; }
.instr { font-size:15px; font-weight:600; color:#fff; }
.row { font-size:12.5px; color:#98a2b3; }
.row span.k { color:#6b7686; margin-right:6px; }
.target { color:#c7d2e0; }
.chips { display:flex; gap:6px; flex-wrap:wrap; }
.chip { font-size:11.5px; padding:2px 9px; border-radius:20px; border:1px solid #333a46; }
.chip.ok { border-color:#2f6b45; color:#7fd6a2; }
.chip.bad { border-color:#7a3030; color:#f18b8b; }
.chip.mid { border-color:#7a6526; color:#e3c46a; }
.chip.na { color:#6b7686; }
.votes { display:flex; gap:8px; padding:11px 15px; border-top:1px solid #2c323c; }
.votes button { flex:1; }
.uid { font-size:11px; color:#5b6574; font-family:ui-monospace,Menlo,monospace; }
.note { width:100%; background:#12151a; border:1px solid #2c323c; border-radius:7px;
        color:#e6e8eb; padding:7px 10px; font:inherit; font-size:12.5px; }
"""

JS = """
const KEY='review-votes-v1';
const all=()=>JSON.parse(localStorage.getItem(KEY)||'{}');
const save=v=>localStorage.setItem(KEY,JSON.stringify(v));
function paint(){
  const v=all(); let n=[0,0,0];
  document.querySelectorAll('.card').forEach(c=>{
    const u=c.dataset.uid, rec=v[u];
    c.classList.remove('keep','reject','unsure');
    c.querySelectorAll('.votes button').forEach(b=>b.classList.remove('on'));
    if(rec){ c.classList.add(rec.vote);
      const b=c.querySelector('.votes button[data-v="'+rec.vote+'"]'); if(b) b.classList.add('on');
      if(rec.vote==='keep')n[0]++; else if(rec.vote==='reject')n[1]++; else n[2]++; }
    const t=c.querySelector('.note'); if(t && rec && rec.note && t.value!==rec.note) t.value=rec.note;
  });
  const total=document.querySelectorAll('.card').length;
  document.getElementById('counts').innerHTML=
    `<b>${n[0]+n[1]+n[2]}</b> / ${total} reviewed &nbsp;·&nbsp; `+
    `<span style="color:#7fd6a2">keep <b>${n[0]}</b></span> &nbsp;·&nbsp; `+
    `<span style="color:#f18b8b">reject <b>${n[1]}</b></span> &nbsp;·&nbsp; `+
    `<span style="color:#e3c46a">unsure <b>${n[2]}</b></span>`;
}
function vote(u,v){ const a=all(); a[u]=Object.assign({},a[u]||{},{vote:v}); save(a); paint(); }
function note(u,t){ const a=all(); a[u]=Object.assign({},a[u]||{},{note:t}); save(a); }
function dump(){
  const a=all();
  const meta=JSON.parse(document.getElementById('meta').textContent);
  const out=Object.entries(a).map(([uid,r])=>({uid,title:meta.title,vote:r.vote||null,note:r.note||''}));
  const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  const link=document.createElement('a');
  link.href=URL.createObjectURL(blob);
  link.download='review_votes.json'; link.click();
}
function setAll(v){ const a=all(); document.querySelectorAll('.card').forEach(c=>{
    a[c.dataset.uid]=Object.assign({},a[c.dataset.uid]||{},{vote:v}); }); save(a); paint(); }
document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.votes button').forEach(b=>b.onclick=()=>{
    const c=b.closest('.card'); const cur=all()[c.dataset.uid];
    vote(c.dataset.uid, (cur&&cur.vote===b.dataset.v)?null:b.dataset.v); });
  document.querySelectorAll('.note').forEach(t=>t.oninput=()=>note(t.closest('.card').dataset.uid,t.value));
  document.getElementById('export').onclick=dump;
  document.querySelectorAll('[data-setall]').forEach(b=>b.onclick=()=>setAll(b.dataset.setall));
  paint();
});
"""

CARD = """
<div class="card" data-uid="{uid}">
  <div class="pair">
    <div class="pane"><span class="tag">A · before</span><img src="{before}"></div>
    <div class="pane"><span class="tag">B · after</span><img src="{after}"></div>
  </div>
  <div class="meta">
    <div class="instr">{instruction}</div>
    <div class="row"><span class="k">target</span><span class="target">{tail}</span></div>
    <div class="chips">{chips}</div>
    <div class="row uid">{uid}</div>
    <input class="note" placeholder="note (what is wrong, or why it is good)">
  </div>
  <div class="votes">
    <button data-v="keep">✓ keep</button>
    <button data-v="reject">✗ reject</button>
    <button data-v="unsure">? unsure</button>
  </div>
</div>
"""


def data_uri(path, side=384):
    """Inline a downscaled JPEG.

    Full 512 PNGs run about 500 KB each, so a 48-card page with two pictures per
    card is a 50 MB file that takes seconds to paint. Review only needs to judge
    whether the edit happened and whether the rest moved; 384 px is enough for
    that and keeps a page well under a megabyte.
    """
    if not path:
        return None
    try:
        from PIL import Image
        import io
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((side, side), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=82, optimize=True)
            payload = buffer.getvalue()
    except Exception:
        try:
            payload = Path(path).read_bytes()
        except OSError:
            return None
        mime = mimetypes.guess_type(path)[0] or "image/png"
        return "data:%s;base64,%s" % (mime, base64.b64encode(payload).decode())
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode()


def esc(text):
    return (str(text or "").replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def chips_for(row):
    out = []
    pix = row.get("pixel") or {}
    if pix:
        delta = pix.get("edit_delta")
        if pix.get("passed"):
            out.append('<span class="chip ok">pixels pass</span>')
        else:
            label = {"no_change": "pixels: no change",
                     "preserve_broken": "pixels: preserve broken"}.get(pix.get("reason"), "pixels fail")
            out.append('<span class="chip bad">%s</span>' % esc(label))
        if delta is not None:
            out.append('<span class="chip na">edit Δ %s</span>' % delta)
        if pix.get("broken"):
            worst = max(b["delta"] for b in pix["broken"])
            out.append('<span class="chip bad">drift Δ %s</span>' % worst)
    g = row.get("glm_fields") or {}
    for key, name in (("EDIT_DONE", "edit"), ("TARGET_MET", "target"),
                      ("KEEP_OK", "keep"), ("QUALITY_OK", "quality")):
        val = (g.get(key) or "").strip().lower()
        if not val:
            continue
        cls = "ok" if val == "yes" else ("mid" if val == "partial" else "bad")
        out.append('<span class="chip %s">%s %s</span>' % (cls, name, esc(val)))
    if row.get("family"):
        out.append('<span class="chip na">%s</span>' % esc(row["family"]))
    if row.get("operation"):
        out.append('<span class="chip na">%s</span>' % esc(row["operation"]))
    if any(str(x).startswith("glm_verified") for x in []):
        pass
    return "".join(out) or '<span class="chip na">unscored</span>'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--title", default="Constructed samples")
    ap.add_argument("--stratum", default="mix",
                    help="mix: balanced sample of scored states; all: every row given")
    ap.add_argument("--with-plan", action="append", default=[],
                    help="plan jsonl files; lets instruction-only rows be reviewed too")
    ap.add_argument("--with-records", action="append", default=[],
                    help="coupled caches; fills picture paths for rows that only "
                         "carry a verdict (the reject file records just the verdict)")
    args = ap.parse_args()

    pairs = {}
    for path in args.with_plan:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                row.setdefault("stage", "plan")
                pairs.setdefault(row["uid"], {}).update(row)

    # Rejected rows store only uid + verdict, so the pictures have to be looked
    # up from the cache that produced them.
    if args.with_records:
        import glob
        for root in args.with_records:
            for path in glob.glob(str(Path(root) / "progress" / "*.json")):
                try:
                    record = json.loads(Path(path).read_text())
                except Exception:
                    continue
                uid = record.get("uid")
                slot = pairs.setdefault(uid, {})
                for key in ("source_image", "target_image", "edit_instruction",
                            "tail_condition", "edit_box", "preserve_boxes",
                            "parent_condition"):
                    if not slot.get(key) and record.get(key):
                        slot[key] = record[key]
    for path in args.rows:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                slot = pairs.setdefault(row["uid"], {})
                slot.update(row)

    cards, skipped = [], 0
    rows = list(pairs.values())
    random.Random(args.seed).shuffle(rows)

    if args.stratum == "mix":
        # Review is only useful if it covers the states that disagree, so the
        # sample is drawn per outcome rather than uniformly.
        buckets = {}
        for row in rows:
            pix = (row.get("pixel") or {}).get("passed")
            approved = row.get("glm_fields") and all(
                (row["glm_fields"].get(k) or "").lower() == "yes" for k in ("EDIT_DONE",))
            if row.get("stage") == "plan":
                key = "plan_only"
            elif pix is False:
                key = "pixel_fail"
            elif approved:
                key = "glm_ok"
            else:
                key = "glm_doubt"
            buckets.setdefault(key, []).append(row)
        share = {"pixel_fail": 0.30, "glm_ok": 0.25, "glm_doubt": 0.25, "plan_only": 0.20}
        # Some buckets come up empty (nothing passes both gates), so the quota
        # is redistributed over what actually exists instead of losing slots.
        picked = []
        for key, weight in share.items():
            take = int(round(args.limit * weight))
            picked.extend(buckets.get(key, [])[:take])
        for key in sorted(buckets, key=lambda k: -len(buckets[k])):
            if len(picked) >= args.limit:
                break
            have = {row["uid"] for row in picked}
            for row in buckets[key]:
                if len(picked) >= args.limit:
                    break
                if row["uid"] not in have:
                    picked.append(row)
        seen = {row["uid"] for row in picked}
        for row in rows:
            if len(picked) >= args.limit:
                break
            if row["uid"] not in seen:
                picked.append(row)
                seen.add(row["uid"])
        rows = picked
    else:
        rows = rows[: args.limit]

    for row in rows:
        before = data_uri(row.get("source_image"))
        after = data_uri(row.get("target_image"))
        if not before:
            skipped += 1
            continue
        if not after:
            # An instruction with no picture yet: show the pre-edit frame alone
            # rather than dropping the row, so instruction quality can be read.
            after = before
        cards.append(CARD.format(
            uid=esc(row["uid"]), before=before, after=after,
            instruction=esc(row.get("edit_instruction") or "(no instruction)"),
            tail=esc(row.get("tail_condition") or "(no target scene)"),
            chips=chips_for(row)))

    meta = json.dumps(dict(title=args.title, count=len(cards)))
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head><body>
<header>
  <h1>{title}</h1>
  <span class="stat" id="counts"></span>
  <span class="stat">A = before edit · B = after edit</span>
  <button data-setall="keep">all keep</button>
  <button data-setall="reject">all reject</button>
  <button id="export">export votes</button>
</header>
<div class="wrap">{cards}</div>
<script id="meta" type="application/json">{meta}</script>
<script>{js}</script>
</body></html>""".format(title=esc(args.title), css=CSS, js=JS, meta=meta,
                        cards="".join(cards))
    Path(args.out).write_text(html)
    print(json.dumps(dict(cards=len(cards), skipped=skipped,
                          kb=round(len(html) / 1024), out=args.out)))


if __name__ == "__main__":
    main()
