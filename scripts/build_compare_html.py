"""Side-by-side page for the v1/v2 teacher comparison.

The comparison is not a verdict table — a number cannot tell whether a chisel
melted into a smear. This shows source, the old hybrid target and the new
full-IT2I target in one row, with the measured drift underneath, so the eye
decides and the numbers only annotate.
"""

import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#14161a;color:#e6e8eb;
     font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:#1b1f26;border-bottom:1px solid #2c323c;
       padding:14px 22px}
h1{font-size:16px;margin:0 0 6px;font-weight:600}
.sub{color:#98a2b3;font-size:13px}
.case{background:#1b1f26;border:1px solid #2c323c;border-radius:12px;margin:22px;
      overflow:hidden}
.case h2{font-size:15px;margin:0;padding:14px 16px 0;font-weight:600}
.case .meta{padding:6px 16px 12px;color:#98a2b3;font-size:13px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1px;
        background:#2c323c;border-top:1px solid #2c323c}
.panel{background:#000;position:relative}
.panel img{width:100%;display:block}
.panel .tag{position:absolute;top:8px;left:8px;background:#000b;padding:3px 9px;border-radius:5px;
            font-size:11px;letter-spacing:.06em;text-transform:uppercase}
.stats{display:flex;flex-wrap:wrap;gap:8px;padding:13px 16px;border-top:1px solid #2c323c;
       font-size:12.5px}
.stat{border:1px solid #333a46;border-radius:20px;padding:2px 10px;color:#98a2b3}
.stat b{color:#e6e8eb}
.bad{border-color:#7a3030}.bad b{color:#f18b8b}
.good{border-color:#2f6b45}.good b{color:#7fd6a2}
"""


def uri(path, side=460):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((side, side), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def panel(src, label):
    return ('<div class="panel"><span class="tag">%s</span>'
            '<img src="%s"></div>' % (label, src))


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    spec = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    html = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<title>%s</title><style>%s</style></head><body><header><h1>%s</h1>"
            "<div class='sub'>%s</div></header>"
            % (esc(spec["title"]), CSS, esc(spec["title"]), esc(spec.get("sub", "")))]

    for case in spec["cases"]:
        html.append("<div class='case'><h2>%s</h2>" % esc(case["name"]))
        html.append("<div class='meta'>%s</div>" % esc(case.get("note", "")))
        html.append("<div class='panels'>")
        html.append(panel(uri(case["source"]), "source  x_src"))
        for item in case["variants"]:
            html.append(panel(uri(item["image"]), esc(item["label"])))
        html.append("</div><div class='stats'>")
        for item in case["variants"]:
            m = item["metrics"]
            html.append(
                "<span class='stat'>%s &middot; edit Δ <b>%s</b> &middot; "
                "keep Δ <b class='%s'>%s</b> &middot; global Δ <b>%s</b> &middot; "
                "hf&times; <b class='%s'>%s</b></span>"
                % (esc(item["label"]), m["edit"], "bad" if m["keep"] > 0.10 else "good",
                   m["keep"], m["global"], "bad" if m["hf"] > 1.5 else "good", m["hf"]))
        html.append("</div></div>")

    html.append("</body></html>")
    out.write_text("".join(html))
    print(json.dumps({"out": str(out), "kb": out.stat().st_size // 1024}))


if __name__ == "__main__":
    main()
