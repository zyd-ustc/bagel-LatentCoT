"""Execution-only test page: given a correct single instruction, does it work?

The point of this page is to separate two failure sources the earlier run
conflated. In the closed loop a bad final image could come from picking the
wrong edit OR from failing to carry out a right one. Here the instruction is
taken from what the diagnosis produced and handed to the editor directly
(`explicit` route), so UND no longer participates and every failure is an
execution failure.

Layout per case: preview at t=0.36 (what the editor had to start from), the
instruction it was given, and the final render.
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
     font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{background:#1b1f26;border-bottom:1px solid #2c323c;padding:14px 22px}
h1{font-size:17px;margin:0 0 4px;font-weight:600}
.sub{color:#98a2b3;font-size:13px}
section{max-width:1400px;margin:22px auto;padding:0 22px}
h2{font-size:15px;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid #2c323c}
p{margin:8px 0;color:#c3cbd6}
ul{margin:8px 0 8px 20px;color:#c3cbd6}
code{background:#232833;padding:1px 6px;border-radius:4px;font-size:12.5px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px}
th,td{border:1px solid #2c323c;padding:7px 10px;text-align:left;vertical-align:top}
th{background:#1b1f26;color:#98a2b3;font-weight:600}
.box{background:#1b1f26;border:1px solid #2c323c;border-left:3px solid #4f7cff;
     border-radius:8px;padding:12px 16px;margin:14px 0}
.box.warn{border-left-color:#c9921f}
.case{background:#1b1f26;border:1px solid #2c323c;border-radius:10px;margin:18px 0;
      overflow:hidden}
.case h3{margin:0;padding:12px 16px 3px;font-size:14px}
.prompt{padding:0 16px 8px;color:#98a2b3;font-size:13px}
.instr{padding:0 16px 12px;color:#cfe0ff;font-size:14px;font-weight:600}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#2c323c;
      border-top:1px solid #2c323c}
.cell{background:#000;position:relative}
.cell img{width:100%;display:block}
.cell .tag{position:absolute;top:7px;left:7px;background:#000c;padding:2px 8px;
           border-radius:4px;font-size:10.5px;letter-spacing:.05em;text-transform:uppercase}
.diag{padding:10px 16px;border-top:1px solid #2c323c;font-size:12.5px;color:#98a2b3}
"""


def uri(path, side=460, quality=86):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((side, side), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def esc(t):
    return str(t or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    exec_dir, spec_path, out_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    spec = {json.loads(l)["uid"]: json.loads(l) for l in open(spec_path) if l.strip()}

    cards = []
    for p in sorted(exec_dir.glob("0*.json")):
        r = json.loads(p.read_text())
        stem = p.stem
        s = spec.get(r.get("uid"), {})
        nfe = r.get("nfe") or {}
        cards.append(
            '<div class="case"><h3>%s</h3>'
            '<div class="prompt">原始提示词：%s</div>'
            '<div class="instr">给定指令：%s</div>'
            '<div class="pair">'
            '<div class="cell"><span class="tag">preview @ t=0.36</span>'
            '<img src="%s"></div>'
            '<div class="cell"><span class="tag">final</span><img src="%s"></div>'
            '</div>'
            '<div class="diag">route=%s · 编辑步数=%s · 尾段调用=%s</div></div>' % (
                esc(stem), esc((s.get("raw_prompt") or "")[:120]),
                esc(r.get("edit_request") or s.get("edit_instruction") or ""),
                uri(exec_dir / ("%s_preview.png" % stem)),
                uri(exec_dir / ("%s_result.png" % stem)),
                esc(r.get("route")), esc(nfe.get("state_edit")), esc(nfe.get("tail"))))

    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>执行能力测试：给定正确单项指令</title>
<style>%(css)s</style></head><body>
<header>
  <h1>执行能力测试：给定正确的单项指令，编辑器能不能执行</h1>
  <div class="sub">7 例 · step_001200 · explicit 路径（绕开 UND）· t=0.36 · 8 步编辑</div>
</header>

<section>
<h2>一、这个测试要分离什么</h2>
<p>上一轮 48 例闭环里，一张图不对可能有两个原因：<b>选错了改什么</b>，或者
<b>选对了但改不动</b>。两者混在一起，无法判断该修哪一边。</p>
<p>本页走 <code>explicit</code> 路径：指令直接给定，<b>UND 不参与</b>。因此：</p>
<ul>
  <li>如果这里也失败 → 是<b>执行能力</b>问题；</li>
  <li>如果这里成功、闭环失败 → 是<b>诊断/指令</b>问题。</li>
</ul>
<div class="box warn">
<b>重要限制：只有 7 例。</b>这些是闭环里<b>恰好只有一条原子要求失败、且指令为单句</b>的案例，
因此可以干净归因。样本量不足以做统计结论，但足以判断"执行路径是否根本不通"。
</div>

<h2>二、实验设计</h2>
<table>
<tr><th>环节</th><th>设置</th></tr>
<tr><td>中间态</td><td>原生 T2I 生成到 <code>t=0.36</code>，取该状态</td></tr>
<tr><td>指令来源</td><td>UND 在上轮诊断中给出的那条指令，人工确认是单句且对应单一动作</td></tr>
<tr><td>目标条件</td><td>原始提示词 + 该编辑的派生描述（<code>derived_tail_condition</code>）</td></tr>
<tr><td>编辑</td><td>等时编辑器 8 步</td></tr>
<tr><td>后续生成</td><td>冻结的文本尾段（38 次调用）</td></tr>
</table>

<h2>三、逐例结果</h2>
<p>左列是编辑器看到的<b>起点</b>（t=0.36 的预览），右列是<b>最终成图</b>。</p>
%(cards)s

<h2>四、怎样判断</h2>
<p>看右列是否在<b>保留原场景主体</b>的前提下，完成了左列尚未满足的那一条要求：</p>
<ul>
  <li>计数类（toys / raccoons）：数量是否变成指令说的数字；</li>
  <li>添加类（giraffe / car / bicycles）：新对象是否出现且不破坏原构图；</li>
  <li>属性类（penguin 的位置）：关系是否真的改变；</li>
</ul>
<p>若右列整体结构崩坏或变得模糊，那是生成质量的问题，不是"没执行"。</p>
</section></body></html>""" % dict(css=CSS, cards="".join(cards))

    Path(out_path).write_text(html)
    print(json.dumps({"out": out_path, "cases": len(cards),
                      "kb": Path(out_path).stat().st_size // 1024}))


if __name__ == "__main__":
    main()
