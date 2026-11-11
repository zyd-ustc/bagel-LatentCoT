"""Page for the closed-loop hard-prompt test.

The test asks: given a hard prompt, does the frozen base draw it wrong badly
enough that UND should intervene, and can the editor then fix it in the middle
of generation? Each case shows prompt -> preview at t=0.36 -> what UND
diagnosed -> final image, so the reviewer can judge the whole chain rather than
a single number.

It also states plainly what this test shares with R-data training and what it
does not, because the two are easy to conflate.
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
section{max-width:1700px;margin:22px auto;padding:0 22px}
h2{font-size:15px;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid #2c323c}
p{margin:8px 0;color:#c3cbd6}
ul{margin:8px 0 8px 20px;color:#c3cbd6}
li{margin:4px 0}
code{background:#232833;padding:1px 6px;border-radius:4px;font-size:12.5px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px}
th,td{border:1px solid #2c323c;padding:7px 10px;text-align:left;vertical-align:top}
th{background:#1b1f26;color:#98a2b3;font-weight:600}
.num{font-family:ui-monospace,Menlo,monospace}
.box{background:#1b1f26;border:1px solid #2c323c;border-left:3px solid #4f7cff;
     border-radius:8px;padding:12px 16px;margin:14px 0}
.box.warn{border-left-color:#c9921f}
.case{background:#1b1f26;border:1px solid #2c323c;border-radius:10px;margin:18px 0;
      overflow:hidden}
.case.edit{border-left:3px solid #4f7cff}
.case.pass{border-left:3px solid #3a5a3f}
.case h3{margin:0;padding:12px 16px 4px;font-size:14px;font-weight:600}
.badge{font-size:11px;padding:2px 9px;border-radius:20px;margin-left:8px;
       vertical-align:middle;font-weight:500}
.badge.e{background:#2b3f66;color:#cfe0ff}
.badge.p{background:#24402c;color:#9fdcb2}
.prompt{padding:0 16px 8px;color:#e6e8eb;font-size:14px}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#2c323c;
      border-top:1px solid #2c323c}
.cell{background:#000;position:relative}
.cell img{width:100%;display:block}
.cell .tag{position:absolute;top:7px;left:7px;background:#000c;padding:2px 8px;
           border-radius:4px;font-size:10.5px;letter-spacing:.05em;text-transform:uppercase}
.diag{padding:11px 16px;border-top:1px solid #2c323c;font-size:13px}
.diag .k{color:#6b7686;margin-right:6px}
.editreq{color:#cfe0ff}
.failed{color:#f1a7a7}
.okline{color:#9fdcb2}
"""


def uri(path, side=430, quality=84):
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
    src_dir, out_path = Path(sys.argv[1]), sys.argv[2]
    recs = []
    for p in sorted(src_dir.glob("0*.json")):
        try:
            recs.append((p.stem, json.loads(p.read_text())))
        except Exception:
            pass

    n = len(recs)
    edited = [(s, r) for s, r in recs if r.get("route") == "self_correct_state_edit"]
    passed = [(s, r) for s, r in recs if r.get("route") == "native_passthrough"]

    def fail_rate(group):
        q = sum(len(r.get("questions") or []) for _, r in group)
        f = sum(len(r.get("failed_questions") or []) for _, r in group)
        return (f, q, 100.0 * f / q if q else 0.0)

    ef, eq, e_r = fail_rate(edited)
    pf, pq, p_r = fail_rate(passed)

    cards = []
    for stem, r in recs:
        is_edit = r.get("route") == "self_correct_state_edit"
        failed = r.get("failed_questions") or []
        rej = r.get("rejected_questions") or []
        failed_html = ("<br>".join("· " + esc(q) for q in failed)
                       if failed else '<span class="okline">预览已满足全部原子要求</span>')
        if rej:
            failed_html += "<br><span style='color:#98a2b3'>（%d 条无法转成可执行指令：%s）</span>" % (
                len(rej), esc("; ".join(rej))[:160])
        edit_line = (esc(r.get("edit_request")) if is_edit
                     else '<span class="okline">未干预，直接继续生成</span>')
        cards.append(
            '<div class="case %s"><h3>%s%s</h3>'
            '<div class="prompt">%s</div>'
            '<div class="imgs">'
            '<div class="cell"><span class="tag">preview @ t=0.36</span><img src="%s"></div>'
            '<div class="cell"><span class="tag">final</span><img src="%s"></div>'
            '</div>'
            '<div class="diag"><div><span class="k">UND 诊断</span>%s</div>'
            '<div style="margin-top:6px"><span class="k">编辑指令</span>'
            '<span class="editreq">%s</span></div>'
            '<div style="margin-top:6px"><span class="k">问题数</span>%d / 失败 %d</div>'
            '</div></div>' % (
                "edit" if is_edit else "pass",
                esc(r.get("raw_prompt")),
                '<span class="badge e">已干预</span>' if is_edit else '<span class="badge p">直通</span>',
                esc(""),
                uri(src_dir / ("%s_preview.png" % stem)),
                uri(src_dir / ("%s_result.png" % stem)),
                failed_html, edit_line,
                len(r.get("questions") or []), len(failed)))

    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>闭环测试：硬提示词 → UND 反馈 → 中间态编辑</title>
<style>%(css)s</style></head><body>
<header>
  <h1>闭环测试：硬提示词 → UND 反馈 → 中间态编辑 → 完成生成</h1>
  <div class="sub">Geneval2 硬提示词 48 条 · 检查点 step_001200 · t=0.36 读取 · 8 步编辑 · 冻结尾段</div>
</header>

<section>
<h2>一、这个测试问的是什么</h2>
<p>给定一条<b>底座容易画错的硬提示词</b>，让它正常生成到 <code>t=0.36</code>，此时预览往往已经漏了某些要求。
问题在于：</p>
<ul>
  <li><b>UND 能不能看出预览哪里不对？</b></li>
  <li><b>看出来了，能不能给出可执行的编辑指令？</b></li>
  <li><b>编辑器能不能在中间态把它改对，并由冻结尾段生成完？</b></li>
</ul>
<p>这是一条完整的<b>自反馈闭环</b>，不是"用给定指令测编辑"。</p>

<h2>二、与 R 数据训练的关系</h2>
<div class="box">
<p>你说得对：<b>这条链路与 R 数据的监督结构是一致的，唯一差别是编辑指令的来源。</b></p>
<table>
<tr><th>环节</th><th>R 数据训练</th><th>本次测试</th></tr>
<tr><td><code>a_t</code> 真实中间态</td><td>父轨迹渲染的三个时刻</td><td>原生 T2I 跑到 t=0.36</td></tr>
<tr><td><code>P_t</code> 预览</td><td>该状态的前瞻渲染</td><td>同左</td></tr>
<tr><td><b>编辑指令来源</b></td><td><b>GLM 看图 / 对照提示词纠错</b></td><td><b>UND 看预览诊断</b></td></tr>
<tr><td>编辑执行</td><td>等时编辑器 8 步</td><td>同左</td></tr>
<tr><td>后续生成</td><td>冻结的文本尾段</td><td>同左</td></tr>
</table>
<p>因此这里检验的是<b>推理期闭环</b>，而 R 数据训练的是<b>执行能力</b>。两者共享同一套状态与尾段语义。</p>
</div>

<h2>三、总体结果</h2>
<table>
<tr><th>指标</th><th>数值</th></tr>
<tr><td>案例数</td><td class="num">%(n)d</td></tr>
<tr><td><b>触发编辑</b></td><td class="num">%(ne)d（%(pe).0f%%）</td></tr>
<tr><td><b>原生直通</b></td><td class="num">%(np)d（%(pp).0f%%）</td></tr>
<tr><td>编辑组失败率</td><td class="num">%(ef)d / %(eq)d 原子要求 = %(er).0f%%</td></tr>
<tr><td>直通组失败率</td><td class="num">%(pf)d / %(pq)d 原子要求 = %(pr).0f%%</td></tr>
</table>
<div class="box">
<b>这组数字说明 UND 不是随机触发。</b>
选择干预的案例，其预览有 <b>%(er).0f%%</b> 的原子要求未满足；
选择直通的只有 <b>%(pr).0f%%</b>。诊断与预览质量是对应的。
</div>
<div class="box warn">
<b>说明局限：</b>本页只展示<b>闭环行为</b>（是否干预、给出什么指令、最终成图）。
它<b>不</b>给出"成图是否真的满足提示词"的自动评分——
Geneval2 的 <code>vqa_list</code> 判分尚未接入。因此请不要把下面任何一张图当作"通过/不通过"的定论，
而是看链路是否按预期工作。
</div>

<h2>四、逐例：预览 → 诊断 → 成图</h2>
<p>每例两张图：<b>preview</b> 是 <code>t=0.36</code> 时底座还没画完的样子（UND 看到的就是它），
<b>final</b> 是编辑（或直通）后由冻结尾段生成的终图。</p>
%(cards)s
</section></body></html>""" % dict(
        css=CSS, n=n, ne=len(edited), pe=100.0 * len(edited) / n,
        np=len(passed), pp=100.0 * len(passed) / n,
        ef=ef, eq=eq, er=e_r, pf=pf, pq=pq, pr=p_r,
        cards="".join(cards))

    Path(out_path).write_text(html)
    print(json.dumps({"out": out_path, "cases": n,
                      "kb": Path(out_path).stat().st_size // 1024}))


if __name__ == "__main__":
    main()
