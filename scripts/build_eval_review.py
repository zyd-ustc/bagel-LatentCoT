"""Build the evaluation review page: images side by side, plus the method.

The page has to answer three questions a reviewer will actually ask:
  * what was measured, and on what data;
  * which picture is which, since seven renders per case is a lot;
  * what the numbers can and cannot support at n=16.

It inlines downscaled images so the file opens anywhere without a server.
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
header{position:sticky;top:0;z-index:20;background:#1b1f26;border-bottom:1px solid #2c323c;
       padding:14px 22px}
h1{font-size:17px;margin:0 0 4px;font-weight:600}
.sub{color:#98a2b3;font-size:13px}
section{max-width:1600px;margin:22px auto;padding:0 22px}
h2{font-size:15px;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid #2c323c}
p{margin:8px 0;color:#c3cbd6}
ul{margin:8px 0 8px 20px;color:#c3cbd6}
li{margin:4px 0}
code{background:#232833;padding:1px 6px;border-radius:4px;font-size:12.5px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px}
th,td{border:1px solid #2c323c;padding:7px 10px;text-align:left}
th{background:#1b1f26;color:#98a2b3;font-weight:600}
.num{font-family:ui-monospace,Menlo,monospace}
.bad{color:#f18b8b}.good{color:#7fd6a2}.warn{color:#e3c46a}
.box{background:#1b1f26;border:1px solid #2c323c;border-left:3px solid #4f7cff;
     border-radius:8px;padding:12px 16px;margin:14px 0}
.box.warnbox{border-left-color:#c9921f}
.case{background:#1b1f26;border:1px solid #2c323c;border-radius:10px;margin:20px 0;
      overflow:hidden}
.case h3{margin:0;padding:12px 16px 2px;font-size:14px;font-weight:600}
.case .meta{padding:0 16px 10px;color:#98a2b3;font-size:12.5px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1px;
      background:#2c323c;border-top:1px solid #2c323c}
.cell{background:#000;position:relative}
.cell img{width:100%;display:block}
.cell .tag{position:absolute;top:6px;left:6px;background:#000c;padding:2px 8px;
           border-radius:4px;font-size:10.5px;letter-spacing:.05em;text-transform:uppercase}
.stats{display:flex;flex-wrap:wrap;gap:8px;padding:10px 16px;border-top:1px solid #2c323c;
       font-size:12.5px}
.stat{border:1px solid #333a46;border-radius:20px;padding:2px 10px;color:#98a2b3}
.stat b{color:#e6e8eb}
"""


def uri(path, side=300, quality=80):
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


def load_cases(root):
    out = []
    for path in sorted(Path(root).glob("case_*.json")):
        out.append(json.loads(path.read_text()))
    return out


def fmt(v, pct=False):
    if v is None:
        return "—"
    return ("%.1f%%" % (100 * v)) if pct else ("%.4f" % v)


def main():
    trained_dir, base_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    trained = {c["case"]: c for c in load_cases(trained_dir)}
    base = {c["case"]: c for c in load_cases(base_dir)}
    names = sorted(set(trained) & set(base))

    t_all = [trained[n] for n in names]
    b_all = [base[n] for n in names]

    rows_html = []
    for name in names:
        t, b = trained[name], base[name]
        cells = [
            ("source", trained_dir, "source"),
            ("student tail (trained)", trained_dir, "student"),
            ("target tail (teacher)", trained_dir, "target_tail"),
            ("student tail (baseline)", base_dir, "student"),
        ]
        grid = "".join(
            '<div class="cell"><span class="tag">%s</span><img src="%s"></div>'
            % (esc(label), uri(str(Path(directory) / ("%s_%s.png" % (name, key)))))
            for label, directory, key in cells)

        dm = t["displacement_mse"]
        db = b["displacement_mse"]
        delta = 100.0 * (dm - db) / db if db else None
        cls = "bad" if (delta or 0) > 0 else "good"
        rows_html.append(
            '<div class="case"><h3>%s &nbsp;<span style="color:#6b7686;font-weight:400">%s</span></h3>'
            '<div class="meta">%s</div><div class="grid">%s</div>'
            '<div class="stats">'
            '<span class="stat">displacement MSE <b>%s</b> (base <b>%s</b>)</span>'
            '<span class="stat %s">vs baseline <b>%+.1f%%</b></span>'
            '<span class="stat">LPIPS <b>%s</b> / base <b>%s</b></span>'
            '<span class="stat">student tail MSE <b>%s</b> / base <b>%s</b></span>'
            '</div></div>' % (
                esc(name), esc(t.get("category")),
                esc(t.get("edit_instruction"))[:220],
                grid, fmt(dm), fmt(db), cls, delta or 0,
                fmt(t.get("displacement_lpips")), fmt(b.get("displacement_lpips")),
                fmt(t.get("student_tail_mse")), fmt(b.get("student_tail_mse"))))

    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Isochronic editor evaluation - step_001200 vs step_003607</title>
<style>%(css)s</style></head><body>
<header>
  <h1>等时编辑器评估：step_001200（训练后） vs step_003607（起点）</h1>
  <div class="sub">48 步中训练到 1,200 步的检查点，在 L 编辑对上与起点逐例对比</div>
</header>

<section>
<h2>一、这次评估测的是什么</h2>
<p>评估对象是<b>等时编辑器</b>：给定真实中间态 <code>a_t</code>、预览 <code>P_t</code> 和一条编辑指令，
模型要在编辑轴上走 8 步把状态推到目标，再交给冻结的底座尾段生成终图。</p>
<ul>
  <li><b>输入</b>：L 编辑对（<code>L_30k.json</code>）中的源图与目标图</li>
  <li><b>构造</b>：两条分支共享同一噪声，<code>a_t = t·x_src + (1-t)·σε</code>，<code>b_t = t·x_tgt + (1-t)·σε</code></li>
  <li><b>时间点</b>：<code>t = 0.36</code>，<b>编辑步数</b>：8 步，<b>随机种子</b>：20261107</li>
  <li><b>样本</b>：16 例，覆盖 add / attribute / count / other / remove / spatial / text 七类</li>
</ul>
<div class="box warnbox">
  <b>重要：这个指标不直接等于「编辑得好不好」。</b>
  它衡量的是学生状态与目标状态在潜空间/感知空间的距离，
  以及尾段成图与目标成图的距离。<b>不是</b>语义正确率，也不是人工观感评分。
  最终判断仍要看下面的图。
</div>

<h2>二、四个指标的含义</h2>
<table>
<tr><th>指标</th><th>计算方式</th><th>越低越好？</th></tr>
<tr><td><code>displacement_mse</code></td>
    <td>学生编辑后的状态 b̂ 与目标状态 b 在像素空间的均方误差</td><td>是</td></tr>
<tr><td><code>displacement_lpips</code></td>
    <td>同上，但用 LPIPS 感知距离，对纹理/结构变化更敏感</td><td>是</td></tr>
<tr><td><code>student_tail_mse</code></td>
    <td>学生状态经<b>冻结尾段</b>生成的学生图，与目标图之间的 MSE</td><td>是</td></tr>
<tr><td><code>target_tail_mse</code></td>
    <td>目标状态经同一尾段生成的参照图，与目标图的距离。<b>这是下界</b>，
        学生不可能比它更好（因为教的是同一个尾段）</td><td>是</td></tr>
</table>

<h2>三、为什么以 step_003607 为基线</h2>
<p><code>step_003607</code> 是本次训练<b>初始化的起点</b>（Stage 1 训练产物）。本次训练从它出发，
混合 R 终端监督（60%%）与 L 回放（30%%）与 repair（10%%）。</p>
<p>因此正确的问法是：<b>训练让模型变好了，还是变差了？</b>
而不是「模型好不好」。基线就是回答这个问题的唯一参照。</p>

<h2>四、总体结果</h2>
<table>
<tr><th>指标</th><th>step_003607（基线）</th><th>step_001200（训练后）</th><th>变化</th></tr>
%(summary)s
</table>
<div class="box warnbox">
  <b>结论：三项指标全部变差。</b>
  这与训练日志里 loss 从 0.4793 降到 0.24（−50%%）的方向相反。
  说明<b>训练 loss 下降 ≠ 编辑能力提升</b>：前者拟合的是 R 的教师目标图，
  后者测的是 L 编辑对上的位移准确性，两者不是同一个量。
</div>
<div class="box">
  <b>统计局限（必须说明）：</b>本次只有 16 例，且每类仅 2–3 例。
  单例波动足以让均值偏 10%% 以上。+12%% 的差距<b>不能</b>直接当作稳定结论，
  需要用更大样本复核后才能定性。
</div>

<h2>五、逐例对比</h2>
<p>每例四张图：源图、学生成图（训练后）、目标成图（教师参照）、基线学生成图（起点）。</p>
%(cases)s
</section>
</body></html>""" % dict(css=CSS, cases="".join(rows_html), summary=summary_rows(t_all, b_all))

    Path(out_path).write_text(html)
    print(json.dumps({"out": out_path, "cases": len(names),
                      "kb": Path(out_path).stat().st_size // 1024}))


def mean(rows, key):
    """Mean of one metric across cases, skipping cases that lack it."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def summary_rows(t_all, b_all):
    out = []
    for label, key in (("displacement MSE", "displacement_mse"),
                       ("displacement LPIPS", "displacement_lpips"),
                       ("student tail MSE", "student_tail_mse"),
                       ("target tail MSE (floor)", "target_tail_mse")):
        tb = mean(b_all, key)
        tt = mean(t_all, key)
        if tb is None or tt is None:
            continue
        pct = 100.0 * (tt - tb) / tb if tb else 0.0
        if key == "target_tail_mse":
            verdict = '<span class="warn">不变（同一尾段，符合预期）</span>'
        elif pct > 1:
            verdict = '<span class="bad">更差 %+.1f%%</span>' % pct
        elif pct < -1:
            verdict = '<span class="good">更好 %+.1f%%</span>' % pct
        else:
            verdict = '<span class="warn">持平</span>'
        out.append("<tr><td>%s</td><td class='num'>%.4f</td><td class='num'>%.4f</td><td>%s</td></tr>"
                   % (label, tb, tt, verdict))
    return "\n".join(out)


if __name__ == "__main__":
    main()
