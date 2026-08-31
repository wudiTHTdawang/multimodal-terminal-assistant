# -*- coding: utf-8 -*-
"""
提交材料 PDF 生成脚本
- 将本目录下的 .md 文档转换为排版良好的 PDF（A4，中文字体）
- 零第三方依赖：markdown -> HTML -> Edge/Chrome 无头模式打印为 PDF
用法：python 生成PDF.py
输出：按 附件1 命名规范生成 <团队名>_<项目名>_<材料名>.pdf
"""
import html
import pathlib
import re
import subprocess
import sys

BASE = pathlib.Path(__file__).resolve().parent

# ============ 按需修改 ============
TEAM_NAME = "XX团队"                      # TODO: 替换为真实团队名称
PROJECT_NAME = "多模态个性化终端助手"
# ================================

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
EDGE = next((p for p in EDGE_CANDIDATES if pathlib.Path(p).exists()), None)

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Microsoft YaHei", "SimSun", sans-serif; font-size: 10.5pt; line-height: 1.65; color: #1a1a1a; }
h1 { font-size: 16pt; border-bottom: 2px solid #2b5ca8; padding-bottom: 6px; color: #1f3a63; }
h2 { font-size: 13.5pt; color: #2b5ca8; margin-top: 18px; border-left: 4px solid #2b5ca8; padding-left: 8px; }
h3 { font-size: 11.5pt; color: #333; margin-top: 12px; }
h4 { font-size: 10.5pt; color: #444; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #eef3fb; }
pre { background: #f6f8fa; border: 1px solid #ddd; border-radius: 4px; padding: 8px;
      font-family: Consolas, monospace; font-size: 9pt; white-space: pre-wrap; word-break: break-all; }
code { font-family: Consolas, monospace; background: #f0f0f0; padding: 1px 3px; border-radius: 3px; font-size: 9pt; }
blockquote { border-left: 3px solid #bbb; margin: 8px 0; padding: 2px 10px; color: #555; background: #fafafa; }
li { margin: 2px 0; }
hr { border: none; border-top: 1px solid #ccc; margin: 10px 0; }
"""


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s


def render_table(rows):
    def cells(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]

    data = [cells(r) for r in rows]
    header = data[0]
    body = [r for r in data[1:] if not re.match(r"^[\s:\-|]*$", "|".join(r))]
    thead = "<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr></thead>"
    tbody = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body
    ) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def md_to_html(text):
    lines = text.split("\n")
    out, i, in_code, code_buf = [], 0, False, []
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if not in_code:
                in_code, code_buf = True, []
            else:
                in_code = False
                out.append("<pre>" + html.escape("\n".join(code_buf)) + "</pre>")
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        # 表格
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|\-]+\|\s*$", lines[i + 1]):
            rows, j = [], i
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j])
                j += 1
            out.append(render_table(rows))
            i = j
            continue
        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lv = len(m.group(1))
            out.append(f"<h{lv}>{inline(m.group(2))}</h{lv}>")
            i += 1
            continue
        if re.match(r"^\s*---+$", line):
            out.append("<hr/>")
            i += 1
            continue
        # 引用
        if line.strip().startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append("<blockquote>" + inline("<br/>".join(buf)) + "</blockquote>")
            continue
        # 列表
        mo = re.match(r"^\s*\d+\.\s+(.*)$", line)
        mu = re.match(r"^\s*[-*]\s+(.*)$", line)
        if mo or mu:
            ordered = bool(mo)
            pat = (r"^\s*\d+\.\s+(.*)$" if ordered else r"^\s*[-*]\s+(.*)$")
            buf = []
            while i < len(lines) and re.match(pat, lines[i]):
                buf.append(inline(re.match(pat, lines[i]).group(1)))
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{b}</li>" for b in buf) + f"</{tag}>")
            continue
        if not line.strip():
            i += 1
            continue
        # 段落（收集到空行或块级开头）
        buf, i = [line], i + 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("```") \
                and not lines[i].strip().startswith("|") and not re.match(r"^(#{1,6})\s", lines[i]) \
                and not re.match(r"^\s*[-*]\s+", lines[i]) and not re.match(r"^\s*\d+\.\s+", lines[i]) \
                and not lines[i].strip().startswith(">"):
            buf.append(lines[i])
            i += 1
        out.append("<p>" + inline(" ".join(x.strip() for x in buf)) + "</p>")
    return "\n".join(out)


def to_pdf(md_path: pathlib.Path, out_pdf: pathlib.Path):
    body = md_to_html(md_path.read_text(encoding="utf-8"))
    doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>{html.escape(md_path.stem)}</title><style>{CSS}</style></head><body>{body}</body></html>"""
    tmp_html = BASE / ("_tmp_" + md_path.stem + ".html")
    tmp_html.write_text(doc, encoding="utf-8")
    try:
        cmd = [EDGE, "--headless=new", "--disable-gpu", "--no-sandbox",
               f"--print-to-pdf={out_pdf}", "--no-pdf-header-footer",
               "--virtual-time-budget=3000", tmp_html.as_uri()]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    finally:
        tmp_html.unlink(missing_ok=True)
    if not out_pdf.exists():
        raise RuntimeError(f"PDF 未生成: {out_pdf}")
    print(f"OK  {out_pdf.name}  ({out_pdf.stat().st_size // 1024} KB)")


def main():
    if EDGE is None:
        sys.exit("未找到 Edge/Chrome，请安装浏览器后重试")
    mds = sorted(p for p in BASE.glob("*.md") if p.name != "生成PDF.py")
    if not mds:
        sys.exit("提交材料目录下没有 .md 文档")
    for md in mds:
        out = BASE / f"{TEAM_NAME}_{PROJECT_NAME}_{md.stem}.pdf"
        to_pdf(md, out)
    print(f"\n共生成 {len(mds)} 个 PDF，输出到 {BASE}")
    print("注意：请将文件名中的 团队名 替换为实际名称。")


if __name__ == "__main__":
    main()
