# -*- coding: utf-8 -*-
"""生成预览 HTML（用于截图检查排版，非提交物）"""
import importlib.util
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gen", BASE / "生成PDF.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

for name in ["参赛作品简介", "项目文档", "可解释性说明"]:
    md = (BASE / f"{name}.md").read_text(encoding="utf-8")
    doc = f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/><style>{m.CSS}</style></head><body>{m.md_to_html(md)}</body></html>'
    (BASE / f"_preview_{name}.html").write_text(doc, encoding="utf-8")
print("preview html written")
