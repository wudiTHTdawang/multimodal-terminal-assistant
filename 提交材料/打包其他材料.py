# -*- coding: utf-8 -*-
"""打包“其他辅助材料”zip（代码 + 文档），命名符合 附件1 规范"""
import pathlib
import zipfile

BASE = pathlib.Path(__file__).resolve().parent.parent
TEAM_NAME = "XX团队"
PROJECT_NAME = "多模态个性化终端助手"
OUT = BASE / "提交材料" / f"{TEAM_NAME}_{PROJECT_NAME}_其他辅助材料.zip"

DOCS = ["可解释性说明.md", "使用说明.md", "测试结果报告.md", "项目文档.md", "项目视频脚本.md"]
INCLUDE = ["app.py", "llm_reasoner.py", "README.md", "web", "data", "scripts", "tests"]

EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", "提交材料", "node_modules"}
EXCLUDE_FILES = {"runtime_state.json"}


def walk(base: pathlib.Path, rel=""):
    for p in sorted(base.iterdir()):
        if p.is_dir():
            if p.name in EXCLUDE_DIRS or rel + p.name + "/" in (".git/",):
                continue
            yield from walk(p, f"{rel}{p.name}/")
        else:
            if p.name in EXCLUDE_FILES or p.name.endswith(".pyc"):
                continue
            yield p, rel + p.name


def main():
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in INCLUDE:
            p = BASE / name
            if p.is_dir():
                for fp, arc in walk(p, f"{name}/"):
                    z.write(fp, arc)
                    n += 1
            elif p.exists():
                z.write(p, name)
                n += 1
        for d in DOCS:
            p = BASE / "提交材料" / d
            if p.exists():
                z.write(p, f"文档/{d}")
                n += 1
    size = OUT.stat().st_size / 1e6
    print(f"OK {OUT.name}  文件数={n}  大小={size:.1f} MB")


if __name__ == "__main__":
    main()
