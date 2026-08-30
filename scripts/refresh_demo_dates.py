"""重新生成演示备忘录日期，使其相对“真实今天”对齐，方便随时测试。

运行：python scripts/refresh_demo_dates.py

- 把 app.py 的 DEMO_REFERENCE_DATE 更新为真实今天的日期；
- 把 memo_demo.json、备忘录演示.txt、备忘录演示1.txt 及已授权副本的日期改写为
  相对布局：昨天(过期灰显示例) / 今天 / 明天 / 后天。
运行后“今天/明天/后天/已过时间”全部与真实日历一致，无需再手动改日期。
"""

import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 标题 -> (日期偏移, 时间, 优先级)；标题/地点/内容保持不变
LAYOUT = {
    "提交周报": (-1, "16:00", "high"),      # 昨天 → 灰显示例，且不会进入提醒建议
    "组会": (1, "10:00", "high"),           # 明天
    "提交实验报告": (1, "14:00", "high"),   # 明天
    "健身": (1, "19:00", "low"),            # 明天
    "论文阅读": (2, "09:30", "medium"),     # 后天
    "午睡": (2, "12:00", "low"),            # 后天
}


def main():
    today = date.today()

    # 1) 基准日期已由 app.py 动态取服务器启动时的真实日期，无需改写；本脚本只负责
    #    重新生成演示备忘录的相对日期（昨天过期示例 / 明天 / 后天）。

    # 2) memo_demo.json
    memo_path = ROOT / "data" / "memo_demo.json"
    items = json.loads(memo_path.read_text(encoding="utf-8"))
    for item in items:
        spec = LAYOUT.get(item["title"])
        if spec:
            offset, time_str, priority = spec
            item["date"] = (today + timedelta(days=offset)).isoformat()
            item["time"] = time_str
            item["priority"] = priority
    memo_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[2/3] memo_demo.json 已更新（{len(items)} 条）")

    # 3) TXT 演示文件与已授权副本
    txt_paths = [
        ROOT / "data" / "备忘录演示.txt",
        ROOT / "data" / "备忘录演示1.txt",
        ROOT / "data" / "authorized_memos" / "19ab078c7e82_备忘录演示.txt",
        ROOT / "data" / "authorized_memos" / "bbc47dccd9ae_备忘录演示1.txt",
    ]
    for path in txt_paths:
        if not path.exists():
            continue
        lines = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.startswith("#"):
                lines.append(raw)
                continue
            fields = [field.strip() for field in raw.split("|")]
            spec = LAYOUT.get(fields[3]) if len(fields) == 7 else None
            if spec:
                offset, time_str, priority = spec
                fields[0] = (today + timedelta(days=offset)).isoformat()
                fields[1] = time_str
                fields[5] = priority
                raw = " | ".join(fields)
            lines.append(raw)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[3/3] {path.name} 已更新")

    print(f"完成：今天={today.isoformat()}，明天={(today + timedelta(days=1)).isoformat()}，"
          f"后天={(today + timedelta(days=2)).isoformat()}，过期示例={today - timedelta(days=1)}。")

    # 4) 日期变更后，提醒/拒绝/完成状态的键全部失效，清空以免残留旧键。
    state_path = ROOT / "data" / "runtime_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

        def clean(node):
            if isinstance(node, dict):
                if "reminders" in node:
                    node["reminders"] = []
                if "declined_reminder_offers" in node:
                    node["declined_reminder_offers"] = []
                if "completed_events" in node:
                    node["completed_events"] = []
                for value in node.values():
                    clean(value)

        clean(state)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[4/4] runtime_state.json 的提醒/拒绝/完成状态已清空")


if __name__ == "__main__":
    main()
