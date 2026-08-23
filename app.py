"""本地模拟终端原型：不依赖第三方 Python 包。"""

from __future__ import annotations

import json
import argparse
import hashlib
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"
STATE_FILE = DATA_DIR / "runtime_state.json"
AUTHORIZED_DIR = DATA_DIR / "authorized_memos"


def read_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def default_state():
    return {
        "focus_mode": False,
        "active_mode": None,
        "selected_contact": None,
        "authorized_sources": [],
        "reminders": [],
        "completed_events": [],
        "preference_adjustments": {},
    }


def load_state():
    state = default_state()
    if STATE_FILE.exists():
        state.update(read_json(STATE_FILE))

    # 兼容早期版本把授权来源保存为字符串的状态文件，避免升级后请求中断。
    state["authorized_sources"] = [
        source if isinstance(source, dict) else {"display_name": source, "stored_name": source}
        for source in state.get("authorized_sources", [])
    ]
    if state["active_mode"] is None and state.get("focus_mode"):
        state["active_mode"] = "focus"

    adjustments = state.get("preference_adjustments", {})
    if any(isinstance(value, int) for value in adjustments.values()):
        state["preference_adjustments"] = {"focus": adjustments}
    return state


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clear_active_music_mode(state):
    state["active_mode"] = None
    state["focus_mode"] = False


def choose_track(genre: str, exclude_id: str | None = None):
    tracks = read_json(DATA_DIR / "music_library.json")
    matching = [track for track in tracks if track["genre"] == genre]
    if not matching:
        return tracks[0]
    return next((track for track in matching if track["id"] != exclude_id), matching[0])


def record_music_preference(state, mode: str, genre: str, delta: int):
    adjustments = state["preference_adjustments"].setdefault(mode, {})
    adjustments[genre] = adjustments.get(genre, 0) + delta
    return adjustments[genre]


def parse_memo_file(path: Path):
    """读取用户已授权的 TXT、MD 或 JSON 备忘录，统一为项目内部字段。"""
    if path.suffix.lower() == ".json":
        entries = read_json(path)
        if not isinstance(entries, list):
            raise ValueError("JSON 备忘录应为事项数组。")
        return entries

    entries = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 7:
            raise ValueError(f"第 {line_number} 行格式错误，应包含 7 个以 | 分隔的字段。")
        date, time, duration, title, location, priority, content = fields
        entries.append(
            {
                "id": f"{path.stem}_{line_number}",
                "date": date,
                "time": time,
                "duration_minutes": int(duration),
                "title": title,
                "location": location,
                "priority": priority,
                "content": content,
            }
        )
    return entries


def authorized_memos(state):
    memos = []
    seen = set()
    for source in state["authorized_sources"]:
        source_path = AUTHORIZED_DIR / source["stored_name"]
        if not source_path.exists() and source["stored_name"] == "memo_demo.json":
            source_path = DATA_DIR / "memo_demo.json"
        if source_path.exists():
            for memo in parse_memo_file(source_path):
                key = (
                    memo.get("date"), memo.get("time"), memo.get("title"),
                    memo.get("location"), memo.get("content"), memo.get("duration_minutes"),
                )
                if key not in seen:
                    seen.add(key)
                    memos.append(memo)
    return memos


def schedule_event_key(memo):
    material = "|".join(
        str(memo.get(field, ""))
        for field in ("date", "time", "title", "location", "content", "duration_minutes")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def is_past_event(memo):
    try:
        event_time = datetime.fromisoformat(f"{memo['date']} {memo.get('time', '23:59')}")
        return event_time < datetime.now()
    except (KeyError, TypeError, ValueError):
        return False


def schedule_items(state):
    completed = set(state.get("completed_events", []))
    items = []
    for memo in authorized_memos(state):
        item = dict(memo)
        item["event_key"] = schedule_event_key(item)
        item["is_completed"] = item["event_key"] in completed
        item["is_past"] = is_past_event(item)
        items.append(item)
    return sorted(items, key=lambda item: (item.get("date", "9999-99-99"), item.get("time", "99:99")))


class AssistantHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        # 演示开发阶段始终加载最新页面脚本，避免浏览器缓存旧操作名。
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def send_json(self, payload, status=HTTPStatus.OK):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/bootstrap":
            profiles = read_json(DATA_DIR / "profiles_demo.json")
            state = load_state()
            # 新打开或刷新页面时不继承上一次会话的音乐模式。
            if state.get("active_mode") or state.get("focus_mode"):
                clear_active_music_mode(state)
                save_state(state)
            self.send_json(
                {
                    "contacts": read_json(DATA_DIR / "contacts.json"),
                    "music_library": read_json(DATA_DIR / "music_library.json"),
                    "profiles": profiles,
                    "state": state,
                    "demo_reference_date": "2026-08-23",
                }
            )
            return
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/action":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.handle_action(payload)
            self.send_json({"ok": True, "result": result, "state": load_state()})
        except Exception as error:
            self.send_json(
                {"ok": False, "error": f"本地处理失败：{error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_action(self, payload):
        action = payload["action"]
        state = load_state()

        if action == "select_contact":
            state["selected_contact"] = payload["contact_id"]
            save_state(state)
            return {"message": "已记录当前注视/选中的联系人。"}

        if action == "prepare_message":
            content = payload.get("content", "").strip()
            if not state["selected_contact"] or not content:
                raise ValueError("请先选择联系人并填写消息内容。")
            contact = next(
                item
                for item in read_json(DATA_DIR / "contacts.json")
                if item["id"] == state["selected_contact"]
            )
            return {
                "message": f"将通过常用消息应用向{contact['name']}发送：‘{content}’，是否确认？",
                "pending": {"contact": contact["name"], "content": content},
            }

        if action == "confirm_send":
            state["selected_contact"] = None
            save_state(state)
            return {"message": "模拟发送成功。", "simulated": True}

        if action == "cancel_message":
            state["selected_contact"] = None
            save_state(state)
            return {"message": "已取消本次发送。"}

        if action == "start_mode":
            mode = payload["mode"]
            if mode not in {"focus", "driving", "entertainment"}:
                raise ValueError("不支持的音乐模式。")
            state["active_mode"] = mode
            state["focus_mode"] = mode == "focus"
            save_state(state)
            return {"message": f"已进入{payload['mode_label']}，请选择想播放的音乐。"}

        if action == "stop_mode":
            clear_active_music_mode(state)
            save_state(state)
            return {"message": "已停止当前音乐模式。当前歌曲仍会显示，但不再记录该模式偏好。"}

        if action == "stop_mode_silent":
            clear_active_music_mode(state)
            save_state(state)
            return {"message": "已结束音乐模式会话。"}

        if action == "play_genre":
            mode = state["active_mode"]
            if not mode:
                raise ValueError("请先选择专注、开车或娱乐模式。")
            track = choose_track(payload["genre"])
            record_music_preference(state, mode, track["genre"], 1)
            save_state(state)
            return {
                "message": f"正在播放：{track['title']}。已记录本次{payload['mode_label']}播放偏好。",
                "track": track,
                "mode": mode,
            }

        if action == "like_track":
            mode = state["active_mode"]
            if not mode:
                raise ValueError("当前没有启用音乐模式，无法记录模式偏好。")
            genre = payload["genre"]
            adjustment = record_music_preference(state, mode, genre, 2)
            save_state(state)
            return {"message": "已记录喜欢反馈，当前歌曲继续播放。", "adjustment": adjustment}

        if action == "next_track":
            mode = state["active_mode"]
            if not mode:
                raise ValueError("当前没有启用音乐模式，无法记录模式偏好。")
            genre = payload["genre"]
            record_music_preference(state, mode, genre, -1)
            track = choose_track(genre, payload.get("current_track_id"))
            record_music_preference(state, mode, track["genre"], 1)
            save_state(state)
            return {"message": f"已跳过当前歌曲，正在播放：{track['title']}。", "track": track}

        if action in {"authorize_memo_file", "authorize_memo_files"}:
            files = payload.get("files") or [{
                "file_name": payload.get("file_name"),
                "file_content": payload.get("file_content"),
            }]
            if not isinstance(files, list) or not 1 <= len(files) <= 10:
                raise ValueError("请一次选择 1 至 10 个备忘录文件。")

            AUTHORIZED_DIR.mkdir(exist_ok=True)
            sources, total_items, names = [], 0, set()
            for uploaded in files:
                filename = Path(str(uploaded["file_name"])).name
                suffix = Path(filename).suffix.lower()
                content = str(uploaded["file_content"])
                if not filename or filename in names:
                    raise ValueError("授权文件名称不能为空，且不能重复。")
                if suffix not in {".txt", ".md", ".json"}:
                    raise ValueError("仅支持 .txt、.md 和 .json 格式的备忘录。")
                if not content.strip() or len(content.encode("utf-8")) > 1_000_000:
                    raise ValueError("每个备忘录内容不能为空，且大小不能超过 1 MB。")
                stored_path = AUTHORIZED_DIR / filename
                stored_path.write_text(content, encoding="utf-8")
                items = parse_memo_file(stored_path)
                if not items:
                    raise ValueError(f"{filename} 中没有可读取的事项。")
                names.add(filename)
                total_items += len(items)
                sources.append({"display_name": filename, "stored_name": filename})

            state["authorized_sources"] = sources
            save_state(state)
            return {
                "message": f"已授权 {len(sources)} 个文件，共读取 {total_items} 条事项；内容仅保存在本地。",
                "sources": sources,
            }

        if action == "query_tomorrow":
            if not state["authorized_sources"]:
                raise ValueError("请先授权备忘录文件。")
            memos = authorized_memos(state)
            # 演示数据固定以 2026-08-23 为基准，明天为 2026-08-24。
            tomorrow = "2026-08-24"
            items = sorted(
                (memo for memo in memos if memo["date"] == tomorrow),
                key=lambda memo: memo.get("time", "99:99"),
            )
            return {
                "message": "已从本地已授权备忘录整理明日事项。",
                "date": tomorrow,
                "items": items,
                "suggestion": "如高优先级事项间隔较短，建议提前完成其中的准备工作。",
            }

        if action == "query_schedule":
            if not state["authorized_sources"]:
                raise ValueError("请先授权备忘录文件。")
            items = schedule_items(state)
            return {
                "message": "已整理所有已授权备忘录中的日程。",
                "items": items,
                "total": len(items),
                "completed": sum(item["is_completed"] for item in items),
                "past": sum(item["is_past"] for item in items),
            }

        if action == "toggle_event_completion":
            event_key = str(payload["event_key"])
            valid_keys = {item["event_key"] for item in schedule_items(state)}
            if event_key not in valid_keys:
                raise ValueError("该日程不在当前已授权文件中。")
            completed = set(state.get("completed_events", []))
            if bool(payload.get("completed")):
                completed.add(event_key)
            else:
                completed.discard(event_key)
            state["completed_events"] = sorted(completed)
            save_state(state)
            return {"message": "已更新日程完成状态。"}

        if action == "create_reminder":
            state["reminders"].append(
                {"memo_id": "memo_001", "time": "2026-08-24 09:10"}
            )
            save_state(state)
            return {"message": "已在本项目内创建 9:10 的组会提醒。"}

        if action == "decline_reminder":
            return {"message": "已取消本次提醒；不会因此修改长期偏好。"}

        if action == "record_event":
            event = payload.get("event", {})
            
            # 1. 验证字段
            required_fields = ["modality", "timestamp_ms", "confidence", "payload"]
            for field in required_fields:
                if field not in event:
                    raise ValueError(f"缺少必填字段: {field}")
            
            # 2. 验证置信度范围
            if not 0.0 <= event["confidence"] <= 1.0:
                raise ValueError("置信度必须在 0~1 之间")
            
            # 3. 存入事件窗口（只保留最近10秒）
            state = load_state()
            event_window = state.get("event_window", [])
            event_window.append(event)
            # 删除10秒前的旧事件
            cutoff = event["timestamp_ms"] - 10000
            event_window = [e for e in event_window if e["timestamp_ms"] > cutoff]
            state["event_window"] = event_window
            save_state(state)
            
            return {"message": f"已记录 {event['modality']} 事件", "window_size": len(event_window)}
        
        raise ValueError("不支持的操作。")


def main():
    parser = argparse.ArgumentParser(description="多模态个性化终端助手本地原型")
    parser.add_argument("--port", type=int, default=8000, help="本地服务端口，默认 8000")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), AssistantHandler)
    print(f"本地原型已启动：http://127.0.0.1:{args.port}")
    print("按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
