"""本地模拟终端原型：不依赖第三方 Python 包。"""

from __future__ import annotations

import json
import argparse
import hashlib
import re
import time
from threading import Lock
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
EVENT_TTL_MS = 10_000
EVENT_BUFFER_LIMIT = 100
MULTIMODAL_EVENT_BUFFER = []
MULTIMODAL_EVENT_LOCK = Lock()


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
        "pending_message": None,
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


def current_time_ms():
    return int(time.time() * 1000)


def recent_multimodal_events():
    """返回仍在有效期内的结构化事件；事件仅存于进程内存。"""
    now = current_time_ms()
    with MULTIMODAL_EVENT_LOCK:
        MULTIMODAL_EVENT_BUFFER[:] = [
            item for item in MULTIMODAL_EVENT_BUFFER
            if item["received_at_ms"] >= now - EVENT_TTL_MS
        ]
        return [item.copy() for item in MULTIMODAL_EVENT_BUFFER]


def record_multimodal_event(event):
    if not isinstance(event, dict):
        raise ValueError("event 必须是对象。")
    modality = str(event.get("modality", "")).strip()
    if modality not in {"gaze", "screen_context", "speech_text", "head_gesture"}:
        raise ValueError("modality 仅支持 gaze、screen_context、speech_text 或 head_gesture。")
    timestamp_ms = event.get("timestamp_ms")
    if not isinstance(timestamp_ms, int) or timestamp_ms <= 0:
        raise ValueError("timestamp_ms 必须是正整数毫秒时间戳。")
    confidence = event.get("confidence", 1.0)
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence 必须在 0 到 1 之间。")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("payload 必须是对象。")
    if modality == "gaze":
        required = {"page", "target_type", "target_id", "zone", "dwell_ms"}
        if not required.issubset(payload):
            raise ValueError("gaze 事件缺少页面、目标、区域或停留时长字段。")
        if not isinstance(payload["dwell_ms"], int) or payload["dwell_ms"] < 0:
            raise ValueError("dwell_ms 必须是非负整数。")
    if modality == "speech_text" and not str(payload.get("text", "")).strip():
        raise ValueError("speech_text 事件必须包含非空 text。")
    if modality == "head_gesture":
        if payload.get("page") != "message" or payload.get("decision") not in {"confirm", "reject"}:
            raise ValueError("head_gesture 事件必须包含消息页和 confirm/reject 决策。")

    received_at_ms = current_time_ms()
    stored = {
        "modality": modality,
        "timestamp_ms": timestamp_ms,
        "confidence": round(float(confidence), 3),
        "payload": payload,
        "received_at_ms": received_at_ms,
    }
    with MULTIMODAL_EVENT_LOCK:
        MULTIMODAL_EVENT_BUFFER[:] = [
            item for item in MULTIMODAL_EVENT_BUFFER
            if item["received_at_ms"] >= received_at_ms - EVENT_TTL_MS
        ]
        MULTIMODAL_EVENT_BUFFER.append(stored)
        del MULTIMODAL_EVENT_BUFFER[:-EVENT_BUFFER_LIMIT]
    return {"message": "已在本地短时缓存中记录多模态事件。", "event": stored}


def find_contact(contact_id):
    return next(
        (item for item in read_json(DATA_DIR / "contacts.json") if item["id"] == contact_id),
        None,
    )


def parse_simulated_speech(text):
    normalized = re.sub(r"\s+", "", text)
    if normalized in {"不", "不是", "不用", "暂不", "取消", "取消发送", "不要", "不要发送", "不发送"}:
        return "cancel", ""
    if normalized in {"是", "是的", "确认", "确认发送", "发送", "好的", "好"}:
        return "confirm", ""
    match = re.search(r"(?:给他|给她|给它|给)(?:发消息|发送消息|发信息|发送信息)[，,、：:]?(.+)", normalized)
    if match and match.group(1).strip():
        return "send_message", match.group(1).strip()
    if any(word in normalized for word in ("发消息", "发送消息", "发信息", "发送信息")):
        return "send_message", ""
    return "unknown", ""


def understand_multimodal_command(state, speech_timestamp_ms, preferred_contact_id=None):
    events = recent_multimodal_events()
    speech_events = [
        item for item in events
        if item["modality"] == "speech_text" and abs(item["timestamp_ms"] - speech_timestamp_ms) <= 1_000
    ]
    if not speech_events:
        raise ValueError("未找到对应的模拟语音事件，请重新提交。")
    speech = min(speech_events, key=lambda item: abs(item["timestamp_ms"] - speech_timestamp_ms))
    intent, content = parse_simulated_speech(str(speech["payload"].get("text", "")))

    if intent in {"confirm", "cancel"}:
        pending = state.get("pending_message")
        if not pending:
            return {"message": "当前没有待确认的消息操作。", "intent": intent, "explanation": ["未检测到待确认任务。"]}
        state["pending_message"] = None
        state["selected_contact"] = None
        save_state(state)
        message = "模拟发送成功。" if intent == "confirm" else "已取消本次发送。"
        return {"message": message, "intent": intent, "clear_message_form": True, "explanation": [f"识别到确认词：{speech['payload']['text']}"]}

    if intent != "send_message":
        return {"message": "暂未理解该模拟语音。可尝试“给他发消息，晚点开会”。", "intent": "unknown", "explanation": ["未匹配到当前支持的消息指令。"]}
    if not content:
        return {"message": "请补充需要发送的消息内容。", "intent": intent, "needs_clarification": True, "explanation": ["识别到发送消息意图，但缺少消息正文。"]}

    manual_contact = find_contact(preferred_contact_id) if preferred_contact_id else None
    gaze_events = [
        item for item in events
        if item["modality"] == "gaze"
        and item["payload"].get("page") == "message"
        and item["payload"].get("target_type") == "contact"
        and speech_timestamp_ms - 4_000 <= item["timestamp_ms"] <= speech_timestamp_ms + 1_000
    ]
    gaze = max(gaze_events, key=lambda item: (item["confidence"], item["payload"].get("dwell_ms", 0)), default=None)
    if manual_contact:
        contact = manual_contact
        target_explanation = f"本轮优先使用手动选中的联系人：{contact['name']}。"
    elif gaze:
        contact = find_contact(gaze["payload"].get("target_id"))
        if not contact:
            return {"message": "注视目标不在当前联系人页面，请重新注视联系人。", "intent": intent, "needs_clarification": True, "explanation": ["注视事件中的联系人 ID 无法匹配本地联系人。"]}
        context_events = [
            item for item in events
            if item["modality"] == "screen_context"
            and item["payload"].get("page") == "message"
            and abs(item["timestamp_ms"] - speech_timestamp_ms) <= 5_000
        ]
        visible_ids = {
            target.get("target_id")
            for context in context_events
            for target in context["payload"].get("visible_targets", [])
        }
        if visible_ids and contact["id"] not in visible_ids:
            return {"message": "联系人页面已变化，请重新注视后再试。", "intent": intent, "needs_clarification": True, "explanation": ["屏幕上下文中不包含该注视联系人。"]}
        target_explanation = f"在语音前后 4 秒内检测到对{contact['name']}的稳定注视（{gaze['payload']['dwell_ms']}ms，置信度 {gaze['confidence']}）"
    else:
        contact = find_contact(state.get("selected_contact"))
        if not contact:
            return {"message": "请先注视确认或手动点击需要联系的联系人，再提交模拟语音。", "intent": intent, "needs_clarification": True, "explanation": ["未找到有效视线事件，也没有手动选中的联系人。"]}
        target_explanation = f"未使用有效视线事件，改用手动选中的联系人：{contact['name']}。"

    pending = {"contact": contact["name"], "contact_id": contact["id"], "content": content}
    state["selected_contact"] = contact["id"]
    state["pending_message"] = pending
    save_state(state)
    return {
        "message": f"将通过常用消息应用向{contact['name']}发送：‘{content}’，是否确认？",
        "intent": intent,
        "pending": pending,
        "explanation": [
            f"识别到模拟语音：{speech['payload']['text']}",
            target_explanation,
            "当前屏幕上下文为联系人页面，因此将“他”解析为该联系人。",
        ],
    }


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

        if action == "record_multimodal_event":
            return record_multimodal_event(payload.get("event"))

        if action == "get_recent_multimodal_events":
            events = recent_multimodal_events()
            return {"message": "已读取本地短时多模态事件。", "events": events}

        if action == "understand_multimodal_command":
            timestamp_ms = payload.get("speech_timestamp_ms")
            if not isinstance(timestamp_ms, int):
                raise ValueError("speech_timestamp_ms 必须是语音事件的毫秒时间戳。")
            return understand_multimodal_command(state, timestamp_ms, payload.get("preferred_contact_id"))

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
            pending = {"contact": contact["name"], "contact_id": contact["id"], "content": content}
            state["pending_message"] = pending
            save_state(state)
            return {
                "message": f"将通过常用消息应用向{contact['name']}发送：‘{content}’，是否确认？",
                "pending": pending,
            }

        if action == "confirm_send":
            state["selected_contact"] = None
            state["pending_message"] = None
            save_state(state)
            return {"message": "模拟发送成功。", "simulated": True}

        if action == "cancel_message":
            state["selected_contact"] = None
            state["pending_message"] = None
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
