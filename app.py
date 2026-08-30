"""本地模拟终端原型：不依赖第三方 Python 包。"""

from __future__ import annotations

import json
import argparse
import copy
import hashlib
import re
import time
from threading import Lock, RLock
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from llm_reasoner import enhance_local_response, suggest_local_actions, normalize_speech_command


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"
STATE_FILE = DATA_DIR / "runtime_state.json"
AUTHORIZED_DIR = DATA_DIR / "authorized_memos"
EVENT_TTL_MS = 10_000
EVENT_BUFFER_LIMIT = 100
DEFAULT_PROFILE_ID = "user_xiaoyu"
# 演示数据基准日期：随服务器启动时的真实日期，“今天/明天/已过时间”据此推算；
# 可用 scripts/refresh_demo_dates.py 重新生成演示备忘录的相对日期以对齐。
DEMO_REFERENCE_DATE = datetime.now().strftime("%Y-%m-%d")
LOCAL_LLM_ENABLED = True
INTERACTION_HISTORY_LIMIT = 200
UNDO_STACK_LIMIT = 12
PROFILE_STATE_FIELDS = (
    "focus_mode", "active_mode", "selected_contact", "authorized_sources", "reminders",
    "declined_reminder_offers", "completed_events", "track_preferences",
    "mode_preference_playlists", "recommendation_turns", "pending_message",
    "interaction_history", "undo_stack", "gesture_profile",
)
MULTIMODAL_EVENT_BUFFER = []
MULTIMODAL_EVENT_LOCK = Lock()
# 状态文件读改写统一由 STATE_LOCK 串行化（RLock 允许同一线程内嵌套加锁）；
# STATE_CACHE 为进程内权威状态，首次从磁盘加载后常驻内存，避免每个请求整文件重读。
STATE_LOCK = RLock()
STATE_CACHE = None


def read_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_memo_text(path: Path):
    """兼容 Windows 记事本常见的 UTF-8 与 GB18030 备忘录编码。"""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别 {path.name} 的文本编码，请保存为 UTF-8 或 GB18030 后重试。")


def default_state():
    return {
        "focus_mode": False,
        "active_mode": None,
        "selected_contact": None,
        "authorized_sources": [],
        "reminders": [],
        "declined_reminder_offers": [],
        "completed_events": [],
        "track_preferences": {},
        "mode_preference_playlists": {},
        "recommendation_turns": {},
        "pending_message": None,
        "interaction_history": [],
        "undo_stack": [],
        "gesture_profile": {"head_min_strength": 0.0, "confirmed_samples": [], "undone_samples": []},
        "active_profile_id": DEFAULT_PROFILE_ID,
        "profile_states": {},
    }


def profile_runtime_defaults(profile_id=None):
    state = {
        "focus_mode": False, "active_mode": None, "selected_contact": None,
        "authorized_sources": [], "reminders": [], "declined_reminder_offers": [],
        "completed_events": [],
        "track_preferences": {},
        "mode_preference_playlists": {}, "recommendation_turns": {}, "pending_message": None,
        "interaction_history": [], "undo_stack": [],
        "gesture_profile": {"head_min_strength": 0.0, "confirmed_samples": [], "undone_samples": []},
    }
    if not profile_id:
        return state
    profile = next((item for item in read_json(DATA_DIR / "profiles_demo.json") if item["id"] == profile_id), None)
    if not profile:
        return state
    tracks = read_json(DATA_DIR / "music_library.json")
    for mode, genre_scores in profile.get("music_preferences", {}).items():
        scores = {
            track["id"]: genre_scores.get(track.get("genre"), 0)
            for track in tracks if mode in track.get("moods", []) and genre_scores.get(track.get("genre"), 0) != 0
        }
        if scores:
            state["track_preferences"][mode] = scores
            state["mode_preference_playlists"][mode] = [track_id for track_id, score in scores.items() if score > 0]
    return state


def profile_snapshot(state):
    defaults = profile_runtime_defaults()
    return {field: copy.deepcopy(state.get(field, defaults[field])) for field in PROFILE_STATE_FIELDS}


def apply_profile_snapshot(state, snapshot):
    defaults = profile_runtime_defaults()
    for field in PROFILE_STATE_FIELDS:
        state[field] = copy.deepcopy(snapshot.get(field, defaults[field]))


def synchronize_active_profile(state):
    profile_id = state.get("active_profile_id", DEFAULT_PROFILE_ID)
    state.setdefault("profile_states", {})[profile_id] = profile_snapshot(state)


def switch_active_profile(state, profile_id):
    available = {profile["id"] for profile in read_json(DATA_DIR / "profiles_demo.json")}
    if profile_id not in available:
        raise ValueError("未找到该演示用户。")
    synchronize_active_profile(state)
    profiles = state.setdefault("profile_states", {})
    profiles.setdefault(profile_id, profile_runtime_defaults(profile_id))
    state["active_profile_id"] = profile_id
    apply_profile_snapshot(state, profiles[profile_id])
    clear_active_music_mode(state)


def load_state():
    """返回进程内权威状态（首次从磁盘加载后常驻内存）。

    所有读改写必须在 STATE_LOCK 内进行；save_state 负责原子落盘，
    避免并发请求 last-writer-wins 互相覆盖，也避免每个请求整文件重读。
    """
    global STATE_CACHE
    if STATE_CACHE is None:
        STATE_CACHE = _load_state_from_disk()
    return STATE_CACHE


def _load_state_from_disk():
    state = default_state()
    if STATE_FILE.exists():
        state.update(read_json(STATE_FILE))

    if not state.get("profile_states"):
        initial_profile = profile_snapshot(state) if STATE_FILE.exists() else profile_runtime_defaults(DEFAULT_PROFILE_ID)
        state["profile_states"] = {DEFAULT_PROFILE_ID: initial_profile}
    state["active_profile_id"] = state.get("active_profile_id") or DEFAULT_PROFILE_ID
    state["profile_states"].setdefault(state["active_profile_id"], profile_runtime_defaults(state["active_profile_id"]))
    apply_profile_snapshot(state, state["profile_states"][state["active_profile_id"]])

    # 兼容早期版本把授权来源保存为字符串的状态文件，避免升级后请求中断。
    state["authorized_sources"] = [
        source if isinstance(source, dict) else {"display_name": source, "stored_name": source}
        for source in state.get("authorized_sources", [])
    ]
    if state["active_mode"] is None and state.get("focus_mode"):
        state["active_mode"] = "focus"

    # 清理已废弃的 genre 级偏好字段：track_preferences / mode_preference_playlists 为当前事实标准。
    state.pop("preference_adjustments", None)
    for profile_state in state.get("profile_states", {}).values():
        if isinstance(profile_state, dict):
            profile_state.pop("preference_adjustments", None)
    state["declined_reminder_offers"] = (
        state.get("declined_reminder_offers") if isinstance(state.get("declined_reminder_offers"), list) else []
    )
    state["track_preferences"] = state.get("track_preferences") if isinstance(state.get("track_preferences"), dict) else {}
    state["mode_preference_playlists"] = state.get("mode_preference_playlists") if isinstance(state.get("mode_preference_playlists"), dict) else {}
    state["recommendation_turns"] = state.get("recommendation_turns") if isinstance(state.get("recommendation_turns"), dict) else {}
    state["interaction_history"] = state.get("interaction_history") if isinstance(state.get("interaction_history"), list) else []
    state["undo_stack"] = state.get("undo_stack") if isinstance(state.get("undo_stack"), list) else []
    state["gesture_profile"] = state.get("gesture_profile") if isinstance(state.get("gesture_profile"), dict) else {}
    state["gesture_profile"].setdefault("head_min_strength", 0.0)
    state["gesture_profile"]["confirmed_samples"] = [
        value for value in state["gesture_profile"].get("confirmed_samples", [])
        if isinstance(value, (int, float)) and 0 < value < 1
    ][-30:]
    state["gesture_profile"]["undone_samples"] = [
        value for value in state["gesture_profile"].get("undone_samples", [])
        if isinstance(value, (int, float)) and 0 < value < 1
    ][-20:]
    return state


def save_state(state):
    with STATE_LOCK:
        synchronize_active_profile(state)
        payload = json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
        # 先写临时文件再原子替换，避免写一半被并发读取到不完整 JSON。
        temp_path = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
        temp_path.write_bytes(payload)
        temp_path.replace(STATE_FILE)


def clear_active_music_mode(state):
    state["active_mode"] = None
    state["focus_mode"] = False


def current_time_ms():
    return int(time.time() * 1000)


def demo_reference_date():
    """演示基准日期（datetime.date）。"""
    return datetime.strptime(DEMO_REFERENCE_DATE, "%Y-%m-%d").date()


def demo_today_str():
    """演示基准日期的字符串形式，如 2026-08-23。"""
    return demo_reference_date().strftime("%Y-%m-%d")


def demo_now():
    """演示“当前时刻”：基准日期 + 运行机器当前钟表时刻。

    日期部分完全确定（测试可复现），时间部分仍能演示“同一天早于现在的事项
    已过时间”的灰显功能，不依赖运行机器的真实日期。
    """
    return datetime.combine(demo_reference_date(), datetime.now().time())


def append_interaction_history(state, *, page, action, modality="ui", target_id=None, outcome="applied", confidence=None, metrics=None):
    """保存可解释的本地操作摘要；绝不保存原始画面、语音文本或消息正文。"""
    record = {
        "timestamp_ms": current_time_ms(),
        "page": str(page)[:20],
        "action": str(action)[:50],
        "modality": str(modality)[:20],
        "outcome": str(outcome)[:20],
    }
    if target_id:
        record["target_id"] = str(target_id)[:80]
    if isinstance(confidence, (int, float)):
        record["confidence"] = round(float(confidence), 3)
    if isinstance(metrics, dict):
        filtered = {
            key: round(float(value), 4)
            for key, value in metrics.items()
            if key in {"motion_strength", "vertical_range", "horizontal_range"}
            and isinstance(value, (int, float))
        }
        # 白名单过滤后为空时不写入空的 metrics 字段，避免历史记录出现无意义噪音。
        if filtered:
            record["metrics"] = filtered
    state.setdefault("interaction_history", []).append(record)
    del state["interaction_history"][:-INTERACTION_HISTORY_LIMIT]
    return record


# 打开页面与撤销属于导航/管理噪音，不参与“高频操作”信号与建议面板匹配。
HISTORY_NOISE_ACTIONS = {"open_page", "undo"}


def history_summary(state, page=None):
    records = state.get("interaction_history", [])[-80:]
    if page:
        records = [item for item in records if item.get("page") == page]
    action_counts, modality_counts = {}, {}
    for record in records:
        action_counts[record.get("action", "unknown")] = action_counts.get(record.get("action", "unknown"), 0) + 1
        modality_counts[record.get("modality", "unknown")] = modality_counts.get(record.get("modality", "unknown"), 0) + 1
    ranked = sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "record_count": len(records),
        "frequent_actions": [key for key, _ in ranked if key not in HISTORY_NOISE_ACTIONS][:3],
        "action_counts": action_counts,
        "modality_counts": modality_counts,
    }


def push_undo(state, descriptor):
    """只登记可逆的端侧状态改动；消息模拟发送等操作不进入撤销栈。"""
    descriptor["created_at_ms"] = current_time_ms()
    state.setdefault("undo_stack", []).append(descriptor)
    del state["undo_stack"][:-UNDO_STACK_LIMIT]


def update_gesture_profile(state, event):
    if event.get("modality") != "head_gesture":
        return
    strength = event.get("payload", {}).get("motion_strength")
    if not isinstance(strength, (int, float)) or not 0 < strength < 1:
        return
    profile = state.setdefault("gesture_profile", {"head_min_strength": 0.0, "confirmed_samples": [], "undone_samples": []})
    profile.setdefault("confirmed_samples", []).append(round(float(strength), 4))
    del profile["confirmed_samples"][:-30]


def undo_last_nontext_operation(state):
    stack = state.get("undo_stack", [])
    if not stack:
        raise ValueError("没有可撤销的操作。")
    item = stack.pop()
    kind = item.get("kind")
    if kind == "clear_contact":
        state["selected_contact"] = None
        state["pending_message"] = None
        if item.get("modality") == "speech_text":
            message = "已撤销语音准备的消息，清除了待发送内容。"
        else:
            message = "已撤销通过视线或手势选中的联系人。"
    elif kind == "restore_track_preference":
        mode, track_id = item["mode"], item["track_id"]
        scores = state.setdefault("track_preferences", {}).setdefault(mode, {})
        if item["had_score"]:
            scores[track_id] = item["previous_score"]
        else:
            scores.pop(track_id, None)
        playlist = state.setdefault("mode_preference_playlists", {}).setdefault(mode, [])
        if item["was_in_playlist"] and track_id not in playlist:
            playlist.append(track_id)
        if not item["was_in_playlist"] and track_id in playlist:
            playlist.remove(track_id)
        message = "已撤销本次手势造成的音乐偏好调整。"
    elif kind == "toggle_playback":
        # 播放暂停/继续是前端演示播放状态，后端不掌握；撤销时返回 kind 由前端恢复播放。
        message = "已撤销上一次手势播放控制。"
    elif kind == "remove_authorized_sources":
        # 撤销本次备忘录授权：删除本次新增的授权副本，恢复上一版授权列表。
        previous = [dict(source) for source in item.get("previous_sources", [])]
        previous_stored = {source.get("stored_name") for source in previous}
        authorized_root = AUTHORIZED_DIR.resolve()
        for source in state.get("authorized_sources", []):
            name = source.get("stored_name")
            if name and name not in previous_stored:
                path = (AUTHORIZED_DIR / name).resolve()
                if path.parent == authorized_root and path.exists():
                    path.unlink()
        state["authorized_sources"] = previous
        valid_event_keys = {entry["event_key"] for entry in schedule_items(state)}
        state["completed_events"] = [key for key in state.get("completed_events", []) if key in valid_event_keys]
        message = "已撤销本次备忘录授权。"
    elif kind == "restore_event_completion":
        completed = set(state.get("completed_events", []))
        if item.get("was_completed"):
            completed.add(item["event_key"])
        else:
            completed.discard(item["event_key"])
        state["completed_events"] = sorted(completed)
        message = "已撤销日程完成状态修改。"
    elif kind == "remove_reminder":
        reminders = state.setdefault("reminders", [])
        state["reminders"] = [
            reminder for reminder in reminders
            if not (reminder.get("event_key") == item.get("event_key") and reminder.get("time") == item.get("time"))
        ]
        message = f"已撤销「{item.get('target_id')}」的 {item.get('time')} 提醒。"
    elif kind == "restore_reminder_offer":
        # 撤销“不用提醒”：恢复该事项的提醒建议，后续查询会再次给出。
        declined = state.setdefault("declined_reminder_offers", [])
        state["declined_reminder_offers"] = [key for key in declined if key != item.get("event_key")]
        message = "已撤销“不用提醒”，该事项的提醒建议将再次给出。"
    elif kind == "restore_reminder":
        # 撤销“取消提醒”：恢复被移除的提醒记录。
        state.setdefault("reminders", []).append(item["reminder"])
        message = f"已恢复 {item['reminder'].get('time')} 的提醒。"
    else:
        raise ValueError("该操作已不能安全撤销。")

    strength = item.get("motion_strength")
    if item.get("modality") == "head_gesture" and isinstance(strength, (int, float)):
        profile = state.setdefault("gesture_profile", {"head_min_strength": 0.0, "confirmed_samples": [], "undone_samples": []})
        profile.setdefault("undone_samples", []).append(round(float(strength), 4))
        del profile["undone_samples"][:-20]
        # 用户撤销说明该幅度可能只是自然晃动：后续需要比它略大的动作才会触发。
        profile["head_min_strength"] = round(min(0.16, max(float(profile.get("head_min_strength", 0)), float(strength) + 0.012)), 4)
    append_interaction_history(state, page=item.get("page", "unknown"), action="undo", modality="ui", target_id=item.get("target_id"), outcome="applied")
    result = {"message": message, "kind": kind, "gesture_profile": state.get("gesture_profile", {})}
    if kind == "remove_authorized_sources":
        result["authorized_sources"] = state["authorized_sources"]
    return result


def page_action_catalog(page):
    # match_actions：把真实历史动作名映射到建议目录 ID，供“基于历史的高频操作”回退与模型提示使用。
    catalogs = {
        "message": [
            {
                "id": "focus_contacts", "label": "查看常用联系人",
                "match_actions": ["select_contact", "send_message"],
            },
            {
                "id": "prepare_message", "label": "准备一条消息",
                "match_actions": ["select_contact", "send_message", "confirm_send", "cancel_message", "confirm", "cancel"],
            },
        ],
        "music": [
            {
                "id": "start_focus", "label": "开启专注模式",
                "match_actions": ["start_mode"],
            },
            {
                "id": "resume_music", "label": "播放热门推荐",
                "match_actions": ["like_track", "dislike_track", "next_track", "toggle_playback", "complete_track", "advance_track"],
            },
        ],
        "memo": [
            {
                "id": "query_schedule", "label": "查看已授权日程",
                "match_actions": ["query_schedule", "toggle_event_completion", "authorize_memo", "revoke_memo", "create_reminder", "decline_reminder"],
            },
            {
                "id": "query_today", "label": "查询今天安排",
                "match_actions": ["query_schedule"],
            },
        ],
    }
    return catalogs.get(page, [])


def page_suggestions(state, page):
    allowed_actions = page_action_catalog(page)
    if not allowed_actions:
        raise ValueError("不支持该页面建议。")
    profile = next(
        (item for item in read_json(DATA_DIR / "profiles_demo.json") if item["id"] == state.get("active_profile_id")),
        {},
    )
    summary = history_summary(state, page)
    action_counts = summary.get("action_counts", {})
    # 规则侧先按“历史动作 → 建议目录”的映射累计分排序，作为模型提示与回退的共同依据，
    # 避免历史动作名（select_contact 等）与目录 ID（prepare_message 等）直接比较永远失配。
    ranked = sorted(
        allowed_actions,
        key=lambda item: (
            -sum(action_counts.get(action, 0) for action in item.get("match_actions", [])),
            allowed_actions.index(item),
        ),
    )
    hint_ids = [
        item["id"] for item in ranked
        if any(action_counts.get(action, 0) > 0 for action in item.get("match_actions", []))
    ][:2]
    proposed = suggest_local_actions(
        scene=page, profile=profile, history=summary, allowed_actions=allowed_actions, hint_ids=hint_ids,
    ) if LOCAL_LLM_ENABLED else None
    action_map = {item["id"]: item for item in allowed_actions}
    chosen_ids = proposed.get("actions", []) if proposed else []
    if not chosen_ids:
        # 模型冷启动/超时时不阻塞页面：使用同一份本地历史的保守候选。
        chosen_ids = hint_ids or [item["id"] for item in allowed_actions[:2]]
    return {
        "message": (proposed or {}).get("response") or "根据你的本地使用习惯，可能想进行这些操作：",
        "actions": [action_map[item_id] for item_id in chosen_ids if item_id in action_map],
        "llm_used": bool(proposed),
    }


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
    if modality not in {"gaze", "screen_context", "speech_text", "head_gesture", "hand_gesture"}:
        raise ValueError("modality 仅支持 gaze、screen_context、speech_text、head_gesture 或 hand_gesture。")
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
    if modality in {"head_gesture", "hand_gesture"}:
        if payload.get("page") not in {"message", "music", "memo"} or payload.get("decision") not in {"confirm", "reject", "toggle_playback", "skip_track"}:
            raise ValueError("视觉手势事件必须包含支持的页面和决策。")

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


def find_contact_by_name(name):
    return next(
        (item for item in read_json(DATA_DIR / "contacts.json") if item.get("name") == str(name).strip()),
        None,
    )


def parse_simulated_speech(text):
    """返回 (intent, content, spoken_contact_name)；spoken_contact_name 可为空。

    intent 覆盖消息、音乐、日程三类语音指令；content 用于发送正文或曲风名。
    """
    normalized = re.sub(r"\s+", "", text)
    if normalized in {"取消当前歌曲选择", "取消歌曲选择", "取消音乐选择", "不选这首", "不操作这首"}:
        return "cancel_music_selection", "", ""
    if normalized in {"下一首", "切下一首", "换一首", "切歌", "换首歌", "跳过这首", "下一曲", "切下一曲"}:
        return "next_track", "", ""
    if normalized in {"喜欢这首", "这首歌好听", "好听", "好听的", "点赞", "收藏这首"}:
        return "like_track", "", ""
    if normalized in {"这首不喜欢", "这首歌不喜欢", "不喜欢这首", "不喜欢这首歌", "不喜欢这个", "这个不喜欢", "不喜欢", "不好听", "难听"}:
        return "dislike_track", "", ""
    if normalized in {"不", "不是", "不用", "暂不", "取消", "取消发送", "不要", "不要发送", "不发送"}:
        return "cancel", "", ""
    if normalized in {"是", "是的", "确认", "确认发送", "发送", "好的", "好", "确认发送消息"}:
        return "confirm", "", ""
    # 显式否定优先于发送指令：不要/别/不 + 给X + 发消息 → 取消（安全原则）
    if re.search(r"(?:不要|别|不)(?:给他|给她|给它|给[\u4e00-\u9fa5A-Za-z0-9]{1,6})?(?:发消息|发送消息|发信息|发送信息)", normalized):
        return "cancel", "", ""
    # 日程：提醒语音
    if normalized in {"不要提醒", "不用提醒", "不提醒", "不需要提醒"}:
        return "decline_reminder", "", ""
    if "以后" in normalized and "不要提醒" in normalized:
        return "update_reminder_preference", "", ""
    if "修改" in normalized:
        return "request_edit_memo", "", ""
    # 语音设置/取消提醒：给X设置提醒 / 提醒我X / 取消X提醒
    match = re.search(r"(?:给|为|帮)([\u4e00-\u9fa5A-Za-z0-9]{1,10})(?:设置|定个|添加)(?:一个)?提醒", normalized)
    if match:
        return "set_reminder", match.group(1), ""
    match = re.search(r"提醒我([\u4e00-\u9fa5A-Za-z0-9]{1,10})", normalized)
    if match:
        return "set_reminder", match.group(1), ""
    match = re.search(r"(?:取消|删除|去掉)([\u4e00-\u9fa5A-Za-z0-9]{1,10})提醒", normalized)
    if match:
        return "unset_reminder", match.group(1), ""
    # 日程：优先级与日期查询
    if "优先" in normalized or "重要" in normalized:
        return "query_priority", "", ""
    if "安排" in normalized or "日程" in normalized:
        if "太赶" in normalized or "压力" in normalized or "来得及" in normalized:
            return "analyze_schedule_pressure", "", ""
        if "后天" in normalized:
            return "query_date_plan", "", ""
        if "明天" in normalized:
            return "query_schedule_tomorrow", "", ""
        if "今天" in normalized:
            return "query_schedule_today", "", ""
        if any(word in normalized for word in ("全部", "所有", "最近")):
            return "query_schedule_all", "", ""
    # 音乐：喜欢 / 不喜欢
    if normalized in {"我喜欢这首", "我喜欢这首歌", "喜欢这首", "喜欢这首歌", "喜欢这个", "喜欢"}:
        return "like_track", "", ""
    if normalized in {"这首不喜欢", "这首歌不喜欢", "不喜欢这首", "不喜欢这首歌", "不喜欢这个", "这个不喜欢", "不喜欢"}:
        return "dislike_track", "", ""
    # 音乐：模式启动与播放
    if normalized in {"我准备学习", "开始专注", "开始学习", "进入专注模式", "我要学习", "学习模式", "专注模式"}:
        return "start_focus", "", ""
    if normalized == "播放学习音乐":
        return "offer_start_focus", "", ""
    if any(word in normalized for word in ("停止模式", "停止播放", "停止音乐", "结束模式")):
        return "stop_mode", "", ""
    # “播放下一首/下一曲”是切歌，不是播放一首叫“下一首”的歌。
    if "下一" in normalized and ("播放" in normalized or "切" in normalized or "换" in normalized or normalized in {"下一首", "下一曲", "下一首歌"}):
        return "next_track", "", ""
    if normalized in {"播放这个", "播放这首", "播放"}:
        return "play_music", "", ""
    match = re.search(r"播放([\u4e00-\u9fa5A-Za-z0-9-]{1,12})", normalized)
    if match:
        return "play_music", match.group(1), ""
    match = re.search(r"(?:来首|放首|放点|来点)([\u4e00-\u9fa5A-Za-z0-9-]{1,12})", normalized)
    if match:
        return "play_music", match.group(1), ""
    # “发消息给X，内容”：先于“给X发消息”处理。
    match = re.search(r"发消息给([\u4e00-\u9fa5A-Za-z0-9]{1,6})[，,、：:]?(.+)", normalized)
    if match:
        content = re.sub(r"^(?:让他|让她|让它|告诉他|告诉她|跟他说|跟她说|给他说|说|请|帮我|麻烦)", "", match.group(2).strip())
        return "send_message", content, match.group(1)
    # “给X发消息，正文”：提取联系人名（“他/她/它”不是联系人名，由视线/手动选择兜底）。
    # 正文去掉“让他/告诉她/跟他说/说”等祈使性插入语（例：“让他回来吃饭”→“回来吃饭”）。
    match = re.search(r"给([\u4e00-\u9fa5A-Za-z0-9]{1,6})(?:发消息|发送消息|发信息|发送信息)[，,、：:]?(.+)", normalized)
    if match:
        content = re.sub(r"^(?:让他|让她|让它|告诉他|告诉她|跟他说|跟她说|给他说|说|请|帮我|麻烦)", "", match.group(2).strip())
        return "send_message", content, match.group(1)
    match = re.search(r"(?:给他|给她|给它)(?:发消息|发送消息|发信息|发送信息)[，,、：:]?(.+)", normalized)
    if match and match.group(1).strip():
        content = re.sub(r"^(?:让他|让她|让它|告诉他|告诉她|跟他说|跟她说|给他说|说|请|帮我|麻烦)", "", match.group(1).strip())
        return "send_message", content, ""
    if any(word in normalized for word in ("发消息", "发送消息", "发信息", "发送信息", "发个消息")):
        return "send_message", "", ""
    # 模糊关键词兜底：容忍语音识别的近音/漏字（如“所以”≈“所有”），贴近系统已有操作。
    if any(word in normalized for word in ("日程", "安排")):
        if "明天" in normalized:
            return "query_schedule_tomorrow", "", ""
        if "后天" in normalized:
            return "query_date_plan", "", ""
        if "今天" in normalized:
            return "query_schedule_today", "", ""
        return "query_schedule_all", "", ""
    if "明天" in normalized:
        return "query_schedule_tomorrow", "", ""
    if "今天" in normalized:
        return "query_schedule_today", "", ""
    if any(word in normalized for word in ("下一", "切歌", "换歌")):
        return "next_track", "", ""
    if any(word in normalized for word in ("不喜欢", "难听", "不好听", "换一首")):
        return "dislike_track", "", ""
    if any(word in normalized for word in ("喜欢", "点赞", "收藏", "好歌")):
        return "like_track", "", ""
    if any(word in normalized for word in ("专注", "学习", "自习")):
        return "start_focus", "", ""
    if any(word in normalized for word in ("发消息", "发送", "发信息", "发个消息")):
        return "send_message", "", ""
    return "unknown", "", ""


def normalize_genre(value):
    """把语音/注视目标中的曲风表述归一化到曲库 genre 值；无法识别时返回原样小写。"""
    if not value:
        return None
    cleaned = re.sub(r"[\s\-_]", "", str(value)).lower()
    if cleaned.startswith("genre"):
        cleaned = cleaned[5:]
    aliases = {
        "lofi": "lofi", "轻音乐": "light_music", "纯音乐": "pure_music",
        "古典": "classical", "流行": "pop", "爵士": "jazz", "摇滚": "rock", "电子": "electronic",
    }
    return aliases.get(cleaned, cleaned)


def visible_target_ids(events, page, timestamp_ms):
    return {
        target.get("target_id")
        for context in events
        if context["modality"] == "screen_context"
        and context["payload"].get("page") == page
        and abs(context["timestamp_ms"] - timestamp_ms) <= 5_000
        for target in context["payload"].get("visible_targets", [])
    }


# 视线对齐时间窗（语音前 3 秒至语音后 1 秒），与 tests/scenarios.json 验收口径一致。
GAZE_ALIGNMENT_BEFORE_MS = 3_000
GAZE_ALIGNMENT_AFTER_MS = 1_000


def summarize_multimodal_alignment(events, page, anchor_timestamp_ms, speech_intent=None):
    """只保留结构化判断依据，不保存原始画面、音频或人脸关键点。"""
    contexts = [
        item for item in events
        if item["modality"] == "screen_context"
        and item["payload"].get("page") == page
        and abs(item["timestamp_ms"] - anchor_timestamp_ms) <= 5_000
    ]
    gazes = [
        item for item in events
        if item["modality"] == "gaze"
        and item["payload"].get("page") == page
        and anchor_timestamp_ms - GAZE_ALIGNMENT_BEFORE_MS <= item["timestamp_ms"] <= anchor_timestamp_ms + GAZE_ALIGNMENT_AFTER_MS
    ]
    decisions = [
        item for item in events
        if item["modality"] in {"head_gesture", "hand_gesture"}
        and item["payload"].get("page") == page
        and anchor_timestamp_ms - 2_000 <= item["timestamp_ms"] <= anchor_timestamp_ms + 1_000
    ]
    best_gaze = max(gazes, key=lambda item: (item["confidence"], item["payload"].get("dwell_ms", 0)), default=None)
    decision_values = [item["payload"].get("decision") for item in decisions]
    conflict = speech_intent == "cancel" and "confirm" in decision_values
    summary = {
        "page": page,
        "anchor_timestamp_ms": anchor_timestamp_ms,
        "windows_ms": {"screen_context": 5_000, "gaze_before": GAZE_ALIGNMENT_BEFORE_MS, "decision_before": 2_000},
        "modalities": ["speech_text"] + (["screen_context"] if contexts else []) + (["gaze"] if best_gaze else []) + sorted({item["modality"] for item in decisions}),
        "screen_context_available": bool(contexts),
        "gaze_target_id": best_gaze["payload"].get("target_id") if best_gaze else None,
        "gaze_confidence": best_gaze["confidence"] if best_gaze else None,
        "decision_values": decision_values,
        "conflict": "explicit_cancel_overrides_visual_confirm" if conflict else None,
    }
    if conflict:
        summary["summary"] = "检测到语音取消与视觉确认冲突，已按安全规则优先执行明确取消。"
    elif best_gaze and contexts:
        summary["summary"] = "已将文本指令、稳定注视和当前页面上下文按时间窗口关联。"
    elif contexts:
        summary["summary"] = "已将文本指令与当前页面上下文按时间窗口关联。"
    else:
        summary["summary"] = "本轮以文本指令为主；未找到同窗口页面上下文。"
    return summary


def understand_multimodal_command_inner(state, speech_timestamp_ms, preferred_contact_id=None, current_track_id=None):
    events = recent_multimodal_events()
    speech_events = [
        item for item in events
        if item["modality"] == "speech_text" and abs(item["timestamp_ms"] - speech_timestamp_ms) <= 1_000
    ]
    if not speech_events:
        raise ValueError("未找到对应的模拟语音事件，请重新提交。")
    speech = min(speech_events, key=lambda item: abs(item["timestamp_ms"] - speech_timestamp_ms))
    intent, content, spoken_contact_name = parse_simulated_speech(str(speech["payload"].get("text", "")))
    # 语音指令必须与当前页面匹配：日程页说“播放下一首”等音乐/消息指令会被忽略，不触发任何效果。
    page = speech["payload"].get("page", "message")
    page_intents = {
        "message": {"send_message", "confirm", "cancel"},
        "music": {"next_track", "cancel_music_selection", "like_track", "dislike_track",
                  "start_focus", "play_music", "offer_start_focus", "stop_mode", "confirm", "cancel"},
        "memo": {"query_schedule_today", "query_schedule_tomorrow", "query_schedule_all",
                 "query_priority", "query_date_plan", "analyze_schedule_pressure",
                 "decline_reminder", "update_reminder_preference", "request_edit_memo",
                 "set_reminder", "unset_reminder", "confirm", "cancel"},
    }
    if intent != "unknown" and intent not in page_intents.get(page, set()):
        page_label = {"message": "消息", "music": "音乐", "memo": "日程"}.get(page, page)
        return {
            "message": f"当前在{page_label}页面，暂不支持“{speech['payload'].get('text', '')}”这个操作。",
            "intent": intent,
            "ignored": True,
            "explanation": ["语音指令与当前页面不匹配，已忽略，未执行任何操作。"],
        }
    # 日程页语音设置/取消提醒：按事项标题解析并执行（与按钮等价）。
    if intent in {"set_reminder", "unset_reminder"} and not state["authorized_sources"]:
        return {
            "message": "请先授权备忘录文件，再设置提醒。",
            "intent": intent, "needs_clarification": True,
            "explanation": ["尚未授权任何备忘录。"],
        }
    if intent in {"set_reminder", "unset_reminder"}:
        candidates = [item for item in schedule_items(state)
                      if item.get("title") == content or (content and content in item.get("title", ""))]
        if not candidates:
            return {
                "message": f"未找到事项“{content}”，请确认名称后再试。",
                "intent": intent, "needs_clarification": True,
                "explanation": ["已授权事项中没有匹配该名称。"],
            }
        event_key = candidates[0]["event_key"]
        if intent == "set_reminder":
            message, _ = create_reminder_for_event(state, event_key)
        else:
            message = remove_reminder_for_event(state, event_key)
        save_state(state)
        return {"message": message, "intent": intent, "explanation": [f"已按语音指令处理事项「{candidates[0]['title']}」。"]}

    if intent == "cancel_music_selection":
        return {"message": "已取消当前歌曲选择；你可以重新注视并确认另一首歌曲。", "intent": intent, "explanation": [f"识别到音乐取消指令：{speech['payload']['text']}"]}
    if intent == "next_track":
        return {"message": "已识别切换下一首指令。", "intent": intent, "explanation": [f"识别到音乐指令：{speech['payload']['text']}"]}
    if intent in {"like_track", "dislike_track", "start_focus", "play_music", "offer_start_focus"}:
        return understand_music_command(state, intent, content, speech, events, current_track_id)

    if intent in {"confirm", "cancel"}:
        pending = state.get("pending_message")
        if intent == "confirm" and not pending:
            return {"message": "当前没有待确认的消息操作。", "intent": intent, "explanation": ["未检测到待确认任务。"]}
        if intent == "cancel" and not pending:
            state["pending_message"] = None
            state["selected_contact"] = None
            save_state(state)
            return {"message": "已取消当前联系人选择，可以重新选择联系人。", "intent": intent, "clear_message_form": True, "explanation": [f"识别到取消指令：{speech['payload']['text']}", "已清除当前联系人选择。"]}
        if intent == "confirm":
            state["pending_message"] = None
            state["selected_contact"] = None
            save_state(state)
            return {
                "message": f"已通过本地模拟消息应用向{pending['contact']}发送：‘{pending['content']}’。",
                "intent": intent,
                "simulated_send": pending,
                "clear_message_form": True,
                "explanation": [
                    f"识别到确认词：{speech['payload']['text']}",
                    "已完成本地模拟发送，未调用或读取任何第三方应用。",
                ],
            }
        state["pending_message"] = None
        state["selected_contact"] = None
        save_state(state)
        return {"message": "已取消本次发送，并清除当前联系人选择。", "intent": intent, "clear_message_form": True, "explanation": [f"识别到确认词：{speech['payload']['text']}"]}

    if intent in {
        "query_schedule_today", "query_schedule_tomorrow", "query_schedule_all",
        "query_priority", "query_date_plan", "analyze_schedule_pressure",
        "decline_reminder", "update_reminder_preference", "request_edit_memo",
    }:
        result = understand_schedule_query(state, intent, speech)
        result["intent"] = intent
        result["explanation"] = [f"识别到模拟语音：{speech['payload']['text']}", "仅读取用户已授权的本地备忘录，不修改日程完成状态。"]
        return result

    if intent != "send_message":
        page_hints = {
            "message": "给他发消息，内容 / 确认 / 取消",
            "music": "下一首 / 我喜欢这首 / 我准备学习 / 播放Lo-fi / 停止模式",
            "memo": "我明天有什么安排 / 查看所有日程 / 给组会设置提醒 / 不要提醒",
        }
        hint = page_hints.get(page, page_hints["message"])
        return {
            "message": f"暂未理解“{speech['payload'].get('text', '')}”。当前页面可尝试：{hint}",
            "intent": "unknown",
            "explanation": ["未匹配到当前页面支持的指令，请换一种说法后重试。"],
        }
    if not content:
        return {"message": "请补充需要发送的消息内容。", "intent": intent, "needs_clarification": True, "explanation": ["识别到发送消息意图，但缺少消息正文。"]}

    manual_contact = find_contact(preferred_contact_id) if preferred_contact_id else None
    spoken_contact = find_contact_by_name(spoken_contact_name) if spoken_contact_name else None
    gaze_events = [
        item for item in events
        if item["modality"] == "gaze"
        and item["payload"].get("page") == "message"
        and item["payload"].get("target_type") == "contact"
        and speech_timestamp_ms - GAZE_ALIGNMENT_BEFORE_MS <= item["timestamp_ms"] <= speech_timestamp_ms + GAZE_ALIGNMENT_AFTER_MS
    ]
    gaze = max(gaze_events, key=lambda item: (item["confidence"], item["payload"].get("dwell_ms", 0)), default=None)
    # 安全原则：置信度不足的视线不作为对象依据，改走澄清，不直接采用。
    low_gaze_confidence = gaze is not None and gaze["confidence"] < 0.55
    if low_gaze_confidence:
        gaze = None
    if spoken_contact:
        contact = spoken_contact
        target_explanation = f"本轮优先使用语音中明确说出的联系人：{contact['name']}。"
    elif manual_contact:
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
            explanation = ["未找到有效视线事件，也没有手动选中的联系人。"]
            if low_gaze_confidence:
                explanation.append("检测到视线事件但置信度不足，按安全原则不直接采用。")
            return {"message": "请先注视确认或手动点击需要联系的联系人，再提交模拟语音。", "intent": intent, "needs_clarification": True, "explanation": explanation}
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


SPEECH_HISTORY_ACTIONS = {
    "send_message": "send_message",
    "confirm": "confirm",
    "cancel": "cancel",
    "query_schedule_today": "query_schedule",
    "query_schedule_tomorrow": "query_schedule",
    "query_schedule_all": "query_schedule",
    "next_track": "next_track",
    "cancel_music_selection": "cancel_music_selection",
    "like_track": "like_track",
    "dislike_track": "dislike_track",
    "start_focus": "start_mode",
    "stop_mode": "stop_mode",
    "play_music": "play_music",
    "query_priority": "query_schedule",
    "query_date_plan": "query_schedule",
    "analyze_schedule_pressure": "query_schedule",
    "decline_reminder": "decline_reminder",
    "update_reminder_preference": "update_reminder_preference",
    "request_edit_memo": "request_edit_memo",
}


def record_speech_operation(state, result, intent, page):
    """语音（模拟语音）操作同样进入本地历史；可逆的「准备发送消息」登记撤销。"""
    action = SPEECH_HISTORY_ACTIONS.get(intent)
    if not action:
        return
    target_id = result.get("pending", {}).get("contact_id") if intent == "send_message" else None
    append_interaction_history(
        state, page=page, action=action, modality="speech_text", target_id=target_id,
    )
    if intent == "send_message" and target_id:
        push_undo(state, {
            "kind": "clear_contact", "page": "message",
            "target_id": target_id, "modality": "speech_text",
        })


def understand_multimodal_command(state, speech_timestamp_ms, preferred_contact_id=None, current_track_id=None):
    """所有文本指令共用同一套时间对齐摘要，再进入对应场景的意图处理。"""
    events = recent_multimodal_events()
    speech_events = [
        item for item in events
        if item["modality"] == "speech_text" and abs(item["timestamp_ms"] - speech_timestamp_ms) <= 1_000
    ]
    if not speech_events:
        raise ValueError("未找到对应的模拟语音事件，请重新提交。")
    speech = min(speech_events, key=lambda item: abs(item["timestamp_ms"] - speech_timestamp_ms))
    raw_text = str(speech["payload"].get("text", ""))
    effective_text = raw_text
    normalized_note = None
    # 麦克风语音（source=mic）先经本地大模型整理成规范指令，规则再做最终裁决；
    # 模型不可用/超时/输出不合规时回退原文，保证交互不中断。
    if speech["payload"].get("source") == "mic" and LOCAL_LLM_ENABLED:
        normalized = normalize_speech_command(raw_text, speech["payload"].get("page", "unknown"))
        if normalized and normalized != raw_text:
            effective_text = normalized
            normalized_note = f"本地大模型已将语音整理为指令：{normalized}"
    intent, _, _ = parse_simulated_speech(effective_text)
    result = understand_multimodal_command_inner(state, speech_timestamp_ms, preferred_contact_id, current_track_id)
    if normalized_note:
        result.setdefault("explanation", []).append(normalized_note)
    fusion = summarize_multimodal_alignment(events, speech["payload"].get("page", "unknown"), speech_timestamp_ms, intent)
    result["fusion"] = fusion
    result.setdefault("explanation", []).append(f"多模态综合判断：{fusion['summary']}")
    # 大模型不参与意图、对象或执行决策：只解释已经通过安全规则的结构化结论。
    # 模型不可用、冷启动或超时时，立即保留规则模板结果，避免影响 <5 秒目标。
    profile = next(
        (item for item in read_json(DATA_DIR / "profiles_demo.json") if item["id"] == state.get("active_profile_id")),
        {},
    )
    llm_result = None
    if LOCAL_LLM_ENABLED:
        llm_result = enhance_local_response(
            scene=speech["payload"].get("page", "unknown"),
            speech_text=speech["payload"].get("text", ""),
            rule_result=result,
            fusion=fusion,
            profile=profile,
            history=history_summary(state, speech["payload"].get("page", "unknown")),
        )
    result["llm"] = {"used": bool(llm_result), "model": "qwen2.5:3b-instruct"}
    if llm_result:
        if llm_result.get("conflict_explanation"):
            result["explanation"].append(f"本地大模型冲突解释：{llm_result['conflict_explanation']}")
        if llm_result.get("personalization_reason"):
            result["explanation"].append(f"本地大模型个性化依据：{llm_result['personalization_reason']}")
        # 操作性反馈（发送、取消、切歌）必须保持规则文案；只让模型润色查询类答复。
        # 这样模型即使措辞失误，也不会让用户误解实际执行状态。
        if llm_result.get("response") and result.get("intent") in {
            "query_schedule_today", "query_schedule_tomorrow", "query_schedule_all", "unknown",
        }:
            result["message"] = llm_result["response"]
    else:
        result["explanation"].append("本地大模型本轮未在时限内返回，已使用安全规则模板。")
    # 语音（模拟语音）操作同样进入本地历史；准备发送消息等可逆操作同时登记撤销。
    record_speech_operation(state, result, intent, speech["payload"].get("page", "unknown"))
    save_state(state)
    return result


DEMO_COMMON_TRACKS = {
    "general": ["track_031", "track_024", "track_010", "track_026", "track_012"],
    # 冷启动第一首用轻音乐（贴合“专注”场景的演示预期），再穿插其他曲风。
    "focus": ["track_018", "track_003", "track_010", "track_015", "track_022"],
    "driving": ["track_012", "track_028", "track_030", "track_007", "track_008"],
    "entertainment": ["track_024", "track_026", "track_013", "track_025", "track_027"],
}


def tracks_by_id():
    return {track["id"]: track for track in read_json(DATA_DIR / "music_library.json")}


def mode_preference_playlist(state, mode):
    track_index = tracks_by_id()
    ids = state["mode_preference_playlists"].get(mode, [])
    return [track_index[track_id] for track_id in ids if track_id in track_index]


def record_track_preference(state, mode, track_id, delta, add_to_playlist=False):
    scores = state["track_preferences"].setdefault(mode, {})
    score = scores.get(track_id, 0) + delta
    scores[track_id] = score
    playlist = state["mode_preference_playlists"].setdefault(mode, [])
    if add_to_playlist and track_id not in playlist:
        playlist.append(track_id)
    if score < 0 and track_id in playlist:
        playlist.remove(track_id)
    return score


def recommend_track(state, mode, exclude_id=None):
    track_index = tracks_by_id()
    candidates = [track for track in track_index.values() if (mode == "general" or mode in track.get("moods", [])) and track["id"] != exclude_id]
    if not candidates:
        candidates = [track for track in track_index.values() if track["id"] != exclude_id] or list(track_index.values())

    common_ids = [track_id for track_id in DEMO_COMMON_TRACKS[mode] if track_id in track_index and track_id != exclude_id]
    scores = state["track_preferences"].get(mode, {})
    preferred_ids = [track["id"] for track in mode_preference_playlist(state, mode) if track["id"] != exclude_id and scores.get(track["id"], 0) >= 0]
    turn = state["recommendation_turns"].get(mode, 0)
    state["recommendation_turns"][mode] = turn + 1

    # 每三次至少推荐一次该模式的本地大众常听曲目，避免推荐只困在历史偏好里。
    if common_ids and (not preferred_ids or turn % 3 == 2):
        return track_index[common_ids[turn % len(common_ids)]], "该模式下的平台大众常听"
    if preferred_ids:
        ranked_ids = sorted(preferred_ids, key=lambda track_id: scores.get(track_id, 0), reverse=True)
        return track_index[ranked_ids[0]], "你的本模式偏好歌单"
    if common_ids:
        return track_index[common_ids[0]], "该模式下的平台大众常听"
    return candidates[0], "与当前模式匹配"


def parse_memo_file(path: Path):
    """读取用户已授权的 TXT、MD 或 JSON 备忘录，统一为项目内部字段。"""
    if path.suffix.lower() == ".json":
        entries = read_json(path)
        if not isinstance(entries, list):
            raise ValueError("JSON 备忘录应为事项数组。")
        return entries

    entries = []
    for line_number, raw_line in enumerate(read_memo_text(path).splitlines(), 1):
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
        return event_time < demo_now()
    except (KeyError, TypeError, ValueError):
        return False


def schedule_items(state):
    completed = set(state.get("completed_events", []))
    reminder_by_key = {reminder.get("event_key"): reminder.get("time") for reminder in state.get("reminders", [])}
    items = []
    for memo in authorized_memos(state):
        item = dict(memo)
        item["event_key"] = schedule_event_key(item)
        item["is_completed"] = item["event_key"] in completed
        item["is_past"] = is_past_event(item)
        item["reminder_time"] = reminder_by_key.get(item["event_key"])
        items.append(item)
    return sorted(items, key=lambda item: (item.get("date", "9999-99-99"), item.get("time", "99:99")))


def build_reminder_offers(state, items, limit=3):
    """返回最多 limit 个“未过期、未拒绝、未完成”的高优先级事项提醒建议。"""
    profile = next(
        (item for item in read_json(DATA_DIR / "profiles_demo.json") if item["id"] == state.get("active_profile_id")),
        {},
    )
    lead_minutes = int(profile.get("schedule_reminder_minutes") or 50)
    existing = {(reminder.get("event_key"), reminder.get("time")) for reminder in state.get("reminders", [])}
    declined = set(state.get("declined_reminder_offers", []))
    offers = []
    for item in sorted(items, key=lambda entry: (entry.get("date", "9999-99-99"), entry.get("time", "99:99"))):
        if len(offers) >= limit:
            break
        if item.get("is_past") or item.get("is_completed"):
            continue
        if item["event_key"] in declined:
            continue
        if item.get("priority") != "high":
            continue
        try:
            event_time = datetime.fromisoformat(f"{item['date']} {item['time']}")
        except (KeyError, TypeError, ValueError):
            continue
        remind_time = (event_time - timedelta(minutes=lead_minutes)).strftime("%Y-%m-%d %H:%M")
        offers.append({
            "event_key": item["event_key"],
            "memo_id": item.get("id"),
            "title": item.get("title"),
            "event_time": f"{item['date']} {item['time']}",
            "remind_time": remind_time,
            "lead_minutes": lead_minutes,
            "due_now": event_time - timedelta(minutes=lead_minutes) <= demo_now(),
            "already_set": (item["event_key"], remind_time) in existing,
        })
    return offers


def build_reminder_offer(state, items):
    """向后兼容：返回第一个提醒建议；多个建议见 build_reminder_offers。"""
    offers = build_reminder_offers(state, items, limit=1)
    return offers[0] if offers else None


def create_reminder_for_event(state, event_key):
    """按事项 event_key 创建提醒（HTTP 动作与语音指令共用）；返回 (message, is_duplicate)。"""
    items_by_key = {item["event_key"]: item for item in schedule_items(state)}
    item = items_by_key.get(event_key)
    if not item:
        raise ValueError("该日程不在当前已授权文件中。")
    if item.get("is_past"):
        raise ValueError("该日程已过时间，无需提醒。")
    profile = next(
        (item for item in read_json(DATA_DIR / "profiles_demo.json") if item["id"] == state.get("active_profile_id")),
        {},
    )
    lead_minutes = int(profile.get("schedule_reminder_minutes") or 50)
    try:
        event_time = datetime.fromisoformat(f"{item['date']} {item['time']}")
    except (KeyError, TypeError, ValueError):
        raise ValueError("该日程时间格式无法识别。")
    remind_time = (event_time - timedelta(minutes=lead_minutes)).strftime("%Y-%m-%d %H:%M")
    reminders = state.setdefault("reminders", [])
    duplicate = any(
        reminder.get("event_key") == event_key and reminder.get("time") == remind_time
        for reminder in reminders
    )
    if duplicate:
        return f"已存在 {remind_time} 的「{item['title']}」提醒，未重复创建。", True
    reminders.append({"memo_id": item.get("id"), "event_key": event_key, "time": remind_time})
    append_interaction_history(state, page="memo", action="create_reminder", modality="ui", target_id=item.get("title"))
    push_undo(state, {
        "kind": "remove_reminder", "page": "memo",
        "target_id": item.get("title"), "event_key": event_key, "time": remind_time,
    })
    return f"已在本项目内创建 {remind_time} 的「{item['title']}」提醒。", False


def remove_reminder_for_event(state, event_key):
    """按事项 event_key 取消提醒（HTTP 动作与语音指令共用）；返回提示信息。"""
    reminders = state.setdefault("reminders", [])
    removed = [reminder for reminder in reminders if reminder.get("event_key") == event_key]
    if not removed:
        raise ValueError("该事项没有已设置的提醒。")
    state["reminders"] = [reminder for reminder in reminders if reminder.get("event_key") != event_key]
    append_interaction_history(state, page="memo", action="remove_reminder", modality="ui")
    push_undo(state, {"kind": "restore_reminder", "page": "memo", "reminder": removed[0]})
    return "已取消该事项的提醒。"


def schedule_query_result(state, target_date=None, title="全部日程"):
    if not state["authorized_sources"]:
        raise ValueError("请先授权备忘录文件。")
    items = schedule_items(state)
    if target_date:
        items = [item for item in items if item.get("date") == target_date]
    result = {
        "message": f"已整理{title}。",
        "title": title,
        "items": items,
        "total": len(items),
        "completed": sum(item["is_completed"] for item in items),
        "past": sum(item["is_past"] for item in items),
        "reminder_offer": build_reminder_offer(state, items),
        "reminder_offers": build_reminder_offers(state, items, limit=3),
    }
    if target_date:
        result["date"] = target_date
    return result


def understand_schedule_query(state, intent, speech):
    reference = demo_now()
    if intent == "query_schedule_today":
        return schedule_query_result(state, reference.strftime("%Y-%m-%d"), f"今天（{reference:%Y-%m-%d}）日程")
    if intent == "query_schedule_tomorrow":
        date = (reference.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        return schedule_query_result(state, date.strftime("%Y-%m-%d"), f"明天（{date:%Y-%m-%d}）日程")
    if intent == "query_date_plan":  # 后天
        date = (reference.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2))
        return schedule_query_result(state, date.strftime("%Y-%m-%d"), f"后天（{date:%Y-%m-%d}）日程")
    if intent == "query_priority":
        tomorrow = (reference.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")
        items = [
            item for item in schedule_items(state)
            if item.get("date") == tomorrow_str and item.get("priority") == "high"
        ]
        return {
            "message": "已整理明天的高优先级事项。",
            "title": "明天高优先级事项",
            "items": items,
            "total": len(items),
            "memo_id": items[0]["id"] if items else None,
        }
    if intent == "analyze_schedule_pressure":
        tomorrow = (reference.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")
        highs = [
            item for item in schedule_items(state)
            if item.get("date") == tomorrow_str and item.get("priority") == "high" and not item.get("is_completed")
        ]
        pair = None
        for index in range(len(highs) - 1):
            try:
                first = datetime.fromisoformat(f"{highs[index]['date']} {highs[index]['time']}")
                second = datetime.fromisoformat(f"{highs[index + 1]['date']} {highs[index + 1]['time']}")
                gap_hours = (second - first).total_seconds() / 3600
            except (KeyError, TypeError, ValueError):
                continue
            if 0 < gap_hours < 5:
                pair = (highs[index], highs[index + 1], gap_hours)
                break
        if pair:
            first_item, second_item, gap = pair
            message = (
                f"「{first_item['title']}」{first_item['time']} 到「{second_item['title']}」"
                f"{second_item['time']} 仅间隔 {gap:.0f} 小时，建议提前完成准备工作。"
            )
        else:
            message = "明天的安排时间充裕，没有明显的时间压力。"
        return {
            "message": message,
            "title": "明天日程压力分析",
            "items": highs,
            "total": len(highs),
            "suggestion": message,
            "should_suggest": pair is not None,
        }
    if intent == "decline_reminder":
        return {"message": "已取消本次提醒；不会因此修改长期偏好。", "intent": intent}
    if intent == "update_reminder_preference":
        return {
            "message": "确定以后同类事项都不要提醒吗？确认后同类事项将不再自动建议提醒。",
            "intent": intent, "needs_clarification": True,
        }
    if intent == "request_edit_memo":
        return {
            "message": "修改备忘录需要明确确认。请告诉我具体事项与新的时间，确认后我才会修改本地副本。",
            "intent": intent, "needs_clarification": True,
        }
    return schedule_query_result(state, title="全部已授权日程")


def understand_music_command(state, intent, content, speech, events, current_track_id=None):
    """音乐页语音指令：模式启动、播放、喜欢/不喜欢；规则决定执行，LLM 只做解释。"""
    if intent == "offer_start_focus":
        return {
            "message": "当前未启用专注模式。需要我为你开启专注模式吗？",
            "intent": intent, "needs_clarification": True,
            "explanation": ["识别到学习/专注诉求，但音乐模式尚未启用。"],
        }
    if intent == "stop_mode":
        clear_active_music_mode(state)
        save_state(state)
        return {
            "message": "已停止当前音乐模式。",
            "intent": intent,
            "explanation": ["识别到停止模式指令。"],
        }
    if intent == "start_focus":
        state["active_mode"] = "focus"
        state["focus_mode"] = True
        track, reason = recommend_track(state, "focus")
        save_state(state)
        return {
            "message": f"已为你开启专注模式，正在播放：{track['title']}。",
            "intent": intent, "mode": "focus", "track": track,
            "recommendation_reason": reason, "recommended_genre": track.get("genre"),
            "preference_playlist": mode_preference_playlist(state, "focus"),
        }
    if intent == "play_music":
        return _resolve_play_music(state, content, events, speech)
    if intent in {"like_track", "dislike_track"}:
        mode = state["active_mode"] or "general"
        if not current_track_id or current_track_id not in tracks_by_id():
            return {
                "message": "当前没有正在播放的歌曲，请先播放一首歌。",
                "intent": intent, "needs_clarification": True,
                "explanation": ["缺少当前播放曲目，无法记录偏好。"],
            }
        if intent == "like_track":
            adjustment = record_track_preference(state, mode, current_track_id, 3, add_to_playlist=True)
            message = "已加入当前模式偏好歌单，当前歌曲继续播放。"
        else:
            adjustment = record_track_preference(state, mode, current_track_id, -2)
            track, reason = recommend_track(state, mode, current_track_id)
            message = f"已降低这首歌的偏好值，正在播放：{track['title']}。"
        save_state(state)
        result = {"message": message, "intent": intent, "adjustment": adjustment,
                  "preference_playlist": mode_preference_playlist(state, mode)}
        if intent == "dislike_track":
            result.update({"track": track, "recommendation_reason": reason})
        return result
    return {"message": "暂未理解该音乐指令。", "intent": "unknown", "explanation": []}


def _resolve_play_music(state, content, events, speech):
    """“播放X/播放这个”：语音曲风优先，其次注视的曲风卡片；置信度不足时澄清。"""
    anchor = speech["timestamp_ms"]
    genre = normalize_genre(content) if content else None
    gaze = None
    if not genre:
        gazes = [
            item for item in events
            if item["modality"] == "gaze"
            and item["payload"].get("page") == "music"
            and anchor - GAZE_ALIGNMENT_BEFORE_MS <= item["timestamp_ms"] <= anchor + GAZE_ALIGNMENT_AFTER_MS
        ]
        gaze = max(gazes, key=lambda item: (item["confidence"], item["payload"].get("dwell_ms", 0)), default=None)
        if gaze and gaze["confidence"] < 0.55:
            return {
                "message": "注视置信度不足，请重新注视想播放的内容，或直接说出曲风。",
                "intent": "clarify_music_selection", "needs_clarification": True,
                "explanation": ["视线置信度不足，按安全原则不直接采用。"],
            }
        if gaze:
            genre = normalize_genre(gaze["payload"].get("target_id"))
    if not genre:
        return {
            "message": "请说出想播放的曲风，或注视对应的歌曲/曲风卡片。",
            "intent": "play_music", "needs_clarification": True,
            "explanation": ["缺少播放对象。"],
        }
    track_index = tracks_by_id()
    known_genres = {track.get("genre") for track in track_index.values()}
    if genre not in known_genres:
        return {
            "message": f"本地曲库中没有“{genre}”曲风，可尝试 Lo-fi、轻音乐、纯音乐、古典、流行、爵士、摇滚、电子。",
            "intent": "play_music", "needs_clarification": True,
            "explanation": [f"曲风 {genre} 不在本地曲库。"],
        }
    mode = state.get("active_mode") or "general"
    candidates = [
        track for track in track_index.values()
        if track.get("genre") == genre and (mode == "general" or mode in track.get("moods", []))
    ]
    if not candidates:
        candidates = [track for track in track_index.values() if track.get("genre") == genre]
    track = candidates[0]
    if mode == "general":
        state["active_mode"] = None
        state["focus_mode"] = False
    save_state(state)
    return {
        "message": f"正在播放：{track['title']}（{track.get('genre')}）。",
        "intent": "play_music", "mode": mode, "track": track,
        "recommendation_reason": f"曲风匹配：{track.get('genre')}",
        "recommended_genre": track.get("genre"),
        "preference_playlist": mode_preference_playlist(state, mode),
    }


class AssistantHandler(SimpleHTTPRequestHandler):
    # Python 在部分 Windows 环境中不会为 .mjs 注册 JavaScript 类型，
    # 浏览器会因此拒绝动态 import。显式声明本地视觉运行时所需的类型。
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "application/javascript",
        ".wasm": "application/wasm",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        # 演示开发阶段始终加载最新页面脚本，避免浏览器缓存旧操作名。
        self.send_header("Cache-Control", "no-store, max-age=0")
        # 跨源隔离：让浏览器端离线语音识别（onnxruntime-web 线程化 WASM）可用 SharedArrayBuffer。
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
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
            with STATE_LOCK:
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
                    "demo_reference_date": DEMO_REFERENCE_DATE,
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
            result, state = self.handle_action(payload)
            self.send_json({"ok": True, "result": result, "state": state})
        except Exception as error:
            self.send_json(
                {"ok": False, "error": f"本地处理失败：{error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def handle_action(self, payload):
        # 整个 load→改→save 在同一把锁内串行执行，避免并发请求互相覆盖状态。
        with STATE_LOCK:
            state = load_state()
            result = self._dispatch_action(payload, state)
        return result, state

    def _dispatch_action(self, payload, state):
        action = payload["action"]

        if action == "record_multimodal_event":
            result = record_multimodal_event(payload.get("event"))
            event = result["event"]
            modality = event["modality"]
            event_payload = event["payload"]
            # 长时历史只记录实际触发的操作；稳定注视、点头/摇头等感知中间态不再逐条写入，
            # 避免把摄像头每帧噪声呈现给用户。头部动作仍用于更新手势自适应画像。
            if modality == "head_gesture":
                update_gesture_profile(state, event)
                save_state(state)
                # 顺带返回最新手势画像，前端据此调整点头/摇头的检测下限（自适应阈值闭环）。
                result["gesture_profile"] = state.get("gesture_profile", {})
            elif modality == "hand_gesture" and event_payload.get("decision") == "toggle_playback":
                # 手势暂停/继续是唯一直接改变前端播放状态、不经过后端业务动作的手势，
                # 在此登记一条可查看、可撤销的操作记录。
                append_interaction_history(
                    state, page="music", action="toggle_playback",
                    modality="hand_gesture", outcome="applied",
                )
                push_undo(state, {"kind": "toggle_playback", "page": "music", "modality": "hand_gesture"})
                save_state(state)
            return result

        if action == "select_profile":
            switch_active_profile(state, str(payload.get("profile_id", "")))
            save_state(state)
            profile = next(item for item in read_json(DATA_DIR / "profiles_demo.json") if item["id"] == state["active_profile_id"])
            return {
                "message": f"已切换到{profile['display_name']}的本地画像；偏好、日程授权和交互状态彼此独立。",
                "profile": profile,
                "state": state,
            }

        if action == "get_recent_multimodal_events":
            events = recent_multimodal_events()
            return {"message": "已读取本地短时多模态事件。", "events": events}

        if action == "get_interaction_history":
            return {
                "message": "已读取本地交互历史摘要。",
                "records": list(reversed(state.get("interaction_history", [])[-30:])),
                "gesture_profile": state.get("gesture_profile", {}),
            }

        if action == "clear_interaction_history":
            state["interaction_history"] = []
            state["undo_stack"] = []
            state["gesture_profile"] = {"head_min_strength": 0.0, "confirmed_samples": [], "undone_samples": []}
            save_state(state)
            return {"message": "已清空本地交互历史、撤销记录和手势自适应信息。"}

        if action == "open_page":
            page = str(payload.get("page", ""))
            if page not in {"message", "music", "memo"}:
                raise ValueError("不支持该页面。")
            append_interaction_history(state, page=page, action="open_page", modality="ui")
            result = page_suggestions(state, page)
            save_state(state)
            return result

        if action == "undo_last_nontext_operation":
            result = undo_last_nontext_operation(state)
            save_state(state)
            return result

        if action == "understand_multimodal_command":
            timestamp_ms = payload.get("speech_timestamp_ms")
            if not isinstance(timestamp_ms, int):
                raise ValueError("speech_timestamp_ms 必须是语音事件的毫秒时间戳。")
            return understand_multimodal_command(
                state, timestamp_ms,
                preferred_contact_id=payload.get("preferred_contact_id"),
                current_track_id=payload.get("current_track_id"),
            )

        if action == "select_contact":
            contact_id = str(payload["contact_id"])
            if not find_contact(contact_id):
                raise ValueError("未找到该联系人。")
            modality = str(payload.get("input_modality", "ui"))
            state["selected_contact"] = contact_id
            append_interaction_history(state, page="message", action="select_contact", modality=modality, target_id=contact_id)
            if modality in {"gaze", "head_gesture", "hand_gesture"}:
                push_undo(state, {"kind": "clear_contact", "page": "message", "target_id": contact_id, "modality": modality})
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
            pending = state.get("pending_message")
            if not pending:
                raise ValueError("当前没有待确认的消息。")
            state["pending_message"] = None
            state["selected_contact"] = None
            append_interaction_history(state, page="message", action="confirm_send", modality="ui", target_id=pending["contact_id"])
            save_state(state)
            return {
                "message": f"已通过本地模拟消息应用向{pending['contact']}发送：‘{pending['content']}’。",
                "simulated_send": pending,
            }

        if action == "cancel_message":
            state["selected_contact"] = None
            state["pending_message"] = None
            append_interaction_history(state, page="message", action="cancel_message", modality="ui")
            save_state(state)
            return {"message": "已取消本次发送。"}

        if action == "start_mode":
            mode = payload["mode"]
            if mode not in {"focus", "driving", "entertainment"}:
                raise ValueError("不支持的音乐模式。")
            state["active_mode"] = mode
            state["focus_mode"] = mode == "focus"
            track, reason = recommend_track(state, mode)
            save_state(state)
            return {
                "message": f"已进入{payload['mode_label']}，已为你检索一首适合当前状态的歌曲。",
                "track": track,
                "recommendation_reason": reason,
                "preference_playlist": mode_preference_playlist(state, mode),
            }

        if action == "start_general_music":
            state["active_mode"] = "general"
            state["focus_mode"] = False
            track, reason = recommend_track(state, "general")
            save_state(state)
            return {
                "message": "未选择特定模式，已为你推荐近期热门歌曲。",
                "track": track,
                "recommendation_reason": "近期热门歌曲",
                "preference_playlist": mode_preference_playlist(state, "general"),
            }

        if action == "stop_mode":
            clear_active_music_mode(state)
            save_state(state)
            return {"message": "已停止当前音乐模式。当前歌曲仍会显示，但不再记录该模式偏好。"}

        if action == "stop_mode_silent":
            clear_active_music_mode(state)
            save_state(state)
            return {"message": "已结束音乐模式会话。"}

        if action == "like_track":
            mode = state["active_mode"] or "general"
            if not mode:
                raise ValueError("当前没有启用音乐模式，无法记录模式偏好。")
            track_id = payload["track_id"]
            if track_id not in tracks_by_id():
                raise ValueError("当前歌曲不在本地音乐库中。")
            modality = str(payload.get("input_modality", "ui"))
            scores = state.setdefault("track_preferences", {}).setdefault(mode, {})
            previous_score = scores.get(track_id, 0)
            had_score = track_id in scores
            was_in_playlist = track_id in state.setdefault("mode_preference_playlists", {}).setdefault(mode, [])
            adjustment = record_track_preference(state, mode, track_id, 3, add_to_playlist=True)
            append_interaction_history(state, page="music", action="like_track", modality=modality, target_id=track_id)
            if modality in {"head_gesture", "hand_gesture"}:
                push_undo(state, {
                    "kind": "restore_track_preference", "page": "music", "target_id": track_id,
                    "mode": mode, "track_id": track_id, "previous_score": previous_score,
                    "had_score": had_score, "was_in_playlist": was_in_playlist, "modality": modality,
                    "motion_strength": payload.get("motion_strength"),
                })
            save_state(state)
            label = "通用偏好歌单" if mode == "general" else "当前模式的偏好歌单"
            return {"message": f"已加入{label}，当前歌曲继续播放。", "adjustment": adjustment, "preference_playlist": mode_preference_playlist(state, mode)}

        if action == "complete_track":
            mode = state["active_mode"] or "general"
            if not mode:
                raise ValueError("当前没有启用音乐模式，无法记录模式偏好。")
            track_id = payload["track_id"]
            if track_id not in tracks_by_id():
                raise ValueError("当前歌曲不在本地音乐库中。")
            adjustment = record_track_preference(state, mode, track_id, 1, add_to_playlist=True)
            track, reason = recommend_track(state, mode, track_id)
            save_state(state)
            label = "通用偏好歌单" if mode == "general" else "当前模式的偏好歌单"
            return {"message": f"已检测到完整收听，并加入{label}。正在播放：{track['title']}。", "adjustment": adjustment, "track": track, "recommendation_reason": reason, "preference_playlist": mode_preference_playlist(state, mode)}

        if action == "advance_track":
            mode = state["active_mode"] or "general"
            if not mode:
                raise ValueError("当前没有启用音乐模式。")
            track_id = payload["current_track_id"]
            if track_id not in tracks_by_id():
                raise ValueError("当前歌曲不在本地音乐库中。")
            track, reason = recommend_track(state, mode, track_id)
            save_state(state)
            return {"message": f"正在播放：{track['title']}。", "track": track, "recommendation_reason": reason, "preference_playlist": mode_preference_playlist(state, mode)}

        if action == "next_track":
            mode = state["active_mode"] or "general"
            if not mode:
                raise ValueError("当前没有启用音乐模式。")
            track_id = payload["current_track_id"]
            if track_id not in tracks_by_id():
                raise ValueError("当前歌曲不在本地音乐库中。")
            track, reason = recommend_track(state, mode, track_id)
            append_interaction_history(
                state, page="music", action="next_track",
                modality=str(payload.get("input_modality", "ui")), target_id=track_id,
            )
            save_state(state)
            return {"message": f"已切换到下一首：{track['title']}。本次未改变偏好。", "track": track, "recommendation_reason": reason, "preference_playlist": mode_preference_playlist(state, mode)}

        if action == "dislike_track":
            mode = state["active_mode"] or "general"
            if not mode:
                raise ValueError("当前没有启用音乐模式，无法记录模式偏好。")
            track_id = payload["current_track_id"]
            if track_id not in tracks_by_id():
                raise ValueError("当前歌曲不在本地音乐库中。")
            modality = str(payload.get("input_modality", "ui"))
            scores = state.setdefault("track_preferences", {}).setdefault(mode, {})
            previous_score = scores.get(track_id, 0)
            had_score = track_id in scores
            was_in_playlist = track_id in state.setdefault("mode_preference_playlists", {}).setdefault(mode, [])
            record_track_preference(state, mode, track_id, -2)
            append_interaction_history(state, page="music", action="dislike_track", modality=modality, target_id=track_id)
            if modality in {"head_gesture", "hand_gesture"}:
                push_undo(state, {
                    "kind": "restore_track_preference", "page": "music", "target_id": track_id,
                    "mode": mode, "track_id": track_id, "previous_score": previous_score,
                    "had_score": had_score, "was_in_playlist": was_in_playlist, "modality": modality,
                    "motion_strength": payload.get("motion_strength"),
                })
            track, reason = recommend_track(state, mode, track_id)
            save_state(state)
            return {"message": f"已降低这首歌的偏好值，正在播放：{track['title']}。", "track": track, "recommendation_reason": reason, "preference_playlist": mode_preference_playlist(state, mode)}

        if action in {"authorize_memo_file", "authorize_memo_files"}:
            files = payload.get("files") or [{
                "file_name": payload.get("file_name"),
                "file_content": payload.get("file_content"),
            }]
            authorization_mode = payload.get("authorization_mode", "merge")
            if not isinstance(files, list) or not 1 <= len(files) <= 10:
                raise ValueError("请一次选择 1 至 10 个备忘录文件。")
            if authorization_mode not in {"merge", "replace"}:
                raise ValueError("不支持的备忘录授权方式。")

            AUTHORIZED_DIR.mkdir(exist_ok=True)
            sources, total_items, names, prepared = [], 0, set(), []
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
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
                stored_name = f"{content_hash}_{filename}"
                stored_path = AUTHORIZED_DIR / stored_name
                stored_path.write_text(content, encoding="utf-8")
                items = parse_memo_file(stored_path)
                if not items:
                    raise ValueError(f"{filename} 中没有可读取的事项。")
                names.add(filename)
                total_items += len(items)
                sources.append({"display_name": filename, "stored_name": stored_name, "item_count": len(items)})
                prepared.append(stored_path)

            # 全部新文件通过校验后才更新授权列表。默认按文件名合并：
            # 用户重新选择同名文件时，新的内容快照会替换旧快照，其余已授权文件保留。
            old_sources = state.get("authorized_sources", [])
            previous_sources = [dict(source) for source in old_sources]
            for source in old_sources:
                if "item_count" not in source:
                    old_path = AUTHORIZED_DIR / source.get("stored_name", "")
                    try:
                        source["item_count"] = len(parse_memo_file(old_path))
                    except (OSError, ValueError):
                        source["item_count"] = 0
            if authorization_mode == "replace":
                final_sources = sources
            else:
                final_sources = [source for source in old_sources if source.get("display_name") not in names] + sources

            authorized_root = AUTHORIZED_DIR.resolve()
            kept_files = {source["stored_name"] for source in final_sources}
            for old_source in old_sources:
                old_path = (AUTHORIZED_DIR / old_source.get("stored_name", "")).resolve()
                if old_path.parent == authorized_root and old_path.exists() and old_path.name not in kept_files:
                    old_path.unlink()
            state["authorized_sources"] = final_sources
            valid_event_keys = {item["event_key"] for item in schedule_items(state)}
            state["completed_events"] = [key for key in state.get("completed_events", []) if key in valid_event_keys]
            append_interaction_history(
                state, page="memo", action="authorize_memo", modality="ui",
                target_id="、".join(sorted(names)),
            )
            push_undo(state, {"kind": "remove_authorized_sources", "page": "memo", "previous_sources": previous_sources})
            save_state(state)
            return {
                "message": f"已记住 {len(final_sources)} 个授权文件；本次读取 {total_items} 条事项。",
                "sources": final_sources,
            }

        if action == "revoke_memo_file":
            stored_name = Path(str(payload.get("stored_name", ""))).name
            source = next((item for item in state.get("authorized_sources", []) if item.get("stored_name") == stored_name), None)
            if not source:
                raise ValueError("未找到该授权文件。")
            state["authorized_sources"] = [item for item in state["authorized_sources"] if item.get("stored_name") != stored_name]
            stored_path = (AUTHORIZED_DIR / stored_name).resolve()
            if stored_path.parent == AUTHORIZED_DIR.resolve() and stored_path.exists():
                stored_path.unlink()
            valid_event_keys = {item["event_key"] for item in schedule_items(state)}
            state["completed_events"] = [key for key in state.get("completed_events", []) if key in valid_event_keys]
            append_interaction_history(state, page="memo", action="revoke_memo", modality="ui", target_id=source.get("display_name"))
            save_state(state)
            return {
                "message": f"已取消对 {source.get('display_name', '该文件')} 的授权。",
                "sources": state["authorized_sources"],
            }

        if action == "query_tomorrow":
            if not state["authorized_sources"]:
                raise ValueError("请先授权备忘录文件。")
            # 演示数据固定以 DEMO_REFERENCE_DATE 为基准，明天为基准日期的次日。
            tomorrow = (demo_reference_date() + timedelta(days=1)).strftime("%Y-%m-%d")
            items = [
                item for item in schedule_items(state)
                if item.get("date") == tomorrow
            ]
            return {
                "message": "已从本地已授权备忘录整理明日事项。",
                "date": tomorrow,
                "items": items,
                "suggestion": "如高优先级事项间隔较短，建议提前完成其中的准备工作。",
                "reminder_offer": build_reminder_offer(state, items),
            }

        if action == "query_schedule":
            scope = str(payload.get("scope", "all"))
            if scope == "today":
                result = schedule_query_result(state, demo_today_str(), f"今天（{demo_today_str()}）日程")
            elif scope == "tomorrow":
                tomorrow = (demo_reference_date() + timedelta(days=1)).strftime("%Y-%m-%d")
                result = schedule_query_result(state, tomorrow, f"明天（{tomorrow}）日程")
            else:
                result = schedule_query_result(state)
            # 默认记录「查看日程」；切换完成状态后的内部刷新不重复记入历史。
            if payload.get("record_history", True):
                append_interaction_history(state, page="memo", action="query_schedule", modality="ui")
                save_state(state)
            return result

        if action == "toggle_event_completion":
            event_key = str(payload["event_key"])
            items_by_key = {item["event_key"]: item for item in schedule_items(state)}
            if event_key not in items_by_key:
                raise ValueError("该日程不在当前已授权文件中。")
            completed = set(state.get("completed_events", []))
            was_completed = event_key in completed
            if bool(payload.get("completed")):
                completed.add(event_key)
            else:
                completed.discard(event_key)
            state["completed_events"] = sorted(completed)
            target_title = items_by_key[event_key].get("title")
            append_interaction_history(state, page="memo", action="toggle_event_completion", modality="ui", target_id=target_title)
            push_undo(state, {
                "kind": "restore_event_completion", "page": "memo",
                "target_id": target_title, "event_key": event_key, "was_completed": was_completed,
            })
            save_state(state)
            return {"message": "已更新日程完成状态。"}

        if action == "create_reminder":
            # 提醒时间由后端根据事项时间与画像提前量推导，不信任前端传入的时间。
            event_key = str(payload.get("event_key", ""))
            message, _ = create_reminder_for_event(state, event_key)
            save_state(state)
            return {"message": message, "reminders": state["reminders"]}

        if action == "remove_reminder":
            # 手动取消某事项的提醒（对应日程项上的“已设置提醒”按钮）。
            event_key = str(payload.get("event_key", ""))
            message = remove_reminder_for_event(state, event_key)
            save_state(state)
            return {"message": message, "reminders": state["reminders"]}

        if action == "decline_reminder":
            # 记录本次拒绝：同一事项的提醒建议不再重复弹出；该拒绝可撤销。
            declined = state.setdefault("declined_reminder_offers", [])
            event_key = str(payload.get("event_key", ""))
            if event_key and event_key not in declined:
                declined.append(event_key)
                push_undo(state, {
                    "kind": "restore_reminder_offer", "page": "memo",
                    "event_key": event_key,
                })
            append_interaction_history(state, page="memo", action="decline_reminder", modality="ui")
            save_state(state)
            return {"message": "已取消本次提醒；不会因此修改长期偏好。", "declined_reminder_offers": declined}

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
