"""scenarios.json 端到端验收样例运行器。

运行：python tests/run_scenarios.py [--verbose]

把 tests/scenarios.json 的 30 条端到端样例转换为 app 的真实调用
（事件缓冲 + 意图理解接口 + 执行动作），按 expected 关键字段断言，
输出每条结果、通过率与当前实现尚未覆盖的能力清单。

说明：
- 运行时不启动 Ollama（LOCAL_LLM_ENABLED=False，纯规则基线，结果可复现）；
- 不写盘：替换 load_state / save_state，全部在内存执行；
- 样例中的“页面”事件主要驱动流程（如 confirm_send 页表示存在待确认消息），
  同时写入 screen_context 供时间窗校验使用。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402

SCENARIOS_PATH = ROOT / "tests" / "scenarios.json"

PAGE_TO_APP = {
    "contacts": "message",
    "confirm_send": "message",
    "focus_music": "music",
    "now_playing_focus": "music",
    "now_playing_driving": "music",
    "now_playing_entertainment": "music",
    "offer_reminder": "memo",
    "offer_reminder_memo_001": "memo",
}

MODE_TRACK = {"focus": "track_010", "driving": "track_012", "entertainment": "track_024", "general": "track_031"}

# expected 意图名 → 运行时可接受的意图名（P4 已全部实现）。
INTENT_EQUIV = {
    "send_message": {"send_message"},
    "clarify_contact": {"send_message"},           # 行为正确：无有效对象时返回澄清
    "confirm_send": {"confirm", "confirm_send"},
    "cancel_send": {"cancel"},
    "next_track": {"next_track"},
    "query_tomorrow_plan": {"query_schedule_tomorrow"},
    "query_date_plan": {"query_date_plan"},
    "request_memo_authorization": {"__no_authorized_memo__"},
    "start_focus": {"start_focus"},
    "play_music": {"play_music"},
    "like_track": {"like_track"},
    "dislike_track": {"dislike_track"},
    "clarify_or_offer_start_focus": {"offer_start_focus"},
    "clarify_music_selection": {"clarify_music_selection"},
    "query_priority": {"query_priority"},
    "decline_reminder": {"decline_reminder"},
    "confirm_reminder": {"__create_reminder_ok__"},
    "update_reminder_preference": {"update_reminder_preference"},
    "request_edit_memo": {"request_edit_memo"},
    "analyze_schedule_pressure": {"analyze_schedule_pressure"},
}

NO_UNDERSTAND = "__no_understand__"


def make_event(modality, timestamp_ms, payload, confidence=1.0):
    return {
        "modality": modality,
        "timestamp_ms": timestamp_ms,
        "received_at_ms": timestamp_ms,
        "confidence": confidence,
        "payload": payload,
    }


def fresh_state(profile_id):
    state = app.default_state()
    state["active_profile_id"] = profile_id
    for key, value in app.profile_runtime_defaults(profile_id).items():
        state[key] = value
    return state


def run_scenario(scenario):
    """执行一条样例，返回 (checks, notes)。"""
    scene = scenario["scene"]
    profile_id = scenario.get("profile_id", "user_xiaoyu")
    expected = scenario["expected"]
    state = fresh_state(profile_id)
    authorized = scenario.get("authorized_sources")
    if authorized:
        state["authorized_sources"] = [
            {"display_name": name, "stored_name": name, "item_count": 0} for name in authorized
        ]

    original_load, original_save = app.load_state, app.save_state
    original_llm = app.LOCAL_LLM_ENABLED
    app.load_state = lambda: state
    app.save_state = lambda _s: None
    app.LOCAL_LLM_ENABLED = False
    app.MULTIMODAL_EVENT_BUFFER.clear()
    handler = object.__new__(app.AssistantHandler)
    base_ts = int(time.time() * 1000)
    results = []          # 每条理解/执行结果
    notes = []
    last_schedule_items = None
    last_offer = None
    last_clarification = None
    current_track_id = None

    try:
        for ev in scenario.get("events", []):
            ts = base_ts + int(ev["timestamp_ms"])
            kind = ev["type"]
            value = str(ev.get("value", ""))
            confidence = float(ev.get("confidence", 1.0))

            if kind == "page":
                page_app = PAGE_TO_APP.get(value, scene)
                visible = (
                    [{"target_type": "contact", "target_id": c["id"], "zone": "center"}
                     for c in json.loads((ROOT / "data" / "contacts.json").read_text("utf-8"))]
                    if page_app == "message" else []
                )
                app.MULTIMODAL_EVENT_BUFFER.append(make_event("screen_context", ts, {
                    "page": page_app, "visible_targets": visible,
                }))
                # confirm_send 页面隐含“已有一条待确认消息”
                if value == "confirm_send" and not state.get("pending_message"):
                    state["pending_message"] = {
                        "contact": "张三", "contact_id": "contact_zhangsan", "content": "测试消息",
                    }
                # now_playing 页面隐含“已在某模式下播放”
                if value.startswith("now_playing"):
                    mode = value.replace("now_playing_", "") or "focus"
                    state["active_mode"] = mode if mode in {"focus", "driving", "entertainment"} else "focus"
                    state["focus_mode"] = state["active_mode"] == "focus"
                    current_track_id = MODE_TRACK.get(state["active_mode"], "track_010")
                # focus_music 页面隐含“专注模式上下文”
                if value == "focus_music":
                    state["active_mode"] = "focus"
                    state["focus_mode"] = True
                # offer_reminder 页面隐含“系统已给出提醒建议”
                if value.startswith("offer_reminder") and scene == "memo" and last_offer is None:
                    try:
                        sched = app.schedule_query_result(state)
                        last_offer = sched.get("reminder_offer")
                        last_schedule_items = sched.get("items")
                    except ValueError:
                        pass

            elif kind == "gaze":
                app.MULTIMODAL_EVENT_BUFFER.append(make_event("gaze", ts, {
                    "page": scene,
                    "target_type": "contact" if scene == "message" else "music_track",
                    "target_id": value, "zone": "center", "dwell_ms": 1000,
                }, confidence))

            elif kind == "speech":
                app.MULTIMODAL_EVENT_BUFFER.append(make_event("speech_text", ts, {
                    "text": value, "page": scene, "source": "simulated",
                }, confidence))
                try:
                    result = app.understand_multimodal_command(
                        state, ts,
                        current_track_id=current_track_id if scene == "music" else None,
                    )
                    results.append({"type": "understand", "intent": result.get("intent"), "result": result})
                    last_clarification = bool(result.get("needs_clarification"))
                    if isinstance(result.get("items"), list):
                        last_schedule_items = result["items"]
                    if isinstance(result.get("reminder_offer"), dict):
                        last_offer = result["reminder_offer"]
                    # 音乐页“下一首”在真实前端会继续调用切歌动作，这里同步执行
                    if result.get("intent") == "next_track" and state.get("active_mode"):
                        track_id = MODE_TRACK.get(state.get("active_mode") or "general", "track_010")
                        app.AssistantHandler._dispatch_action(handler, {
                            "action": "next_track", "current_track_id": track_id,
                        }, state)
                        results.append({"type": "next_track", "intent": "next_track", "result": {}})
                except ValueError as error:
                    notes.append(f"__no_authorized_memo__" if "授权" in str(error) else f"__error__:{error}")
                    results.append({"type": "understand", "intent": NO_UNDERSTAND, "result": {}})

            elif kind == "nod":
                app.MULTIMODAL_EVENT_BUFFER.append(make_event("head_gesture", ts, {
                    "page": scene if scene in {"message", "music"} else "message",
                    "decision": "confirm", "gesture": "nod", "purpose": "message_confirmation",
                }, confidence))
                if scene == "message" and state.get("pending_message"):
                    out = app.AssistantHandler._dispatch_action(handler, {"action": "confirm_send"}, state)
                    results.append({"type": "confirm_send", "intent": "confirm_send", "result": out})
                    notes.append("已执行本地模拟发送")
                elif scene == "memo" and last_offer:
                    out = app.AssistantHandler._dispatch_action(handler, {
                        "action": "create_reminder", "event_key": last_offer["event_key"],
                    }, state)
                    results.append({"type": "create_reminder", "intent": "confirm_reminder", "result": out})
                    notes.append("已创建本地提醒")
    finally:
        app.load_state, app.save_state = original_load, original_save
        app.LOCAL_LLM_ENABLED = original_llm

    checks = _assert(scenario, results, state, notes, last_clarification, last_schedule_items, last_offer)
    if expected.get("reason") == "gaze_outside_alignment_window" and not any(c[0] == "clarify" and c[3] for c in checks):
        notes.append("时间窗口径差异：样例按语音前 3s，运行实现按语音前 4s（见前后端多模态接口说明.md）")
    if scenario["id"] == "B05":
        notes.append("偏好惩罚差异：运行 next_track 不惩罚当前曲目偏好（README 场景 B 设计），样例期望 -1")
    if scenario["expected"].get("preference_delta"):
        notes.append("偏好口径差异：样例按曲风记 delta，运行按曲目记分（like +3 / dislike -2）")
    return checks, notes


def _assert(scenario, results, state, notes, last_clarification, last_schedule_items, last_offer):
    expected = scenario["expected"]
    checks = []
    actual_intents = [r["intent"] for r in results if r["type"] in {"understand", "confirm_send", "create_reminder", "next_track"}] or [NO_UNDERSTAND]
    exp_intent = expected.get("intent")

    # 意图
    if exp_intent == "request_memo_authorization":
        ok = any("__no_authorized_memo__" in note for note in notes)
        checks.append(("intent", exp_intent, "no_authorized_memo" if ok else actual_intents, ok))
    elif exp_intent == "confirm_reminder":
        ok = any(r["type"] == "create_reminder" for r in results)
        checks.append(("intent", exp_intent, "create_reminder" if ok else actual_intents, ok))
    else:
        ok = any(ai in INTENT_EQUIV.get(exp_intent, {exp_intent}) for ai in actual_intents)
        checks.append(("intent", exp_intent, actual_intents, ok))

    # 澄清类（无有效对象 / 低置信度视线 / 时间窗超界）
    reason = expected.get("reason")
    if reason in {"no_visual_or_spoken_contact", "low_gaze_confidence", "gaze_outside_alignment_window"}:
        ok = bool(last_clarification) or any("请先注视" in str(r.get("result", {}).get("message", "")) for r in results)
        checks.append(("clarify", reason, last_clarification, ok))

    # 消息联系人 / 内容 / 需确认
    last_understand = next((r["result"] for r in reversed(results) if r["type"] == "understand"), {})
    pending = last_understand.get("pending") or {}
    if "contact" in expected:
        actual_contact = pending.get("contact_id") or state.get("selected_contact")
        checks.append(("contact", expected["contact"], actual_contact, actual_contact == expected["contact"]))
    if "content" in expected:
        actual_content = pending.get("content")
        checks.append(("content", expected["content"], actual_content, actual_content == expected["content"]))
    if expected.get("requires_confirmation"):
        # 消息流：出现待确认内容；提醒偏好类：要求二次确认
        pending_ok = bool(pending) or any(r["type"] == "confirm_send" for r in results)
        clarify_ok = bool(last_clarification)
        checks.append(("requires_confirmation", True, pending or last_clarification, pending_ok or clarify_ok))

    # 日程：memo_ids / memo_id / date / 提醒 offer
    if "memo_ids" in expected:
        items = last_schedule_items or []
        actual_ids = [item.get("id") for item in items]
        checks.append(("memo_ids", expected["memo_ids"], actual_ids, sorted(actual_ids) == sorted(expected["memo_ids"])))
    if "memo_id" in expected:
        actual_memo = next((r["result"].get("memo_id") for r in reversed(results)
                            if r["type"] == "understand" and r["result"].get("memo_id")), None)
        if actual_memo is None and last_offer:
            actual_memo = last_offer.get("memo_id")  # 提醒确认场景：offer 即对应事项
        checks.append(("memo_id", expected["memo_id"], actual_memo, actual_memo == expected["memo_id"]))
    if "date" in expected:
        actual_date = next((r["result"].get("date") for r in reversed(results)
                            if r["type"] == "understand" and r["result"].get("date")), None)
        checks.append(("date", expected["date"], actual_date, actual_date == expected["date"]))
    if expected.get("should_offer_reminder"):
        checks.append(("should_offer_reminder", True, bool(last_offer), bool(last_offer)))
    if expected.get("should_suggest"):
        actual_suggest = next((r["result"].get("should_suggest") for r in reversed(results)
                               if r["type"] == "understand" and r["result"].get("should_suggest") is not None), None)
        checks.append(("should_suggest", True, actual_suggest, bool(actual_suggest)))

    # 音乐：曲风 / 模式上下文
    if "genre" in expected:
        actual_genre = next((r["result"].get("recommended_genre") for r in reversed(results)
                             if r["type"] == "understand" and r["result"].get("recommended_genre")), None)
        checks.append(("genre", expected["genre"], actual_genre, actual_genre == expected["genre"]))
    if "recommended_genre" in expected:
        actual_genre = next((r["result"].get("recommended_genre") for r in reversed(results)
                             if r["type"] == "understand" and r["result"].get("recommended_genre")), None)
        checks.append(("recommended_genre", expected["recommended_genre"], actual_genre, actual_genre == expected["recommended_genre"]))
    if "context" in expected:
        actual_mode = next((r["result"].get("mode") for r in reversed(results)
                            if r["type"] == "understand" and r["result"].get("mode")), state.get("active_mode"))
        # 样例语境名（focus_mode 等）与运行时模式键（focus）的映射
        context_map = {"focus_mode": "focus", "driving_mode": "driving", "entertainment_mode": "entertainment", "general_mode": "general"}
        expected_mode = context_map.get(expected["context"], expected["context"])
        checks.append(("context", expected["context"], actual_mode, actual_mode == expected_mode))

    # 动作
    exp_action = expected.get("action")
    if exp_action == "simulate_send":
        fired = any(
            r["type"] == "confirm_send"
            or (r["type"] == "understand" and bool(r["result"].get("simulated_send")))
            for r in results
        )
        checks.append(("action", exp_action, "simulate_send" if fired else None, fired))
    elif exp_action == "cancel":
        fired = any(r["intent"] in {"cancel", "cancel_music_selection"} for r in results)
        checks.append(("action", exp_action, "cancel" if fired else None, fired))
    elif exp_action == "create_local_reminder":
        fired = any(r["type"] == "create_reminder" for r in results)
        checks.append(("action", exp_action, "create_reminder" if fired else None, fired))
    elif exp_action == "skip_current_track":
        fired = any(r["type"] == "next_track" for r in results) or any(
            r["intent"] == "dislike_track" and bool(r["result"].get("track")) for r in results
        )
        checks.append(("action", exp_action, "skip" if fired else None, fired))
    elif exp_action == "show_focus_music_options":
        fired = any(r["intent"] == "start_focus" and bool(r["result"].get("track")) for r in results)
        checks.append(("action", exp_action, "start_focus" if fired else None, fired))
    elif exp_action == "do_not_create_current_reminder":
        declined = any(r["intent"] == "decline_reminder" for r in results)
        created = any(r["type"] == "create_reminder" for r in results)
        checks.append(("action", exp_action, "decline" if declined and not created else None, declined and not created))
    elif exp_action == "ask_for_edit_details_and_confirmation":
        fired = any(r["intent"] == "request_edit_memo" and bool(r["result"].get("needs_clarification")) for r in results)
        checks.append(("action", exp_action, "request_edit_memo" if fired else None, fired))
    elif exp_action:
        # 其余动作当前实现不触发，记为未校验
        checks.append(("action", exp_action, "not-asserted", False))

    return checks


def main():
    verbose = "--verbose" in sys.argv
    scenarios = json.loads(SCENARIOS_PATH.read_text("utf-8"))
    passed, failed, unsupported = [], [], []
    unsupported_intents = {name for name, equiv in INTENT_EQUIV.items() if not equiv}

    print(f"===== scenarios.json 验收样例运行结果（共 {len(scenarios)} 条，规则基线，LLM 关闭）=====")
    for scenario in scenarios:
        checks, notes = run_scenario(scenario)
        ok = all(check[3] for check in checks)
        expected_intent = scenario["expected"].get("intent")
        if expected_intent in unsupported_intents:
            bucket = unsupported
        elif ok:
            bucket = passed
        else:
            bucket = failed
        bucket.append(scenario["id"])
        status = "PASS" if ok else ("UNSUPPORTED" if expected_intent in unsupported_intents else "FAIL")
        print(f"  {scenario['id']} [{scenario['scene']}] {status:<12} intent={expected_intent}")
        if not ok and verbose:
            for field, exp, actual, check_ok in checks:
                if not check_ok:
                    print(f"      - {field}: 期望 {exp!r}，实际 {actual!r}")
        if notes and verbose:
            for note in notes:
                print(f"      * {note}")

    print(f"\n通过 {len(passed)} 条，失败 {len(failed)} 条，未实现意图 {len(unsupported)} 条")
    print(f"实际支持并通过率：{len(passed)}/{len(scenarios)}（{len(passed) / len(scenarios) * 100:.1f}%）")
    if failed:
        print(f"\n失败样例：{', '.join(failed)}")
    if unsupported:
        print(f"未实现意图样例：{', '.join(unsupported)}")
        print("未实现能力清单：")
        for name in sorted(unsupported_intents):
            print(f"  - {name}")


if __name__ == "__main__":
    main()
