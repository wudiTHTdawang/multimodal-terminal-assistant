"""P0/P1/P2 与验收样例修复的回归测试。

运行：python -m unittest tests.test_regression
覆盖：演示基准日期、提醒 offer/去重/撤销、建议面板历史映射、手势自适应画像、
next_track 历史、遗留字段清理，以及运行器暴露的新安全修复
（“给X发消息”联系人名解析、否定句取消、低置信度视线澄清）。
"""

import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import app


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.original_save = app.save_state
        self.original_llm = app.LOCAL_LLM_ENABLED
        app.save_state = lambda _state: None
        app.LOCAL_LLM_ENABLED = False
        app.MULTIMODAL_EVENT_BUFFER.clear()
        self.handler = object.__new__(app.AssistantHandler)

    def tearDown(self):
        app.save_state = self.original_save
        app.LOCAL_LLM_ENABLED = self.original_llm
        app.MULTIMODAL_EVENT_BUFFER.clear()
        tmp = getattr(self, "_memo_tmp_name", None)
        if tmp:
            path = app.AUTHORIZED_DIR / tmp
            if path.exists():
                path.unlink()

    def state_with_memo(self):
        """按真实“今天”自生成临时备忘录数据，避免依赖提交文件的日期。"""
        state = app.default_state()
        state["active_profile_id"] = "user_xiaoyu"
        today = app.demo_reference_date()
        entries = [
            {"id": "memo_000", "date": (today - timedelta(days=1)).isoformat(), "time": "16:00", "duration_minutes": 30,
             "title": "提交周报", "content": "周报", "location": "会议室", "priority": "high"},
            {"id": "memo_001", "date": (today + timedelta(days=1)).isoformat(), "time": "10:00", "duration_minutes": 90,
             "title": "组会", "content": "组会", "location": "实验室 302", "priority": "high"},
            {"id": "memo_002", "date": (today + timedelta(days=1)).isoformat(), "time": "14:00", "duration_minutes": 30,
             "title": "提交实验报告", "content": "报告", "location": "线上", "priority": "high"},
            {"id": "memo_003", "date": (today + timedelta(days=1)).isoformat(), "time": "19:00", "duration_minutes": 60,
             "title": "健身", "content": "力量训练", "location": "体育馆", "priority": "low"},
            {"id": "memo_004", "date": (today + timedelta(days=2)).isoformat(), "time": "09:30", "duration_minutes": 60,
             "title": "论文阅读", "content": "论文", "location": "图书馆", "priority": "medium"},
        ]
        app.AUTHORIZED_DIR.mkdir(exist_ok=True)
        self._memo_tmp_name = "test_memo_tmp.json"
        (app.AUTHORIZED_DIR / self._memo_tmp_name).write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        state["authorized_sources"] = [{
            "display_name": "test_memo.json", "stored_name": self._memo_tmp_name, "item_count": len(entries),
        }]
        return state

    # ---------- 演示基准日期 ----------

    @staticmethod
    def tomorrow_str():
        return (app.demo_reference_date() + timedelta(days=1)).isoformat()

    def test_past_event_is_grayed_and_not_offered(self):
        state = self.state_with_memo()
        all_items = app.schedule_items(state)
        past_date = (app.demo_reference_date() - timedelta(days=1)).isoformat()
        past_item = next(item for item in all_items if item.get("date") == past_date)
        self.assertTrue(past_item["is_past"])  # 过期事项标记为灰显
        offer = app.build_reminder_offer(state, all_items)
        self.assertEqual(offer["title"], "组会")  # 过期的高优先级周报不被建议提醒

    def test_declined_reminder_offer_persists(self):
        state = self.state_with_memo()
        first_offer = app.schedule_query_result(state, self.tomorrow_str())["reminder_offer"]
        self.assertEqual(first_offer["title"], "组会")
        app.AssistantHandler._dispatch_action(self.handler, {
            "action": "decline_reminder", "event_key": first_offer["event_key"],
        }, state)
        self.assertIn(first_offer["event_key"], state["declined_reminder_offers"])
        # 拒绝后同一事项不再建议（可转向下一高优先级事项）
        next_offer = app.schedule_query_result(state, self.tomorrow_str())["reminder_offer"]
        self.assertNotEqual(next_offer["event_key"], first_offer["event_key"])
        self.assertEqual(next_offer["title"], "提交实验报告")

    def test_page_gated_voice_command_is_ignored(self):
        # 日程页说音乐指令“播放下一首”应被忽略，不触发任何效果
        now = int(time.time() * 1000)
        app.MULTIMODAL_EVENT_BUFFER.append({
            "modality": "speech_text", "timestamp_ms": now, "received_at_ms": now,
            "confidence": 1.0, "payload": {"page": "memo", "text": "播放下一首", "source": "simulated"},
        })
        result = app.understand_multimodal_command(app.default_state(), now)
        self.assertTrue(result.get("ignored"))
        self.assertIn("日程", result["message"])
        # 音乐页说同样的话应正常解析为切歌
        now2 = int(time.time() * 1000)
        app.MULTIMODAL_EVENT_BUFFER.append({
            "modality": "speech_text", "timestamp_ms": now2, "received_at_ms": now2,
            "confidence": 1.0, "payload": {"page": "music", "text": "播放下一首", "source": "simulated"},
        })
        result2 = app.understand_multimodal_command(app.default_state(), now2)
        self.assertEqual(result2["intent"], "next_track")

    def test_parse_new_phrasings(self):
        cases = {
            "发消息给李四，明天见": ("send_message", "明天见", "李四"),
            "来首轻音乐": ("play_music", "轻音乐", ""),
            "停止模式": ("stop_mode", "", ""),
            "切歌": ("next_track", "", ""),
            "这首歌好听": ("like_track", "", ""),
            "看看日程": ("query_schedule_all", "", ""),
        }
        for text, expected in cases.items():
            result = app.parse_simulated_speech(text)
            self.assertEqual(result, expected, text)

    def test_voice_set_and_unset_reminder_by_title(self):
        state = self.state_with_memo()
        now = int(time.time() * 1000)
        app.MULTIMODAL_EVENT_BUFFER.append({
            "modality": "speech_text", "timestamp_ms": now, "received_at_ms": now,
            "confidence": 1.0, "payload": {"page": "memo", "text": "给组会设置提醒", "source": "simulated"},
        })
        result = app.understand_multimodal_command(state, now)
        self.assertEqual(result["intent"], "set_reminder")
        self.assertIn("已在本项目内创建", result["message"])
        self.assertEqual(len(state["reminders"]), 1)
        now2 = int(time.time() * 1000)
        app.MULTIMODAL_EVENT_BUFFER.append({
            "modality": "speech_text", "timestamp_ms": now2, "received_at_ms": now2,
            "confidence": 1.0, "payload": {"page": "memo", "text": "取消组会提醒", "source": "simulated"},
        })
        result2 = app.understand_multimodal_command(state, now2)
        self.assertEqual(result2["intent"], "unset_reminder")
        self.assertEqual(state["reminders"], [])

    def test_demo_now_anchors_past_and_schedule(self):
        reference = app.demo_reference_date()
        original_now = app.demo_now
        app.demo_now = lambda: datetime.combine(reference, datetime.min.time()).replace(hour=12)
        try:
            tomorrow = (reference + timedelta(days=1)).isoformat()
            self.assertFalse(app.is_past_event({"date": tomorrow, "time": "10:00"}))
            self.assertTrue(app.is_past_event({"date": reference.isoformat(), "time": "10:00"}))   # 早于演示正午
            self.assertFalse(app.is_past_event({"date": reference.isoformat(), "time": "18:00"}))  # 晚于演示正午
            result = app.schedule_query_result(self.state_with_memo(), tomorrow, f"明天（{tomorrow}）日程")
            self.assertEqual(result["date"], tomorrow)
            self.assertEqual(result["total"], 3)
        finally:
            app.demo_now = original_now

    def test_query_schedule_scope_today_and_tomorrow(self):
        state = self.state_with_memo()
        today = app.AssistantHandler._dispatch_action(self.handler, {
            "action": "query_schedule", "scope": "today", "record_history": False,
        }, state)
        self.assertEqual(today["title"], f"今天（{app.demo_today_str()}）日程")
        tomorrow = app.AssistantHandler._dispatch_action(self.handler, {
            "action": "query_schedule", "scope": "tomorrow", "record_history": False,
        }, state)
        self.assertEqual(tomorrow["date"], self.tomorrow_str())
        self.assertEqual(tomorrow["total"], 3)

    # ---------- 提醒链路 ----------

    def test_reminder_offer_dedupe_and_undo(self):
        state = self.state_with_memo()
        offer = app.schedule_query_result(state, self.tomorrow_str())["reminder_offer"]
        self.assertIsNotNone(offer)
        self.assertEqual(offer["remind_time"], f"{self.tomorrow_str()} 09:10")  # 明天 10:00 - 画像提前量 50 分钟
        first = app.AssistantHandler._dispatch_action(self.handler, {
            "action": "create_reminder", "event_key": offer["event_key"],
        }, state)
        self.assertIn("已在本项目内创建", first["message"])
        second = app.AssistantHandler._dispatch_action(self.handler, {
            "action": "create_reminder", "event_key": offer["event_key"],
        }, state)
        self.assertIn("未重复创建", second["message"])
        self.assertEqual(len(state["reminders"]), 1)
        undo = app.AssistantHandler._dispatch_action(self.handler, {
            "action": "undo_last_nontext_operation",
        }, state)
        self.assertIn("已撤销", undo["message"])
        self.assertEqual(state["reminders"], [])

    # ---------- 建议面板个性化 ----------

    def test_page_suggestions_use_history_mapping(self):
        state = app.default_state()
        state["active_profile_id"] = "user_xiaoyu"
        state["interaction_history"] = [
            {"page": "message", "action": "open_page", "modality": "ui"},
            {"page": "message", "action": "open_page", "modality": "ui"},
            {"page": "message", "action": "select_contact", "modality": "gaze", "target_id": "contact_zhangsan"},
            {"page": "message", "action": "send_message", "modality": "speech_text"},
            {"page": "message", "action": "confirm_send", "modality": "ui"},
            {"page": "message", "action": "undo", "modality": "ui"},
        ]
        summary = app.history_summary(state, "message")
        self.assertNotIn("open_page", summary["frequent_actions"])
        self.assertNotIn("undo", summary["frequent_actions"])
        result = app.page_suggestions(state, "message")
        ids = [item["id"] for item in result["actions"]]
        self.assertEqual(ids[0], "prepare_message")  # 3 分，高于 focus_contacts 的 2 分
        self.assertIn("focus_contacts", ids)

    # ---------- 手势自适应画像 ----------

    def test_gesture_profile_learning_and_adaptive_threshold(self):
        state = app.default_state()
        state["gesture_profile"] = {"head_min_strength": 0.0, "confirmed_samples": [], "undone_samples": []}
        app.update_gesture_profile(state, {
            "modality": "head_gesture",
            "payload": {"page": "music", "decision": "confirm", "motion_strength": 0.42},
        })
        self.assertEqual(state["gesture_profile"]["confirmed_samples"], [0.42])
        state["active_profile_id"] = "user_xiaoyu"
        state["active_mode"] = "focus"
        app.AssistantHandler._dispatch_action(self.handler, {
            "action": "like_track", "track_id": "track_010",
            "input_modality": "head_gesture", "motion_strength": 0.31,
        }, state)
        app.AssistantHandler._dispatch_action(self.handler, {
            "action": "undo_last_nontext_operation",
        }, state)
        self.assertEqual(state["gesture_profile"]["undone_samples"], [0.31])
        self.assertGreater(state["gesture_profile"]["head_min_strength"], 0)

    # ---------- 挥手切歌历史 ----------

    def test_next_track_records_history_with_modality(self):
        state = app.default_state()
        state["active_profile_id"] = "user_xiaoyu"
        state["active_mode"] = "focus"
        state["recommendation_turns"] = {"focus": 0}
        state["track_preferences"] = {"focus": {"track_010": 1}}
        state["mode_preference_playlists"] = {"focus": ["track_010"]}
        app.AssistantHandler._dispatch_action(self.handler, {
            "action": "next_track", "current_track_id": "track_010", "input_modality": "hand_gesture",
        }, state)
        last = state["interaction_history"][-1]
        self.assertEqual(last["action"], "next_track")
        self.assertEqual(last["modality"], "hand_gesture")

    # ---------- 遗留字段清理 ----------

    def test_load_state_cleans_legacy_field(self):
        original_file, original_cache = app.STATE_FILE, app.STATE_CACHE
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
        json.dump({
            "active_profile_id": "user_xiaoyu",
            "preference_adjustments": {"focus": {"lofi": 1}},
            "profile_states": {"user_xiaoyu": {"preference_adjustments": {"focus": {"lofi": 1}}}},
            "track_preferences": {"focus": {"track_010": 1}},
        }, tmp, ensure_ascii=False)
        tmp.close()
        try:
            app.STATE_FILE = Path(tmp.name)
            app.STATE_CACHE = None
            state = app.load_state()
            self.assertNotIn("preference_adjustments", state)
            self.assertNotIn("preference_adjustments", state["profile_states"]["user_xiaoyu"])
        finally:
            app.STATE_FILE = original_file
            app.STATE_CACHE = original_cache
            Path(tmp.name).unlink(missing_ok=True)

    # ---------- 验收样例暴露的新安全修复 ----------

    def test_spoken_contact_name_resolves(self):
        now = int(time.time() * 1000)
        app.MULTIMODAL_EVENT_BUFFER.append({
            "modality": "speech_text", "timestamp_ms": now, "received_at_ms": now,
            "confidence": 1.0,
            "payload": {"page": "message", "text": "给妈妈发消息，我晚点回家", "source": "simulated"},
        })
        state = {"selected_contact": None, "pending_message": None}
        result = app.understand_multimodal_command(state, now)
        self.assertEqual(result["pending"]["contact_id"], "contact_mama")
        self.assertEqual(result["pending"]["content"], "我晚点回家")

    def test_negative_speech_cancels_instead_of_send(self):
        intent, content, spoken = app.parse_simulated_speech("不要给他发消息")
        self.assertEqual(intent, "cancel")
        self.assertEqual(content, "")
        intent2, _, _ = app.parse_simulated_speech("别给她发消息")
        self.assertEqual(intent2, "cancel")

    def test_low_confidence_gaze_clarifies(self):
        now = int(time.time() * 1000)
        app.MULTIMODAL_EVENT_BUFFER.extend([
            {
                "modality": "gaze", "timestamp_ms": now - 500, "received_at_ms": now - 500,
                "confidence": 0.43,
                "payload": {"page": "message", "target_type": "contact", "target_id": "contact_zhangsan",
                            "zone": "center", "dwell_ms": 1000},
            },
            {
                "modality": "speech_text", "timestamp_ms": now, "received_at_ms": now,
                "confidence": 1.0,
                "payload": {"page": "message", "text": "给他发消息，晚点开会", "source": "simulated"},
            },
        ])
        state = {"selected_contact": None, "pending_message": None}
        result = app.understand_multimodal_command(state, now)
        self.assertTrue(result.get("needs_clarification"))
        self.assertNotIn("pending", result)


if __name__ == "__main__":
    unittest.main()
