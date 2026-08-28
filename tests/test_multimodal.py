"""本地多模态规则基线的最小回归测试。

运行：python -m unittest tests.test_multimodal
测试仅替换 save_state，不会写入用户的运行状态或备忘录文件。
"""

import time
import unittest

import app


class MultimodalUnderstandingTests(unittest.TestCase):
    def setUp(self):
        self.original_save_state = app.save_state
        app.save_state = lambda _state: None
        app.MULTIMODAL_EVENT_BUFFER.clear()

    def tearDown(self):
        app.save_state = self.original_save_state
        app.MULTIMODAL_EVENT_BUFFER.clear()

    @staticmethod
    def event(modality, timestamp_ms, payload, confidence=1.0):
        return {
            "modality": modality,
            "timestamp_ms": timestamp_ms,
            "received_at_ms": timestamp_ms,
            "confidence": confidence,
            "payload": payload,
        }

    def test_gaze_and_speech_prepare_a_message(self):
        now = int(time.time() * 1000)
        app.MULTIMODAL_EVENT_BUFFER.extend([
            self.event("screen_context", now - 900, {
                "page": "message",
                "visible_targets": [{"target_id": "contact_zhangsan"}],
            }),
            self.event("gaze", now - 600, {
                "page": "message",
                "target_type": "contact",
                "target_id": "contact_zhangsan",
                "zone": "middle-left",
                "dwell_ms": 900,
            }, confidence=0.84),
            self.event("speech_text", now, {
                "page": "message",
                "text": "给他发消息，晚点开会",
                "source": "simulated",
            }),
        ])
        state = {"selected_contact": None, "pending_message": None}

        result = app.understand_multimodal_command(state, now)

        self.assertEqual(result["intent"], "send_message")
        self.assertEqual(result["pending"]["contact_id"], "contact_zhangsan")
        self.assertEqual(result["pending"]["content"], "晚点开会")

    def test_cancel_speech_clears_current_contact_selection(self):
        now = int(time.time() * 1000)
        app.MULTIMODAL_EVENT_BUFFER.append(self.event("speech_text", now, {
            "page": "message",
            "text": "取消发送",
            "source": "simulated",
        }))
        state = {"selected_contact": "contact_zhangsan", "pending_message": None}

        result = app.understand_multimodal_command(state, now)

        self.assertTrue(result["clear_message_form"])
        self.assertIsNone(state["selected_contact"])

    def test_hand_gesture_event_is_accepted_without_camera_data(self):
        now = int(time.time() * 1000)

        result = app.record_multimodal_event({
            "modality": "hand_gesture",
            "timestamp_ms": now,
            "confidence": 0.86,
            "payload": {
                "page": "message",
                "decision": "confirm",
                "gesture": "Thumb_Up",
                "purpose": "message_confirmation",
            },
        })

        self.assertEqual(result["event"]["modality"], "hand_gesture")
        self.assertEqual(result["event"]["payload"]["gesture"], "Thumb_Up")

    def test_hand_pointing_up_can_request_next_track(self):
        now = int(time.time() * 1000)

        result = app.record_multimodal_event({
            "modality": "hand_gesture",
            "timestamp_ms": now,
            "confidence": 0.81,
            "payload": {
                "page": "music",
                "decision": "skip_track",
                "gesture": "Pointing_Up",
                "purpose": "music_skip",
            },
        })

        self.assertEqual(result["event"]["payload"]["decision"], "skip_track")

    def test_music_speech_can_cancel_selected_track(self):
        now = int(time.time() * 1000)
        app.MULTIMODAL_EVENT_BUFFER.append(self.event("speech_text", now, {
            "page": "music",
            "text": "取消当前歌曲选择",
            "source": "simulated",
        }))

        result = app.understand_multimodal_command({"selected_contact": None, "pending_message": None}, now)

        self.assertEqual(result["intent"], "cancel_music_selection")


if __name__ == "__main__":
    unittest.main()
