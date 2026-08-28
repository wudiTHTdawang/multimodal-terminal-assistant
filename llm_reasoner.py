"""本地大模型的受限推理适配器。

模型只能阅读已经脱敏、结构化的多模态结果，并且只负责解释冲突和润色答复。
意图、目标对象和最终执行仍由 app.py 中的确定性安全规则负责。
"""

from __future__ import annotations

import json
import os
from urllib import error, request


DEFAULT_MODEL = os.getenv("ZHIJI_LLM_MODEL", "qwen2.5:3b-instruct")
DEFAULT_URL = os.getenv("ZHIJI_LLM_URL", "http://127.0.0.1:11434/api/chat")
TIME_BUDGET_SECONDS = float(os.getenv("ZHIJI_LLM_TIMEOUT_SECONDS", "2.8"))


def _safe_text(value, limit=100):
    return str(value or "").replace("\n", " ").strip()[:limit]


def _parse_model_json(text):
    """接受模型偶尔加上的 Markdown 代码块，但只返回白名单字段。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("模型输出不是对象")
    return {
        # 紧凑键名可明显缩短端侧 3B 模型的生成时间；同时兼容便于人工阅读的旧键名。
        "response": _safe_text(data.get("r", data.get("response")), 60),
        "conflict_explanation": _safe_text(data.get("c", data.get("conflict_explanation")), 60),
        "personalization_reason": _safe_text(data.get("p", data.get("personalization_reason")), 60),
    }


def enhance_local_response(*, scene, speech_text, rule_result, fusion, profile):
    """在限定时间内请求 Ollama；不可用时返回 None，由调用方使用规则模板。"""
    prompt_data = {
        "scene": _safe_text(scene, 30),
        "speech_text": _safe_text(speech_text, 120),
        "rule_result": {
            "intent": _safe_text(rule_result.get("intent"), 40),
            "message": _safe_text(rule_result.get("message"), 150),
            "needs_clarification": bool(rule_result.get("needs_clarification")),
            # 日程只给聚合统计，不把备忘录正文交给语言模型。
            "schedule_summary": {
                "title": _safe_text(rule_result.get("title"), 50),
                "total": rule_result.get("total"),
                "completed": rule_result.get("completed"),
                "past": rule_result.get("past"),
            },
        },
        "fusion": {
            "modalities": fusion.get("modalities", []),
            "gaze_target_id": _safe_text(fusion.get("gaze_target_id"), 60),
            "decision_values": fusion.get("decision_values", []),
            "conflict": _safe_text(fusion.get("conflict"), 80),
        },
        "profile": {
            "display_name": _safe_text(profile.get("display_name"), 40),
            "response_style": _safe_text(profile.get("response_style"), 30),
            "schedule_reminder_minutes": profile.get("schedule_reminder_minutes"),
        },
    }
    system = (
        "你是端侧多模态助手的解释模块。只能根据提供的结构化事实生成简洁中文，"
        "不能新增联系人、日程、操作或改变 intent。若 conflict 非空，说明为什么系统要再次确认或优先取消。"
        "必须只输出紧凑 JSON：{\"r\":\"最多20字\",\"c\":\"最多20字\",\"p\":\"最多20字\"}。未知字段用空字符串。"
    )
    body = {
        "model": DEFAULT_MODEL,
        "stream": False,
        "keep_alive": "5m",
        # 32 token 足够生成紧凑 JSON；实测可将预热后响应从 5 秒以上降至约 1 秒。
        "options": {"temperature": 0, "num_predict": 32, "num_ctx": 512},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False)},
        ],
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(DEFAULT_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(http_request, timeout=TIME_BUDGET_SECONDS) as response:
            raw = json.loads(response.read().decode("utf-8"))
        return _parse_model_json(raw.get("message", {}).get("content", ""))
    except (OSError, ValueError, TypeError, json.JSONDecodeError, error.URLError):
        return None
