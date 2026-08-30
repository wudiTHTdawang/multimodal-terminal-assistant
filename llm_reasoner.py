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
        "actions": [str(value)[:50] for value in data.get("a", []) if isinstance(value, str)][:2],
    }


def _call_local_model(system, prompt_data, num_predict=32):
    body = {
        "model": DEFAULT_MODEL,
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 512},
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


def enhance_local_response(*, scene, speech_text, rule_result, fusion, profile, history=None):
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
        # 仅提供次数和类别，不提供任何原始视觉数据、语音文本历史或消息正文。
        "history_summary": history or {},
    }
    system = (
        "你是端侧多模态助手的解释模块。只能根据提供的结构化事实生成简洁中文，"
        "不能新增联系人、日程、操作或改变 intent。若 conflict 非空，说明为什么系统要再次确认或优先取消。"
        "不得输出思考过程、Markdown 或额外文本。必须只输出紧凑 JSON："
        "{\"r\":\"最多20字\",\"c\":\"最多20字\",\"p\":\"最多20字\",\"a\":[]}。未知字段用空字符串。"
    )
    return _call_local_model(system, prompt_data)


def normalize_speech_command(text, scene=None):
    """把麦克风语音转写文本整理成规则可解析的规范指令。

    语音转写常有近音错误（“所以”≈“所有”、“下一收”≈“下一首”）和祈使性插入语
    （“让他/告诉他/跟他说/说”）。本函数只做校正与改写，不得新增人物、消息内容、
    曲目或操作语义；失败或不合规时返回 None，由 app.py 回退原文。最终意图由安全规则决定。
    """
    canonical = {
        "message": "给X发消息，内容（内容里不要带“让他/告诉他”等插入语）/ 确认 / 取消",
        "music": "下一首 / 播放下一首 / 我喜欢这首 / 这首不喜欢 / 我准备学习 / 播放Lo-fi / 取消当前歌曲选择",
        "memo": "我今天有什么安排 / 我明天有什么安排 / 后天有什么安排 / 查看所有日程 / 不要提醒",
    }.get(scene or "message", "给X发消息，内容")
    prompt_data = {
        "raw_text": _safe_text(text, 200),
        "scene": _safe_text(scene, 30),
        "allowed_examples": canonical,
    }
    system = (
        "你是端侧助手的语音指令整理模块。语音识别可能有近音错误（如“所以”≈“所有”、"
        "“下一收”≈“下一首”），也可能带语气词、重复、口误或“让他/告诉她/跟他说/说”等插入语。"
        "请把转写校正为最接近的一条规范指令：纠正近音词、去掉祈使性插入语，"
        "只按 allowed_examples 的句式改写，不得新增人物、消息内容、曲目或操作语义；"
        "若无法整理则输出空字符串。不输出思考过程、Markdown 或额外文本。"
        "只输出紧凑 JSON：{\"r\":\"规范指令，最多40字\"}。"
    )
    result = _call_local_model(system, prompt_data, num_predict=24)
    if not result:
        return None
    cleaned = str(result.get("response") or "").strip()
    return cleaned or None


def suggest_local_actions(*, scene, profile, history, allowed_actions, hint_ids=None):
    """页面打开后的异步建议；模型只能在后端白名单中选 ID，前端仍要求用户点击。"""
    allowed_ids = [str(item["id"]) for item in allowed_actions]
    prompt_data = {
        "scene": _safe_text(scene, 30),
        "profile": {
            "response_style": _safe_text(profile.get("response_style"), 30),
        },
        "history_summary": history,
        "allowed_action_ids": allowed_ids,
        # 规则侧已按历史动作映射出的候选，作为模型的强提示；模型仍不能超出白名单。
        "hint_ids": [str(action_id) for action_id in (hint_ids or []) if str(action_id) in allowed_ids][:2],
    }
    system = (
        "你是端侧助手的页面建议模块。只根据 history_summary 和 profile 推荐 0 到 2 个可能操作，"
        "优先考虑 hint_ids 中的候选。不得猜测用户意图、不得执行操作、不得输出 allowed_action_ids 以外的 ID。"
        "不输出思考过程、Markdown 或额外文本。只输出 JSON：{\"r\":\"最多16字\",\"c\":\"\",\"p\":\"\",\"a\":[\"允许ID\"]}。"
    )
    result = _call_local_model(system, prompt_data, num_predict=28)
    if not result:
        return None
    result["actions"] = [action_id for action_id in result["actions"] if action_id in allowed_ids]
    return result
