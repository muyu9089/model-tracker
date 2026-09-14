import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import Settings
from .mcp_client import AIHotMCPClient


class ReleasedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=2, max_length=200)
    source_item_id: str = Field(min_length=1)


class ReleaseExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ReleasedModel]


LATEST_TOOL = {
    "type": "function",
    "function": {
        "name": "aihot_get_latest",
        "description": "获取 AIHOT 过去 24 小时或最近 7 天的精选或全部 AI 资讯。",
        "parameters": {
            "type": "object",
            "properties": {
                "window": {"type": "string", "enum": ["24h", "7d"]},
                "mode": {"type": "string", "enum": ["selected", "all"]},
                "category": {
                    "type": "string",
                    "enum": ["ai-models", "ai-products", "industry", "paper", "tip"],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            "required": ["window", "mode", "category", "limit"],
            "additionalProperties": False,
        },
    },
}


def _message(response: dict[str, Any]) -> dict[str, Any]:
    try:
        return response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Qwen 返回结构中缺少 choices[0].message") from exc


def _json_content(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Qwen 未返回 JSON 内容")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Qwen 返回的内容不是合法 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Qwen 返回的 JSON 顶层必须是对象")
    return value


@dataclass
class QwenAgent:
    settings: Settings

    async def _completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        if not self.settings.qwen_api_key:
            raise RuntimeError("缺少 QWEN_API_KEY，无法运行模型追踪任务")
        payload: dict[str, Any] = {
            "model": self.settings.qwen_model,
            "messages": messages,
            "temperature": self.settings.qwen_temperature,
            "enable_thinking": self.settings.qwen_enable_thinking,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.settings.qwen_api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.settings.qwen_base_url.rstrip('/')}/chat/completions"
        timeout = httpx.Timeout(self.settings.qwen_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            for attempt in range(self.settings.qwen_request_attempts):
                response = await client.post(endpoint, headers=headers, json=payload)
                if response.status_code not in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    return response.json()
                if attempt + 1 == self.settings.qwen_request_attempts:
                    response.raise_for_status()
                await asyncio.sleep(min(2**attempt, 4))
        raise RuntimeError("Qwen 请求未返回结果")

    async def discover_model_releases(self, mcp_client: AIHotMCPClient) -> list[dict[str, Any]]:
        """让 Qwen 调用 AIHOT MCP，并从真实返回项中抽取正式发布的模型。"""
        system = (
            "你是模型发布信息抽取代理。必须先调用提供的 AIHOT 工具获取最近 7 天资讯。"
            "随后只提取已经正式发布、正式上线、正式开放 API 或正式开源权重的新模型名称。"
            "只关注 DeepSeek、Qwen、GPT、Claude、Gemini、Grok、Muse Spark、GLM、Kimi 系列模型。"
            "只提取资讯中明确出现的具体模型型号，例如 GPT-6 Astra、Claude Fable 5、Kimi K3；"
            "不得输出 GPT、Claude、Kimi、Qwen、Gemini 或‘某某系列’等家族名。"
            "如果资讯只说明模型家族更新但没有给出具体新型号，则忽略该资讯。"
            "排除传闻、预告、评测对比、价格变化、融资、旧模型部署和仅发布产品而未发布模型的资讯。"
            "模型名称尽量保持 AIHOT 原文写法，不得补充、改写或猜测工具结果中不存在的型号。"
        )
        user = (
            "调用 aihot_get_latest，参数必须为 window=7d、mode=all、category=ai-models、limit=30；"
            "然后输出 JSON：{\"models\":[{\"model_name\":\"正式型号\","
            "\"source_item_id\":\"AIHOT条目id\"}]}。同一条资讯包含多个新模型时分别列出。"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        first = _message(
            await self._completion(
                messages,
                tools=[LATEST_TOOL],
                tool_choice={"type": "function", "function": {"name": "aihot_get_latest"}},
            )
        )
        tool_calls = first.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise RuntimeError("Qwen 没有按要求调用一次 aihot_get_latest")
        call = tool_calls[0]
        function = call.get("function") or {}
        if function.get("name") != "aihot_get_latest":
            raise RuntimeError("Qwen 调用了非预期工具")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Qwen 生成的 MCP 参数不是合法 JSON") from exc
        expected = {"window": "7d", "mode": "all", "category": "ai-models", "limit": 30}
        if arguments != expected:
            raise RuntimeError(f"Qwen 生成的 MCP 参数不符合要求：{arguments}")

        result = await mcp_client.call_tool("aihot_get_latest", arguments)
        if result.isError:
            raise RuntimeError("aihot_get_latest 返回错误")
        structured = result.structuredContent or {}
        items = structured.get("items")
        if not isinstance(items, list):
            raise RuntimeError("aihot_get_latest 返回结构中缺少 items 数组")

        messages.append(first)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id"),
                "content": json.dumps(result.model_dump(mode="json", exclude_none=True), ensure_ascii=False),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": "根据工具结果完成抽取。只输出约定的 JSON 对象，不要输出解释或 Markdown。",
            }
        )
        second = _message(await self._completion(messages, json_mode=True))
        try:
            extracted = ReleaseExtraction.model_validate(_json_content(second))
        except ValidationError as exc:
            raise RuntimeError(f"Qwen 模型发布抽取结果不符合 Schema：{exc}") from exc

        item_by_id = {
            str(item.get("id")): item
            for item in items
            if isinstance(item, dict) and item.get("id") is not None
        }
        releases: list[dict[str, Any]] = []
        for model in extracted.models:
            item = item_by_id.get(model.source_item_id)
            if item is None:
                raise RuntimeError(f"Qwen 引用了不存在的 AIHOT 条目：{model.source_item_id}")
            releases.append({"name": model.model_name.strip(), "item": item})
        return releases
