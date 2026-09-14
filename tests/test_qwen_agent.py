import json
import unittest
from types import SimpleNamespace
from typing import Any

from app.qwen_agent import QwenAgent


class FakeToolResult:
    isError = False

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.structuredContent = {"items": items}

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"structuredContent": self.structuredContent, "isError": False, "content": []}


class FakeMCPClient:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeToolResult:
        self.calls.append((name, arguments))
        return FakeToolResult(self.items)


class StubQwenAgent(QwenAgent):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(SimpleNamespace())
        self.responses = responses

    async def _completion(self, *_: Any, **__: Any) -> dict[str, Any]:
        return self.responses.pop(0)


def completion(message: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"message": message}]}


class QwenAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_calls_latest_and_uses_real_item(self) -> None:
        arguments = {"window": "7d", "mode": "all", "category": "ai-models", "limit": 30}
        agent = StubQwenAgent(
            [
                completion(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "aihot_get_latest",
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    }
                ),
                completion(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"models": [{"model_name": "Demo-2", "source_item_id": "item-1"}]}
                        ),
                    }
                ),
            ]
        )
        item = {"id": "item-1", "title": "Demo 发布 Demo-2"}
        mcp = FakeMCPClient([item])

        releases = await agent.discover_model_releases(mcp)

        self.assertEqual(mcp.calls, [("aihot_get_latest", arguments)])
        self.assertEqual(releases, [{"name": "Demo-2", "item": item}])

    async def test_discovery_rejects_hallucinated_item_id(self) -> None:
        arguments = {"window": "7d", "mode": "all", "category": "ai-models", "limit": 30}
        agent = StubQwenAgent(
            [
                completion(
                    {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "aihot_get_latest",
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ]
                    }
                ),
                completion(
                    {
                        "content": json.dumps(
                            {"models": [{"model_name": "Invented-1", "source_item_id": "missing"}]}
                        )
                    }
                ),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "不存在的 AIHOT 条目"):
            await agent.discover_model_releases(FakeMCPClient([]))

if __name__ == "__main__":
    unittest.main()
