from datetime import date
from typing import Any, Literal

import anyio
from fastapi import Depends, FastAPI, HTTPException, Query
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .mcp_client import AIHotMCPClient

app = FastAPI(title="AIHot MCP Gateway", version="1.0.0")


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class FetchRequest(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LatestQuery(BaseModel):
    window: Literal["24h", "7d"] = "24h"
    mode: Literal["selected", "all"] = "selected"
    category: Literal["ai-models", "ai-products", "industry", "paper", "tip"] | None = None
    limit: int = Field(default=10, ge=1, le=30)


class SearchRequest(BaseModel):
    q: str = Field(min_length=2, max_length=200)
    window: Literal["24h", "7d"] = "7d"
    category: Literal["ai-models", "ai-products", "industry", "paper", "tip"] | None = None
    limit: int = Field(default=10, ge=1, le=30)


def get_client(settings: Settings = Depends(get_settings)) -> AIHotMCPClient:
    return AIHotMCPClient(settings)


def dump(value: Any) -> Any:
    return value.model_dump(mode="json", exclude_none=True)


async def invoke(client: AIHotMCPClient, tool: str, arguments: dict[str, Any]) -> Any:
    try:
        result = await client.call_tool(tool, arguments)
        if result.isError:
            raise HTTPException(status_code=502, detail=dump(result))
        return dump(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise_gateway_error(exc)


def raise_gateway_error(exc: Exception) -> None:
    if isinstance(exc, TimeoutError):
        raise HTTPException(status_code=504, detail="AIHot MCP 请求超时") from exc
    if isinstance(exc, (McpError, OSError, anyio.BrokenResourceError)):
        raise HTTPException(status_code=502, detail=f"AIHot MCP 调用失败: {exc}") from exc
    raise exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tools")
async def list_tools(client: AIHotMCPClient = Depends(get_client)) -> Any:
    try:
        result = await client.list_tools()
        return dump(result)
    except Exception as exc:
        raise_gateway_error(exc)


@app.post("/call/{tool_name}")
async def call_tool(
    tool_name: str,
    request: ToolCallRequest,
    client: AIHotMCPClient = Depends(get_client),
) -> Any:
    try:
        result = await client.call_tool(tool_name, request.arguments)
        if result.isError:
            raise HTTPException(status_code=502, detail=dump(result))
        return dump(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise_gateway_error(exc)


@app.post("/fetch")
async def fetch(
    request: FetchRequest,
    client: AIHotMCPClient = Depends(get_client),
) -> Any:
    """通用信息获取入口：指定发现到的 MCP 工具及其参数。"""
    return await invoke(client, request.tool, request.arguments)


@app.get("/news/latest")
async def latest_news(
    window: Literal["24h", "7d"] = "24h",
    mode: Literal["selected", "all"] = "selected",
    category: Literal["ai-models", "ai-products", "industry", "paper", "tip"] | None = None,
    limit: int = Query(default=10, ge=1, le=30),
    client: AIHotMCPClient = Depends(get_client),
) -> Any:
    query = LatestQuery(window=window, mode=mode, category=category, limit=limit)
    return await invoke(client, "aihot_get_latest", query.model_dump(exclude_none=True))


@app.post("/news/search")
async def search_news(
    request: SearchRequest,
    client: AIHotMCPClient = Depends(get_client),
) -> Any:
    return await invoke(client, "aihot_search", request.model_dump(exclude_none=True))


@app.get("/topics/hot")
async def hot_topics(
    limit: int = Query(default=10, ge=1, le=10),
    client: AIHotMCPClient = Depends(get_client),
) -> Any:
    return await invoke(client, "aihot_get_hot_topics", {"limit": limit})


@app.get("/stories/{public_id}")
async def story(
    public_id: str,
    report_limit: int = Query(default=20, ge=1, le=50),
    client: AIHotMCPClient = Depends(get_client),
) -> Any:
    return await invoke(
        client,
        "aihot_get_story",
        {"public_id": public_id, "report_limit": report_limit},
    )


@app.get("/daily")
async def daily_report(
    report_date: date | None = None,
    client: AIHotMCPClient = Depends(get_client),
) -> Any:
    arguments = {"date": report_date.isoformat()} if report_date else {}
    return await invoke(client, "aihot_get_daily", arguments)
