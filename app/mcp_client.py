from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, ListToolsResult

from .config import Settings


class AIHotMCPClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        timeout = httpx.Timeout(self._settings.mcp_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with streamable_http_client(
                self._settings.aihot_mcp_url,
                http_client=client,
                terminate_on_close=True,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    with anyio.fail_after(self._settings.mcp_timeout_seconds):
                        await session.initialize()
                    yield session

    async def list_tools(self) -> ListToolsResult:
        async with self._session() as session:
            with anyio.fail_after(self._settings.mcp_timeout_seconds):
                return await session.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        async with self._session() as session:
            with anyio.fail_after(self._settings.mcp_timeout_seconds):
                return await session.call_tool(name, arguments=arguments)
