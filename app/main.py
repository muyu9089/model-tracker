from datetime import date
from html import escape
from pathlib import Path
from secrets import compare_digest
from typing import Any, Literal

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .db import TrackerRepository
from .db.repositories import PROFILE_FIELDS
from .mcp_client import AIHotMCPClient

app = FastAPI(title="AIHot MCP Gateway", version="1.0.0")

DATABASE_TABLES = {
    "models": "模型",
    "model_profiles": "模型详情",
    "benchmarks": "评测项",
    "benchmark_states": "评测状态",
    "benchmark_results": "评测结果",
    "scan_runs": "扫描记录",
    "scan_run_models": "扫描模型",
    "source_versions": "数据源版本",
}


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


def get_repository(
    request: Request, settings: Settings = Depends(get_settings)
) -> TrackerRepository:
    path = getattr(request.app.state, "tracker_database_path", settings.tracker_database_path)
    return TrackerRepository(Path(path).resolve())


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


def _cell(value: Any) -> str:
    text = "—" if value is None else str(value)
    safe = escape(text)
    if len(text) <= 160:
        return safe
    return f"<details><summary>{escape(text[:80])}…</summary><pre>{safe}</pre></details>"


def _database_page(repository: TrackerRepository, table: str) -> str:
    counts = {
        name: repository.rows(f'SELECT COUNT(*) AS count FROM "{name}"')[0]["count"]
        for name in DATABASE_TABLES
    }
    tabs = f'<a class="{"active" if table == "overview" else ""}" href="/?table=overview">模型总览<small>{counts["models"]}</small></a>' + "".join(
        f'<a class="{"active" if name == table else ""}" href="/?table={name}">'
        f'{label}<small>{counts[name]}</small></a>'
        for name, label in DATABASE_TABLES.items()
    )
    if table == "overview":
        models = repository.rows(
            """SELECT m.id, m.model_name, p.vendor_or_organization,
                      p.open_source_attribute, p.region
               FROM models m LEFT JOIN model_profiles p ON p.model_id = m.id
               ORDER BY m.id"""
        )
        benchmarks = repository.rows("SELECT id, display_name FROM benchmarks ORDER BY id")
        scores = {
            (row["model_id"], row["benchmark_id"]): row
            for row in repository.rows(
                "SELECT model_id, benchmark_id, score_value, score_unit FROM benchmark_results"
            )
        }
        columns = ("模型名称", "评测项", "评测结果", "所属组织", "开源状态", "所属地域")
        items = "".join(f"<div>{_cell(item['display_name'])}</div>" for item in benchmarks)
        body_rows = []
        for model in models:
            results = []
            for item in benchmarks:
                score = scores.get((model["id"], item["id"]))
                value = "—" if not score or score["score_value"] is None else f"{score['score_value']:g}"
                if score and score["score_value"] is not None and score["score_unit"] == "percent":
                    value += "%"
                results.append(f"<div>{_cell(value)}</div>")
            body_rows.append("<tr>" + "".join(
                f"<td>{value}</td>" for value in (
                    _cell(model["model_name"]), items, "".join(results),
                    _cell(model["vendor_or_organization"]),
                    _cell(model["open_source_attribute"]), _cell(model["region"]),
                )
            ) + "</tr>")
        body = "".join(body_rows)
    else:
        columns = [row["name"] for row in repository.rows(f'PRAGMA table_info("{table}")')]
        rows = repository.rows(f'SELECT * FROM "{table}" ORDER BY rowid DESC LIMIT 200')
        body = "".join(
            "<tr>" + "".join(f"<td>{_cell(row[column])}</td>" for column in columns) + "</tr>"
            for row in rows
        )
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = body or f'<tr><td class="empty" colspan="{len(columns)}">暂无数据</td></tr>'
    title = "模型总览" if table == "overview" else DATABASE_TABLES[table]
    subtitle = "全部模型 · 只读" if table == "overview" else "最新 200 行 · 只读"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>模型数据库</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7fb;color:#182230;font:14px system-ui,sans-serif}}
main{{max-width:1600px;margin:auto;padding:32px}}h1{{margin:0 0 6px;font-size:26px}}p{{margin:0 0 24px;color:#667085}}
nav{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}}nav a{{display:flex;gap:8px;padding:9px 12px;border:1px solid #d0d5dd;border-radius:8px;background:white;color:#344054;text-decoration:none}}
nav a.active{{background:#155eef;color:white;border-color:#155eef}}small{{opacity:.75}}section{{overflow:auto;border:1px solid #e4e7ec;border-radius:10px;background:white}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:11px 14px;border-bottom:1px solid #eaecf0;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#f9fafb;color:#475467;font-size:12px}}
td{{max-width:420px;overflow:hidden;text-overflow:ellipsis}}tr:last-child td{{border-bottom:0}}tr:hover td{{background:#f9fafb}}details{{white-space:normal}}summary{{cursor:pointer}}pre{{max-width:800px;white-space:pre-wrap;word-break:break-word}}.empty{{padding:40px;text-align:center;color:#98a2b3}}
</style></head><body><main><h1>模型数据库</h1><p>{title} · {subtitle}</p><nav aria-label="数据菜单">{tabs}</nav>
<section><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></section></main></body></html>"""


def _models_updated_on(repository: TrackerRepository, updated_date: date) -> list[dict[str, Any]]:
    profile_columns = ", ".join(f"p.{field}" for field in PROFILE_FIELDS)
    rows = repository.rows(
        f"""SELECT m.id, m.model_name, m.first_seen_at, m.published_at, m.title,
                   m.source_url, m.aihot_url, m.aihot_id, m.created_at, m.updated_at,
                   {profile_columns}, p.collected_at AS profile_collected_at
            FROM models m LEFT JOIN model_profiles p ON p.model_id=m.id
            WHERE substr(m.updated_at, 1, 10)=? ORDER BY m.id""",
        (updated_date.isoformat(),),
    )
    models = {}
    for row in rows:
        profile = {field: row[field] for field in PROFILE_FIELDS}
        if row["profile_collected_at"] is not None:
            profile["collected_at"] = row["profile_collected_at"]
        models[row["id"]] = {
            key: row[key]
            for key in (
                "id", "model_name", "first_seen_at", "published_at", "title",
                "source_url", "aihot_url", "aihot_id", "created_at", "updated_at",
            )
        } | {
            "profile": profile if row["profile_collected_at"] is not None else None,
            "benchmarks": {},
        }

    if not models:
        return []

    benchmark_rows = repository.rows(
        """SELECT m.id AS model_id, b.code, b.display_name, s.status, s.update_error,
                  s.updated_at, r.score_value, r.score_unit, r.output_tokens, r.time_seconds,
                  r.matched_model_name, r.matched_slug, r.matched_parameters_b,
                  r.matched_reasoning_depth, r.match_method,
                  r.source_url, r.collected_at
           FROM models m JOIN benchmark_states s ON s.model_id=m.id
           JOIN benchmarks b ON b.id=s.benchmark_id
           LEFT JOIN benchmark_results r
             ON r.model_id=m.id AND r.benchmark_id=b.id
           WHERE substr(m.updated_at, 1, 10)=? ORDER BY m.id, b.id""",
        (updated_date.isoformat(),),
    )
    for row in benchmark_rows:
        models[row["model_id"]]["benchmarks"][row["code"]] = {
            key: row[key]
            for key in (
                "display_name", "status", "update_error", "updated_at", "score_value",
                "score_unit", "output_tokens", "time_seconds", "matched_model_name",
                "matched_slug", "matched_parameters_b", "matched_reasoning_depth",
                "match_method", "source_url", "collected_at",
            )
        }
    return list(models.values())


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def database_view(
    table: str = Query(default="models"),
    repository: TrackerRepository = Depends(get_repository),
) -> HTMLResponse:
    if table != "overview" and table not in DATABASE_TABLES:
        raise HTTPException(status_code=404, detail="数据表不存在")
    return HTMLResponse(_database_page(repository, table))


@app.get("/models")
def models_by_updated_date(
    updated_date: date = Query(alias="date", description="models.updated_at 的 UTC 日期"),
    repository: TrackerRepository = Depends(get_repository),
) -> dict[str, Any]:
    models = _models_updated_on(repository, updated_date)
    return {"date": updated_date.isoformat(), "count": len(models), "models": models}


@app.post("/models/scan")
async def scan_models(
    request: Request,
    x_scan_token: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if not settings.tracker_scan_token:
        raise HTTPException(status_code=503, detail="未配置 TRACKER_SCAN_TOKEN")
    if not compare_digest(x_scan_token.encode(), settings.tracker_scan_token.encode()):
        raise HTTPException(status_code=401, detail="无效的扫描令牌")
    from .model_tracker import SCAN_LOCK, scan_once

    if SCAN_LOCK.locked():
        raise HTTPException(status_code=409, detail="扫描正在进行中")
    args = getattr(request.app.state, "tracker_scan_args", (Path("data").resolve(),))
    models = await scan_once(*args)
    return {"count": len(models), "models": models}


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
