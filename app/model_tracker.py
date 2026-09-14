import argparse
import asyncio
import json
import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .artificial_analysis import ArtificialAnalysisClient, parse_index_update_text
from .config import get_settings
from .mcp_client import AIHotMCPClient
from .qwen_agent import QwenAgent

LOGGER = logging.getLogger("aihot.model_tracker")
MODEL_STATUS_FILENAME = "存量模型信息记录状态.json"
PENDING = "待更新"
RECORDED = "已记录"
METRIC_FIELDS = (
    "intelligence_index",
    "output_tokens_per_task",
    "time_per_task_minutes",
)

def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).casefold()
    return "".join(character for character in normalized if character.isalnum())


def load_seen(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


async def update_model_status_table(
    path: Path,
    new_names: list[str],
    aa_client: ArtificialAnalysisClient | None,
) -> dict[str, dict[str, Any]]:
    records: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    known = {normalize_name(str(record.get("model_name", ""))) for record in records}
    for name in new_names:
        key = normalize_name(name)
        if key and key not in known:
            records.append({"model_name": name, "record_status": PENDING})
            known.add(key)

    if aa_client is None:
        atomic_write_json(path, records)
        return {}

    metrics: dict[str, dict[str, Any]] = {}
    try:
        models_html = await aa_client.fetch_models_html()
        current_text = parse_index_update_text(models_html)
        text_path = path.with_name(f"{path.stem}.intelligence-index.txt")
        previous_text = text_path.read_text(encoding="utf-8") if text_path.exists() else None
        if current_text is not None and current_text != previous_text:
            for record in records:
                record["record_status"] = PENDING
            atomic_write_json(path, records)
            atomic_write_text(text_path, current_text)

        pending_names = [
            str(record["model_name"])
            for record in records
            if record.get("record_status") == PENDING
        ]
        if pending_names:
            metrics = await aa_client.enrich_models(pending_names, models_html)
            for record in records:
                name = str(record.get("model_name", ""))
                metric = metrics.get(name)
                if record.get("record_status") != PENDING or not metric:
                    continue
                if metric.get("matched"):
                    record.update({field: metric.get(field) for field in METRIC_FIELDS})
                    record["record_status"] = RECORDED
                    record.pop("update_error", None)
                else:
                    record["update_error"] = metric.get("error") or "未匹配到 Artificial Analysis 模型"
    except Exception as exc:
        LOGGER.warning("Artificial Analysis 状态更新失败：%s", exc)
    atomic_write_json(path, records)
    return metrics


def _display_metric(value: Any, digits: int = 0) -> str:
    if isinstance(value, str) and value:
        return value
    return f"{value:,.{digits}f}" if isinstance(value, (int, float)) else "未获取"


def write_markdown(path: Path, run_at: str, models: list[dict[str, Any]]) -> None:
    lines = [f"# 最近七天新增模型（{run_at}）", ""]
    if not models:
        lines.append("本周期未发现新增模型。")
    else:
        lines.extend(
            [
                f"共发现 {len(models)} 个此前未记录的模型。",
                "",
                "| 模型名称 | AA 匹配型号 | Intelligence Index | Output Tokens / Task | Time / Task | 发布时间 | 相关资讯 | 来源 |",
                "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for model in models:
            title = str(model["title"]).replace("|", "\\|")
            aa = model.get("artificial_analysis") or {}
            aa_name = str(aa.get("model_name") or "未匹配").replace("|", "\\|")
            lines.append(
                f"| {model['name']} | {aa_name} | "
                f"{_display_metric(aa.get('intelligence_index'))} | "
                f"{_display_metric(aa.get('output_tokens_per_task'))} | "
                f"{_display_metric(aa.get('time_per_task_minutes'), 1)} 分钟 | "
                f"{model['published_at']} | {title} | "
                f"[原文]({model['source_url']}) |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def scan_once(data_dir: Path, status_path: Path | None = None) -> list[dict[str, Any]]:
    timezone = ZoneInfo("Asia/Shanghai")
    now = datetime.now(timezone)
    status_path = status_path or Path(MODEL_STATUS_FILENAME).resolve()
    status_records = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else []
    known_models = {
        normalize_name(str(record.get("model_name", "")))
        for record in status_records
    }
    seen_path = data_dir / "seen_models.json"
    seen = load_seen(seen_path)
    new_models: list[dict[str, Any]] = []
    settings = get_settings()
    mcp_client = AIHotMCPClient(settings)
    qwen_agent = QwenAgent(settings)
    for release in await qwen_agent.discover_model_releases(mcp_client):
        name = release["name"]
        item = release["item"]
        key = normalize_name(name)
        if not key or key in known_models:
            continue
        known_models.add(key)
        links = item.get("links") if isinstance(item.get("links"), dict) else {}
        record = {
            "name": name,
            "first_seen_at": now.isoformat(),
            "published_at": item.get("publishedAt", ""),
            "title": item.get("title", ""),
            "source_url": links.get("original") or links.get("aihot", ""),
            "aihot_url": links.get("aihot", ""),
            "aihot_id": item.get("id", ""),
        }
        seen[key] = record
        new_models.append(record)

    aa_client = None
    if settings.artificial_analysis_enabled:
        aa_client = ArtificialAnalysisClient(
            base_url=settings.artificial_analysis_base_url,
            timeout_seconds=settings.artificial_analysis_timeout_seconds,
        )
    metrics = await update_model_status_table(
        status_path,
        [model["name"] for model in new_models],
        aa_client,
    )
    for model in new_models:
        if model["name"] in metrics:
            model["artificial_analysis"] = metrics[model["name"]]

    atomic_write_json(seen_path, seen)
    report_stem = f"weekly_models_{now:%Y%m%d_%H%M%S}"
    atomic_write_json(data_dir / "reports" / f"{report_stem}.json", new_models)
    write_markdown(
        data_dir / "reports" / f"{report_stem}.md",
        now.strftime("%Y-%m-%d %H:%M %Z"),
        new_models,
    )
    LOGGER.info("扫描完成，本次新增模型数：%d", len(new_models))
    for model in new_models:
        LOGGER.info("新增模型：%s", model["name"])
    return new_models


async def run_daemon(data_dir: Path, status_path: Path) -> None:
    settings = get_settings()
    scheduler = AsyncIOScheduler(timezone=settings.tracker_timezone)
    scheduler.add_job(
        scan_once,
        CronTrigger(
            day_of_week=settings.tracker_day_of_week,
            hour=settings.tracker_hour,
            minute=settings.tracker_minute,
            timezone=settings.tracker_timezone,
        ),
        args=[data_dir, status_path],
        id="weekly-model-scan",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    LOGGER.info(
        "定时器已启动：每周 %s %02d:%02d（%s）",
        settings.tracker_day_of_week,
        settings.tracker_hour,
        settings.tracker_minute,
        settings.tracker_timezone,
    )
    await asyncio.Event().wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="每周从 AIHot 提取新增模型名称")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="立即执行一次后退出")
    mode.add_argument("--daemon", action="store_true", help="启动常驻定时任务")
    parser.add_argument("--data-dir", default="data", help="状态和报告保存目录")
    parser.add_argument("--status-table", default=MODEL_STATUS_FILENAME, help="模型信息状态表路径")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_dir = Path(args.data_dir).resolve()
    status_path = Path(args.status_table).resolve()
    if args.once:
        asyncio.run(scan_once(data_dir, status_path))
    else:
        asyncio.run(run_daemon(data_dir, status_path))


if __name__ == "__main__":
    main()
