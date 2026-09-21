import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import uvicorn

from .artificial_analysis import ArtificialAnalysisClient, parse_index_update_text
from .config import Settings, get_settings
from .db import TrackerRepository
from .db.repositories import normalize_name
from .exporters import JsonExporter
from .mcp_client import AIHotMCPClient
from .qwen_agent import ModelInfo, QwenAgent


LOGGER = logging.getLogger("aihot.model_tracker")
MODEL_STATUS_FILENAME = "存量模型信息记录状态.json"
MODEL_INFO_FILENAME = "model_info.json"


async def collect_model_info(
    names: list[str], repository: TrackerRepository, qwen_agent: QwenAgent
) -> list[dict[str, Any]]:
    """Collect profiles for models newly inserted by the current scan."""
    profiles = []
    failures = []
    for name in names:
        if repository.has_profile(name):
            continue
        try:
            profile = ModelInfo.model_validate(
                await qwen_agent.research_model_info(name)
            ).model_dump()
            repository.upsert_profile(profile)
            profiles.append(profile)
            LOGGER.info("模型资料已记录：%s", name)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            LOGGER.error("模型资料采集失败：%s：%s", name, exc)
    if failures:
        raise RuntimeError(f"{len(failures)} 个新模型资料采集失败：" + "；".join(failures))
    return profiles


async def update_intelligence_index(
    repository: TrackerRepository,
    aa_client: ArtificialAnalysisClient,
) -> dict[str, dict[str, Any]]:
    try:
        html = await aa_client.fetch_models_html()
        description = parse_index_update_text(html)
        if description:
            repository.record_source_version("intelligence_index", description)
        names = repository.pending_names("intelligence_index")
        if not names:
            return {}
        metrics = await aa_client.enrich_models(names, html)
        repository.save_benchmark_metrics("intelligence_index", names, metrics)
        return metrics
    except Exception as exc:
        LOGGER.warning("Artificial Analysis 状态更新失败：%s", exc)
        return {}


async def update_benchmark(
    repository: TrackerRepository,
    aa_client: ArtificialAnalysisClient,
    code: str,
) -> dict[str, dict[str, Any]]:
    names = repository.model_names()
    if not names:
        return {}
    try:
        if code == "terminalbench-4-0":
            html = await aa_client.fetch_terminalbench_html()
            metrics = await aa_client.enrich_terminalbench_models(names, html)
        elif code == "tau3-banking":
            html = await aa_client.fetch_tau_banking_html()
            metrics = await aa_client.enrich_tau_banking_models(names, html)
        else:
            raise ValueError(f"未知 benchmark：{code}")
        repository.save_benchmark_metrics(code, names, metrics)
        return metrics
    except Exception as exc:
        LOGGER.warning("%s 状态更新失败：%s", code, exc)
        return {}


def make_exporter(
    repository: TrackerRepository,
    data_dir: Path,
    status_path: Path,
    model_info_path: Path,
    export_dir: Path,
) -> JsonExporter:
    return JsonExporter(repository, data_dir, status_path, model_info_path, export_dir)


async def scan_once(
    data_dir: Path,
    database_path: Path | None = None,
    status_path: Path | None = None,
    model_info_path: Path | None = None,
    export_dir: Path | None = None,
    export_json: bool | None = None,
) -> list[dict[str, Any]]:
    settings = get_settings()
    now = datetime.now(ZoneInfo(settings.tracker_timezone))
    database_path = database_path or Path(settings.tracker_database_path).resolve()
    status_path = status_path or Path(MODEL_STATUS_FILENAME).resolve()
    model_info_path = model_info_path or Path(MODEL_INFO_FILENAME).resolve()
    export_dir = export_dir or Path(settings.tracker_export_dir).resolve()
    export_json = settings.tracker_json_export_enabled if export_json is None else export_json
    repository = TrackerRepository(database_path)
    run_id = repository.start_run()
    new_models: list[dict[str, Any]] = []

    try:
        qwen_agent = QwenAgent(settings)
        releases = await qwen_agent.discover_model_releases(AIHotMCPClient(settings))
        for release in releases:
            name = str(release["name"]).strip()
            item = release["item"]
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
            model_id, created = repository.add_model(record)
            if not created:
                continue
            repository.add_run_model(run_id, model_id, len(new_models))
            new_models.append(record)

        if new_models:
            await collect_model_info(
                [model["name"] for model in new_models], repository, qwen_agent
            )

        if settings.artificial_analysis_enabled:
            aa_client = ArtificialAnalysisClient(
                base_url=settings.artificial_analysis_base_url,
                timeout_seconds=settings.artificial_analysis_timeout_seconds,
            )
            await update_intelligence_index(repository, aa_client)
            await update_benchmark(repository, aa_client, "terminalbench-4-0")
            await update_benchmark(repository, aa_client, "tau3-banking")
        repository.finish_run(run_id, "completed", len(new_models))
    except Exception as exc:
        repository.finish_run(run_id, "failed", len(new_models), str(exc))
        raise

    if export_json:
        exporter = make_exporter(
            repository, data_dir, status_path, model_info_path, export_dir
        )
        try:
            exporter.export_all()
            exporter.export_run_report(run_id, now)
        except Exception as exc:
            LOGGER.error("数据库已提交，但 JSON/Markdown 导出失败：%s", exc)

    LOGGER.info("扫描完成，本次新增模型数：%d", len(new_models))
    return new_models


async def run_daemon(
    data_dir: Path,
    database_path: Path,
    status_path: Path,
    model_info_path: Path,
    export_dir: Path,
    export_json: bool,
) -> None:
    from .main import app

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
        args=[data_dir, database_path, status_path, model_info_path, export_dir, export_json],
        id="weekly-model-scan",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    app.state.tracker_database_path = database_path
    try:
        await uvicorn.Server(uvicorn.Config(
            app, host=settings.tracker_web_host, port=settings.tracker_web_port
        )).serve()
    finally:
        scheduler.shutdown(wait=False)
        del app.state.tracker_database_path


def _paths(args: argparse.Namespace, settings: Settings) -> tuple[Path, ...]:
    return (
        Path(args.data_dir).resolve(),
        Path(args.database or settings.tracker_database_path).resolve(),
        Path(args.status_table).resolve(),
        Path(args.model_info_output).resolve(),
        Path(args.export_dir or settings.tracker_export_dir).resolve(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 模型追踪器")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="立即执行一次扫描")
    mode.add_argument("--daemon", action="store_true", help="启动常驻定时任务")
    mode.add_argument("--tau3-banking", action="store_true", help="仅更新 Tau3-Banking")
    mode.add_argument("--terminalbench-4-0", action="store_true", help="仅更新 Terminal-Bench 4.0")
    mode.add_argument("--export-json", action="store_true", help="仅从数据库重新生成 JSON")
    mode.add_argument("--migrate-legacy", action="store_true", help="导入旧 JSON")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--database")
    parser.add_argument("--export-dir")
    parser.add_argument("--status-table", default=MODEL_STATUS_FILENAME)
    parser.add_argument("--model-info-output", default=MODEL_INFO_FILENAME)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    data_dir, database_path, status_path, model_info_path, export_dir = _paths(args, settings)
    repository = TrackerRepository(database_path)
    exporter = make_exporter(repository, data_dir, status_path, model_info_path, export_dir)

    if args.migrate_legacy:
        from .migration.import_legacy import import_legacy

        report = import_legacy(repository, data_dir, status_path, model_info_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.export_json:
        exporter.export_all()
        return
    if args.tau3_banking or args.terminalbench_4_0:
        client = ArtificialAnalysisClient(
            base_url=settings.artificial_analysis_base_url,
            timeout_seconds=settings.artificial_analysis_timeout_seconds,
        )
        code = "tau3-banking" if args.tau3_banking else "terminalbench-4-0"
        asyncio.run(update_benchmark(repository, client, code))
        if not args.no_export and settings.tracker_json_export_enabled:
            exporter.export_all()
        return
    export_json = not args.no_export and settings.tracker_json_export_enabled
    if args.once:
        asyncio.run(scan_once(
            data_dir, database_path, status_path, model_info_path, export_dir, export_json
        ))
    else:
        asyncio.run(run_daemon(
            data_dir, database_path, status_path, model_info_path, export_dir, export_json
        ))


if __name__ == "__main__":
    main()
