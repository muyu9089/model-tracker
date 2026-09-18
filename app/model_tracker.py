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
from pydantic import ValidationError

from .artificial_analysis import ArtificialAnalysisClient, parse_index_update_text
from .config import get_settings
from .mcp_client import AIHotMCPClient
from .qwen_agent import ModelInfo, QwenAgent

LOGGER = logging.getLogger("aihot.model_tracker")
MODEL_STATUS_FILENAME = "存量模型信息记录状态.json"
ALL_MODEL_DATA_FILENAME = "all_model_data.json"
TAU_BANKING_DATA_FILENAME = "tau3_banking_model_data.json"
TERMINALBENCH_DATA_FILENAME = "terminalbench_4-0_model_data.json"
MODEL_INFO_FILENAME = "model_info.json"
PENDING = "待更新"
RECORDED = "已记录"
METRIC_FIELDS = (
    "intelligence_index",
    "output_tokens_per_task",
    "time_per_task_minutes",
)
STATUS_LAYERS = ("intelligence_index", "terminalbench-4-0", "tau3-banking")

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


def load_status_table(path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    table = {"intelligence_index": raw, "tau3-banking": []} if isinstance(raw, list) else raw
    for layer in STATUS_LAYERS:
        table.setdefault(layer, [])
    names = {}
    for layer in STATUS_LAYERS:
        for record in table[layer]:
            name = str(record.get("model_name", ""))
            key = normalize_name(name)
            if key:
                names.setdefault(key, name)
    for layer in STATUS_LAYERS:
        for record in table[layer]:
            key = normalize_name(str(record.get("model_name", "")))
            if key:
                record["model_name"] = names[key]
        known = {normalize_name(str(record.get("model_name", ""))) for record in table[layer]}
        for key, name in names.items():
            if key not in known:
                table[layer].append({"model_name": name, "record_status": PENDING})
    return table


async def collect_model_info(
    names: list[str],
    output_path: Path,
    qwen_agent: QwenAgent,
) -> list[dict[str, Any]]:
    """Collect model info only for names discovered by the current scan."""
    existing: list[dict[str, Any]] = []
    if output_path.exists():
        value = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("model_info.json 顶层必须是数组")
        for record in value:
            try:
                existing.append(ModelInfo.model_validate(record).model_dump())
            except (TypeError, ValidationError):
                continue
    by_name = {normalize_name(record["model_name"]): record for record in existing}
    result = list(existing)
    failures: list[str] = []
    for name in names:
        key = normalize_name(name)
        if key in by_name:
            continue
        try:
            by_name[key] = await qwen_agent.research_model_info(name)
            result.append(by_name[key])
            atomic_write_json(output_path, result)
            LOGGER.info("模型资料已记录：%s", name)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            LOGGER.error("模型资料采集失败：%s：%s", name, exc)
    atomic_write_json(output_path, result)
    if failures:
        raise RuntimeError(f"{len(failures)} 个模型采集失败，可重新运行续采：" + "；".join(failures))
    return result


async def update_model_status_table(
    path: Path,
    new_names: list[str],
    aa_client: ArtificialAnalysisClient | None,
    all_model_data_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    status_table = load_status_table(path)
    records = status_table["intelligence_index"]
    all_model_data_path = all_model_data_path or Path("data", ALL_MODEL_DATA_FILENAME).resolve()
    all_model_data: list[dict[str, Any]] = (
        json.loads(all_model_data_path.read_text(encoding="utf-8"))
        if all_model_data_path.exists()
        else []
    )
    data_by_name = {
        normalize_name(str(record.get("model_name", ""))): record
        for record in all_model_data
    }
    cleaned_records: list[dict[str, Any]] = []
    for record in records:
        name = str(record.get("model_name", ""))
        cleaned = {
            "model_name": name,
            "record_status": record.get("record_status", PENDING),
        }
        if record.get("update_error"):
            cleaned["update_error"] = record["update_error"]
        legacy_metrics = {field: record.get(field) for field in METRIC_FIELDS if field in record}
        if legacy_metrics:
            key = normalize_name(name)
            model_data = data_by_name.get(key)
            if model_data is None:
                model_data = {"model_name": name}
                all_model_data.append(model_data)
                data_by_name[key] = model_data
            model_data.update(legacy_metrics)
        cleaned_records.append(cleaned)
    records = cleaned_records
    status_table["intelligence_index"] = records
    known = {normalize_name(str(record.get("model_name", ""))) for record in records}
    for name in new_names:
        key = normalize_name(name)
        if key and key not in known:
            for layer in STATUS_LAYERS:
                status_table[layer].append({"model_name": name, "record_status": PENDING})
            known.add(key)

    if aa_client is None:
        atomic_write_json(path, status_table)
        atomic_write_json(all_model_data_path, all_model_data)
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
            atomic_write_json(path, status_table)
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
                    key = normalize_name(name)
                    model_data = data_by_name.get(key)
                    if model_data is None:
                        model_data = {"model_name": name}
                        all_model_data.append(model_data)
                        data_by_name[key] = model_data
                    model_data.update({field: metric.get(field) for field in METRIC_FIELDS})
                    record["record_status"] = RECORDED
                    record.pop("update_error", None)
                else:
                    record["update_error"] = metric.get("error") or "未匹配到 Artificial Analysis 模型"
    except Exception as exc:
        LOGGER.warning("Artificial Analysis 状态更新失败：%s", exc)
    atomic_write_json(path, status_table)
    atomic_write_json(all_model_data_path, all_model_data)
    return metrics


async def update_tau_banking_status_table(
    path: Path,
    aa_client: ArtificialAnalysisClient,
    data_path: Path,
) -> dict[str, dict[str, Any]]:
    status_table = load_status_table(path)
    records = status_table["tau3-banking"]
    try:
        html = await aa_client.fetch_tau_banking_html()
        metrics = await aa_client.enrich_tau_banking_models(
            [str(record["model_name"]) for record in records], html
        )
    except Exception as exc:
        LOGGER.warning("𝜏³-Banking 状态更新失败：%s", exc)
        return {}
    model_data = []
    for record in records:
        name = str(record["model_name"])
        metric = metrics.get(name) or {}
        if not metric.get("matched"):
            record["record_status"] = PENDING
            record["update_error"] = metric.get("error") or "未获取到 𝜏³-Banking 指标"
            continue
        item = {"model_name": name}
        item.update({field: metric[field] for field in (
            "tau3_banking_score", "output_tokens_per_task", "time_per_task_minutes"
        )})
        model_data.append(item)
        record["record_status"] = RECORDED
        record.pop("update_error", None)
    atomic_write_json(path, status_table)
    atomic_write_json(data_path, model_data)
    return metrics


async def update_terminalbench_status_table(
    path: Path,
    aa_client: ArtificialAnalysisClient,
    data_path: Path,
) -> dict[str, dict[str, Any]]:
    status_table = load_status_table(path)
    records = status_table["terminalbench-4-0"]
    try:
        html = await aa_client.fetch_terminalbench_html()
        metrics = await aa_client.enrich_terminalbench_models(
            [str(record["model_name"]) for record in records], html
        )
    except Exception as exc:
        LOGGER.warning("Terminal-Bench 4.0 状态更新失败：%s", exc)
        return {}
    model_data = []
    for record in records:
        name = str(record["model_name"])
        metric = metrics.get(name) or {}
        if not metric.get("matched"):
            record["record_status"] = PENDING
            record["update_error"] = metric.get("error") or "未获取到 Terminal-Bench 4.0 指标"
            continue
        item = {"model_name": name}
        item.update({field: metric[field] for field in (
            "terminalbench_4_0_score", "output_tokens_per_task", "time_per_task_minutes"
        )})
        model_data.append(item)
        record["record_status"] = RECORDED
        record.pop("update_error", None)
    atomic_write_json(path, status_table)
    atomic_write_json(data_path, model_data)
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


async def scan_once(
    data_dir: Path,
    status_path: Path | None = None,
    model_info_path: Path | None = None,
) -> list[dict[str, Any]]:
    timezone = ZoneInfo("Asia/Shanghai")
    now = datetime.now(timezone)
    status_path = status_path or Path(MODEL_STATUS_FILENAME).resolve()
    model_info_path = model_info_path or Path(MODEL_INFO_FILENAME).resolve()
    status_table = load_status_table(status_path)
    known_models = {
        normalize_name(str(record.get("model_name", "")))
        for layer in STATUS_LAYERS
        for record in status_table[layer]
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

    if new_models:
        await collect_model_info(
            [model["name"] for model in new_models],
            model_info_path,
            qwen_agent,
        )

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
        data_dir / ALL_MODEL_DATA_FILENAME,
    )
    if aa_client is not None:
        await update_terminalbench_status_table(
            status_path, aa_client, data_dir / TERMINALBENCH_DATA_FILENAME
        )
        await update_tau_banking_status_table(
            status_path, aa_client, data_dir / TAU_BANKING_DATA_FILENAME
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


async def run_daemon(data_dir: Path, status_path: Path, model_info_path: Path) -> None:
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
        args=[data_dir, status_path, model_info_path],
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
    mode.add_argument("--tau3-banking", action="store_true", help="仅更新 𝜏³-Banking 指标")
    mode.add_argument("--terminalbench-4-0", action="store_true", help="仅更新 Terminal-Bench 4.0 指标")
    parser.add_argument("--data-dir", default="data", help="状态和报告保存目录")
    parser.add_argument("--status-table", default=MODEL_STATUS_FILENAME, help="模型信息状态表路径")
    parser.add_argument("--model-info-output", default=MODEL_INFO_FILENAME, help="模型资料 JSON 输出路径")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_dir = Path(args.data_dir).resolve()
    status_path = Path(args.status_table).resolve()
    model_info_path = Path(args.model_info_output).resolve()
    if args.tau3_banking or args.terminalbench_4_0:
        settings = get_settings()
        client = ArtificialAnalysisClient(
            base_url=settings.artificial_analysis_base_url,
            timeout_seconds=settings.artificial_analysis_timeout_seconds,
        )
        if args.tau3_banking:
            asyncio.run(update_tau_banking_status_table(
                status_path, client, data_dir / TAU_BANKING_DATA_FILENAME
            ))
        else:
            asyncio.run(update_terminalbench_status_table(
                status_path, client, data_dir / TERMINALBENCH_DATA_FILENAME
            ))
    elif args.once:
        asyncio.run(scan_once(data_dir, status_path, model_info_path))
    else:
        asyncio.run(run_daemon(data_dir, status_path, model_info_path))


if __name__ == "__main__":
    main()
