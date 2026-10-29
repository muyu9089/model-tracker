import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.db import BENCHMARKS, TrackerRepository
from app.db.repositories import normalize_name
from app.qwen_agent import ModelInfo


STATUS_MAP = {"已记录": "recorded", "待更新": "pending"}


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _metric_from_legacy(code: str, record: dict[str, Any]) -> dict[str, Any]:
    score_field = {
        "intelligence_index": "intelligence_index",
        "terminalbench-4-0": "terminalbench_4_0_score",
        "aa-briefcase": "aa_briefcase_score",
        "automationbench-aa": "automationbench_aa_score",
    }[code]
    return {
        "matched": True,
        score_field: record.get(score_field),
        "output_tokens_per_task": record.get("output_tokens_per_task"),
        "time_per_task_minutes": record.get("time_per_task_minutes"),
    }


def import_legacy(
    repository: TrackerRepository,
    data_dir: Path,
    status_path: Path,
    model_info_path: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {"models": 0, "conflicts": 0, "skipped": 0, "errors": []}
    seen = _read_json(data_dir / "seen_models.json", {})
    raw_status = _read_json(status_path, {})
    if isinstance(raw_status, list):
        raw_status = {"intelligence_index": raw_status}

    names: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for value in seen.values() if isinstance(seen, dict) else []:
        if not isinstance(value, dict):
            continue
        name = str(value.get("name") or "").strip()
        if name:
            names.setdefault(normalize_name(name), name)
            metadata[normalize_name(name)] = value
    states: dict[tuple[str, str], tuple[str, str | None]] = {}
    embedded_metrics: dict[str, dict[str, Any]] = {}
    for code in BENCHMARKS:
        records = raw_status.get(code, []) if isinstance(raw_status, dict) else []
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            name = str(record.get("model_name") or "").strip()
            key = normalize_name(name)
            if not key:
                report["skipped"] += 1
                continue
            names.setdefault(key, name)
            status = STATUS_MAP.get(str(record.get("record_status")), "pending")
            state_key = (key, code)
            previous = states.get(state_key)
            if previous and previous[0] != status:
                report["conflicts"] += 1
                status = "pending"
            states[state_key] = (status, record.get("update_error"))
            if code == "intelligence_index" and any(
                field in record for field in (
                    "intelligence_index", "output_tokens_per_task", "time_per_task_minutes"
                )
            ):
                embedded_metrics[key] = record

    for key, name in names.items():
        source = metadata.get(key, {})
        repository.add_model({"name": name, **source})
    report["models"] = len(repository.model_names())

    version_path = status_path.with_name(f"{status_path.stem}.intelligence-index.txt")
    if version_path.exists():
        repository.record_source_version(
            "intelligence_index", version_path.read_text(encoding="utf-8")
        )

    legacy_result_files = {
        "intelligence_index": data_dir / "all_model_data.json",
        "terminalbench-4-0": data_dir / "terminalbench_4-0_model_data.json",
        "aa-briefcase": data_dir / "aa_briefcase_model_data.json",
        "automationbench-aa": data_dir / "automationbench_aa_model_data.json",
    }
    for code, path in legacy_result_files.items():
        records = _read_json(path, [])
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict) or not record.get("model_name"):
                continue
            name = str(record["model_name"])
            repository.add_model({"name": name})
            repository.save_benchmark_metrics(code, [name], {name: _metric_from_legacy(code, record)})

    for key, record in embedded_metrics.items():
        name = names[key]
        repository.save_benchmark_metrics(
            "intelligence_index", [name], {name: _metric_from_legacy("intelligence_index", record)}
        )

    profiles = _read_json(model_info_path, [])
    for index, raw in enumerate(profiles if isinstance(profiles, list) else []):
        try:
            profile = ModelInfo.model_validate(raw).model_dump()
            repository.add_model({"name": profile["model_name"]})
            repository.upsert_profile(profile)
        except (TypeError, ValidationError, ValueError) as exc:
            report["errors"].append({"source": str(model_info_path), "index": index, "error": str(exc)})

    # The legacy status table remains authoritative when result files and states disagree.
    for key, name in names.items():
        for code in BENCHMARKS:
            status, error = states.get((key, code), ("pending", None))
            repository.set_state(name, code, status, error)

    counts = repository.rows(
        """SELECT b.code, s.status, COUNT(*) AS count FROM benchmark_states s
           JOIN benchmarks b ON b.id=s.benchmark_id GROUP BY b.code, s.status"""
    )
    report["models"] = len(repository.model_names())
    report["benchmark_states"] = counts
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="将旧 JSON 幂等导入 SQLite")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--status-table", default="存量模型信息记录状态.json")
    parser.add_argument("--model-info", default="model_info.json")
    parser.add_argument("--database", default="data/model_tracker.db")
    args = parser.parse_args()
    report = import_legacy(
        TrackerRepository(Path(args.database).resolve()),
        Path(args.data_dir).resolve(),
        Path(args.status_table).resolve(),
        Path(args.model_info).resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
