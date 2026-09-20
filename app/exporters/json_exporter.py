import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.artificial_analysis import format_thousands, round_half_up
from app.db import BENCHMARKS, TrackerRepository
from app.db.repositories import PROFILE_FIELDS, normalize_name


STATUS_LABELS = {"pending": "待更新", "recorded": "已记录", "failed": "待更新"}
SCORE_FIELDS = {
    "intelligence_index": "intelligence_index",
    "terminalbench-4-0": "terminalbench_4_0_score",
    "tau3-banking": "tau3_banking_score",
}


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _display_result(code: str, row: dict[str, Any]) -> dict[str, Any]:
    score_field = SCORE_FIELDS[code]
    score = row.get("score_value")
    if code == "intelligence_index":
        score = round_half_up(score)
    elif isinstance(score, (int, float)):
        score = round_half_up(score, 1)
    return {
        "model_name": row["model_name"],
        score_field: score,
        "output_tokens_per_task": format_thousands(row.get("output_tokens")),
        "time_per_task_minutes": round_half_up(row["time_seconds"] / 60, 1)
        if isinstance(row.get("time_seconds"), (int, float)) else None,
    }


class JsonExporter:
    def __init__(
        self,
        repository: TrackerRepository,
        data_dir: Path,
        status_path: Path,
        model_info_path: Path,
        export_dir: Path,
    ) -> None:
        self.repository = repository
        self.data_dir = Path(data_dir)
        self.status_path = Path(status_path)
        self.model_info_path = Path(model_info_path)
        self.export_dir = Path(export_dir)

    def _states(self) -> dict[str, list[dict[str, Any]]]:
        rows = self.repository.rows(
            """SELECT m.model_name, b.code, s.status, s.update_error
               FROM benchmark_states s JOIN models m ON m.id=s.model_id
               JOIN benchmarks b ON b.id=s.benchmark_id ORDER BY m.id, b.id"""
        )
        result = {code: [] for code in BENCHMARKS}
        for row in rows:
            item = {"model_name": row["model_name"], "record_status": STATUS_LABELS[row["status"]]}
            if row.get("update_error"):
                item["update_error"] = row["update_error"]
            result[row["code"]].append(item)
        return result

    def _results(self, code: str) -> list[dict[str, Any]]:
        rows = self.repository.rows(
            """SELECT m.model_name, r.* FROM benchmark_results r
               JOIN models m ON m.id=r.model_id JOIN benchmarks b ON b.id=r.benchmark_id
               WHERE b.code=? ORDER BY m.id""", (code,)
        )
        return [_display_result(code, row) for row in rows]

    def _profiles(self) -> list[dict[str, Any]]:
        columns = ", ".join(f"p.{field}" for field in PROFILE_FIELDS)
        return self.repository.rows(
            f"""SELECT m.model_name, {columns} FROM model_profiles p
                JOIN models m ON m.id=p.model_id ORDER BY m.id"""
        )

    def _seen(self) -> dict[str, dict[str, Any]]:
        rows = self.repository.rows(
            """SELECT model_name, normalized_name, first_seen_at, published_at, title,
               source_url, aihot_url, aihot_id FROM models ORDER BY id"""
        )
        return {
            row["normalized_name"]: {
                "name": row["model_name"], "first_seen_at": row["first_seen_at"],
                "published_at": row["published_at"] or "", "title": row["title"] or "",
                "source_url": row["source_url"] or "", "aihot_url": row["aihot_url"] or "",
                "aihot_id": row["aihot_id"] or "",
            }
            for row in rows
        }

    def export_all(self) -> None:
        states = self._states()
        profiles = self._profiles()
        results = {code: self._results(code) for code in BENCHMARKS}
        atomic_write_json(self.status_path, states)
        atomic_write_json(self.data_dir / "seen_models.json", self._seen())
        atomic_write_json(self.data_dir / "all_model_data.json", results["intelligence_index"])
        atomic_write_json(
            self.data_dir / "terminalbench_4-0_model_data.json", results["terminalbench-4-0"]
        )
        atomic_write_json(self.data_dir / "tau3_banking_model_data.json", results["tau3-banking"])
        atomic_write_json(self.model_info_path, profiles)

        profiles_by_name = {normalize_name(row["model_name"]): row for row in profiles}
        results_by_code = {
            code: {normalize_name(row["model_name"]): row for row in values}
            for code, values in results.items()
        }
        state_by_code = {
            code: {normalize_name(row["model_name"]): row for row in values}
            for code, values in states.items()
        }
        models = []
        for row in self.repository.rows(
            "SELECT model_name, normalized_name, first_seen_at FROM models ORDER BY id"
        ):
            key = row["normalized_name"]
            benchmarks = {}
            for code in BENCHMARKS:
                state = state_by_code[code].get(key, {})
                result = results_by_code[code].get(key, {})
                benchmarks[code] = {
                    "status": state.get("record_status", "待更新"),
                    "score": result.get(SCORE_FIELDS[code]),
                    "output_tokens_per_task": result.get("output_tokens_per_task"),
                    "time_per_task_minutes": result.get("time_per_task_minutes"),
                }
                if state.get("update_error"):
                    benchmarks[code]["update_error"] = state["update_error"]
            profile = profiles_by_name.get(key)
            models.append({
                "model_name": row["model_name"],
                "first_seen_at": row["first_seen_at"],
                "profile": {field: profile.get(field) for field in PROFILE_FIELDS} if profile else None,
                "benchmarks": benchmarks,
            })
        latest_run = self.repository.rows("SELECT id FROM scan_runs ORDER BY id DESC LIMIT 1")
        atomic_write_json(self.export_dir / "model_tracker.json", {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": latest_run[0]["id"] if latest_run else None,
            "models": models,
        })

    def export_run_report(self, run_id: int, run_at: datetime) -> None:
        rows = self.repository.rows(
            """SELECT m.model_name AS name, m.published_at, m.title, m.source_url,
               r.score_value AS intelligence_index, r.matched_model_name AS aa_name,
               r.output_tokens, r.time_seconds
               FROM scan_run_models s JOIN models m ON m.id=s.model_id
               LEFT JOIN benchmarks b ON b.code='intelligence_index'
               LEFT JOIN benchmark_results r ON r.model_id=m.id AND r.benchmark_id=b.id
               WHERE s.run_id=? ORDER BY s.discovery_order""", (run_id,)
        )
        models = []
        for row in rows:
            model = {
                "name": row["name"], "published_at": row["published_at"] or "",
                "title": row["title"] or "", "source_url": row["source_url"] or "",
            }
            if row.get("intelligence_index") is not None:
                model["artificial_analysis"] = {
                    "model_name": row.get("aa_name"),
                    "intelligence_index": round_half_up(row["intelligence_index"]),
                    "output_tokens_per_task": format_thousands(row.get("output_tokens")),
                    "time_per_task_minutes": round_half_up(row["time_seconds"] / 60, 1)
                    if isinstance(row.get("time_seconds"), (int, float)) else None,
                }
            models.append(model)
        stem = f"weekly_models_{run_at:%Y%m%d_%H%M%S}"
        report_dir = self.data_dir / "reports"
        atomic_write_json(report_dir / f"{stem}.json", models)
        atomic_write_text(report_dir / f"{stem}.md", _markdown(run_at, models))


def _markdown(run_at: datetime, models: list[dict[str, Any]]) -> str:
    lines = [f"# 最近七天新增模型（{run_at:%Y-%m-%d %H:%M %Z}）", ""]
    if not models:
        lines.append("本周期未发现新增模型。")
    else:
        lines.extend([
            f"共发现 {len(models)} 个此前未记录的模型。", "",
            "| 模型名称 | AA 匹配型号 | Intelligence Index | Output Tokens / Task | Time / Task | 发布时间 | 相关资讯 | 来源 |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ])
        for model in models:
            aa = model.get("artificial_analysis") or {}
            title = str(model["title"]).replace("|", "\\|")
            lines.append(
                f"| {model['name']} | {aa.get('model_name') or '未匹配'} | "
                f"{aa.get('intelligence_index') if aa.get('intelligence_index') is not None else '未获取'} | "
                f"{aa.get('output_tokens_per_task') or '未获取'} | "
                f"{aa.get('time_per_task_minutes') if aa.get('time_per_task_minutes') is not None else '未获取'} 分钟 | "
                f"{model['published_at']} | {title} | "
                f"[原文]({model['source_url']}) |"
            )
    return "\n".join(lines) + "\n"
