import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .connection import transaction
from .migrations import migrate


BENCHMARKS = {
    "intelligence_index": "Intelligence Index",
    "terminalbench-4-0": "Terminal-Bench 4.0",
    "tau3-banking": "τ³-Banking",
}
PROFILE_FIELDS = (
    "open_source_attribute", "vendor_or_organization", "model_series",
    "release_or_api_url", "release_date", "model_architecture", "region",
    "bank_applicable_scenarios", "minimum_gpu_and_count", "minimum_context_length",
    "minimum_concurrency", "recommended_gpu_and_count", "recommended_context_length",
    "recommended_concurrency", "license", "total_parameters_b", "active_parameters_b",
    "input_modalities", "output_modalities",
)


def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).casefold()
    return "".join(character for character in normalized if character.isalnum())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_tokens_to_int(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(round(value))
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*K\s*", str(value or ""), re.I)
    return int(round(float(match.group(1)) * 1000)) if match else None


class TrackerRepository:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        migrate(self.path)
        with transaction(self.path) as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO benchmarks(code, display_name) VALUES (?, ?)",
                BENCHMARKS.items(),
            )

    def integrity_check(self) -> str:
        with transaction(self.path) as connection:
            return str(connection.execute("PRAGMA integrity_check").fetchone()[0])

    def add_model(self, record: dict[str, Any]) -> tuple[int, bool]:
        name = str(record.get("name") or record.get("model_name") or "").strip()
        normalized = normalize_name(name)
        if not normalized:
            raise ValueError("模型名称不能为空")
        now = utc_now()
        first_seen = str(record.get("first_seen_at") or now)
        with transaction(self.path) as connection:
            row = connection.execute(
                "SELECT id FROM models WHERE normalized_name = ?", (normalized,)
            ).fetchone()
            if row:
                model_id = int(row["id"])
                connection.execute(
                    """UPDATE models SET
                       model_name = ?, published_at = COALESCE(NULLIF(?, ''), published_at),
                       title = COALESCE(NULLIF(?, ''), title),
                       source_url = COALESCE(NULLIF(?, ''), source_url),
                       aihot_url = COALESCE(NULLIF(?, ''), aihot_url),
                       aihot_id = COALESCE(NULLIF(?, ''), aihot_id), updated_at = ?
                       WHERE id = ?""",
                    (name, record.get("published_at"), record.get("title"),
                     record.get("source_url"), record.get("aihot_url"),
                     record.get("aihot_id"), now, model_id),
                )
                return model_id, False
            cursor = connection.execute(
                """INSERT INTO models(
                   model_name, normalized_name, first_seen_at, published_at, title,
                   source_url, aihot_url, aihot_id, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, normalized, first_seen, record.get("published_at"), record.get("title"),
                 record.get("source_url"), record.get("aihot_url"), record.get("aihot_id"),
                 now, now),
            )
            model_id = int(cursor.lastrowid)
            connection.execute(
                """INSERT INTO benchmark_states(model_id, benchmark_id, status, updated_at)
                   SELECT ?, id, 'pending', ? FROM benchmarks""",
                (model_id, now),
            )
            return model_id, True

    def model_names(self) -> list[str]:
        with transaction(self.path) as connection:
            return [str(row[0]) for row in connection.execute(
                "SELECT model_name FROM models ORDER BY id"
            )]

    def has_profile(self, name: str) -> bool:
        with transaction(self.path) as connection:
            row = connection.execute(
                """SELECT 1 FROM model_profiles p JOIN models m ON m.id = p.model_id
                   WHERE m.normalized_name = ?""",
                (normalize_name(name),),
            ).fetchone()
            return row is not None

    def upsert_profile(self, profile: dict[str, Any]) -> None:
        name = str(profile["model_name"])
        with transaction(self.path) as connection:
            row = connection.execute(
                "SELECT id FROM models WHERE normalized_name = ?", (normalize_name(name),)
            ).fetchone()
            if row is None:
                raise ValueError(f"模型不存在：{name}")
            columns = ", ".join(PROFILE_FIELDS)
            placeholders = ", ".join("?" for _ in PROFILE_FIELDS)
            updates = ", ".join(f"{field}=excluded.{field}" for field in PROFILE_FIELDS)
            connection.execute(
                f"""INSERT INTO model_profiles(model_id, {columns}, collected_at)
                    VALUES (?, {placeholders}, ?)
                    ON CONFLICT(model_id) DO UPDATE SET {updates}, collected_at=excluded.collected_at""",
                (int(row["id"]), *(profile.get(field) for field in PROFILE_FIELDS), utc_now()),
            )

    def pending_names(self, benchmark_code: str) -> list[str]:
        with transaction(self.path) as connection:
            return [str(row[0]) for row in connection.execute(
                """SELECT m.model_name FROM benchmark_states s
                   JOIN models m ON m.id = s.model_id
                   JOIN benchmarks b ON b.id = s.benchmark_id
                   WHERE b.code = ? AND s.status != 'recorded' ORDER BY m.id""",
                (benchmark_code,),
            )]

    def record_source_version(self, source_code: str, content: str) -> bool:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        now = utc_now()
        with transaction(self.path) as connection:
            previous = connection.execute(
                """SELECT content_hash FROM source_versions WHERE source_code = ?
                   ORDER BY id DESC LIMIT 1""", (source_code,)
            ).fetchone()
            if previous and previous["content_hash"] == digest:
                return False
            connection.execute(
                """INSERT INTO source_versions(
                   source_code, content_hash, content_text, detected_at) VALUES (?, ?, ?, ?)""",
                (source_code, digest, content, now),
            )
            benchmark = connection.execute(
                "SELECT id FROM benchmarks WHERE code = ?", (source_code,)
            ).fetchone()
            if benchmark:
                connection.execute(
                    """UPDATE benchmark_states SET status='pending', update_error=NULL, updated_at=?
                       WHERE benchmark_id=?""", (now, int(benchmark["id"])),
                )
            return True

    def set_state(self, name: str, benchmark_code: str, status: str, error: str | None = None) -> None:
        with transaction(self.path) as connection:
            connection.execute(
                """UPDATE benchmark_states SET status=?, update_error=?, updated_at=?
                   WHERE model_id=(SELECT id FROM models WHERE normalized_name=?)
                   AND benchmark_id=(SELECT id FROM benchmarks WHERE code=?)""",
                (status, error, utc_now(), normalize_name(name), benchmark_code),
            )

    def save_benchmark_metrics(
        self, benchmark_code: str, names: list[str], metrics: dict[str, dict[str, Any]]
    ) -> None:
        now = utc_now()
        with transaction(self.path) as connection:
            benchmark_id = int(connection.execute(
                "SELECT id FROM benchmarks WHERE code = ?", (benchmark_code,)
            ).fetchone()[0])
            for name in names:
                model = connection.execute(
                    "SELECT id FROM models WHERE normalized_name = ?", (normalize_name(name),)
                ).fetchone()
                if model is None:
                    continue
                model_id = int(model["id"])
                metric = metrics.get(name) or {}
                if not metric.get("matched"):
                    connection.execute(
                        """UPDATE benchmark_states SET status='pending', update_error=?, updated_at=?
                           WHERE model_id=? AND benchmark_id=?""",
                        (metric.get("error") or f"未获取到 {BENCHMARKS[benchmark_code]} 指标",
                         now, model_id, benchmark_id),
                    )
                    continue
                score = metric.get({
                    "intelligence_index": "intelligence_index",
                    "terminalbench-4-0": "terminalbench_4_0_score",
                    "tau3-banking": "tau3_banking_score",
                }[benchmark_code])
                tokens = metric.get("output_tokens")
                if not isinstance(tokens, (int, float)):
                    tokens = _display_tokens_to_int(metric.get("output_tokens_per_task"))
                seconds = metric.get("time_seconds")
                if not isinstance(seconds, (int, float)):
                    minutes = metric.get("time_per_task_minutes")
                    seconds = float(minutes) * 60 if isinstance(minutes, (int, float)) else None
                connection.execute(
                    """INSERT INTO benchmark_results(
                       model_id, benchmark_id, score_value, score_unit, output_tokens, time_seconds,
                       matched_model_name, matched_slug, matched_parameters_b,
                       matched_reasoning_depth, match_method, source_url, collected_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(model_id, benchmark_id) DO UPDATE SET
                       score_value=excluded.score_value, score_unit=excluded.score_unit,
                       output_tokens=excluded.output_tokens, time_seconds=excluded.time_seconds,
                       matched_model_name=excluded.matched_model_name,
                       matched_slug=excluded.matched_slug,
                       matched_parameters_b=excluded.matched_parameters_b,
                       matched_reasoning_depth=excluded.matched_reasoning_depth,
                       match_method=excluded.match_method, source_url=excluded.source_url,
                       collected_at=excluded.collected_at""",
                    (model_id, benchmark_id, score,
                     "index" if benchmark_code == "intelligence_index" else "percent",
                     int(round(tokens)) if isinstance(tokens, (int, float)) else None,
                     float(seconds) if isinstance(seconds, (int, float)) else None,
                     metric.get("model_name"), metric.get("slug"),
                     metric.get("parameters_billions"), metric.get("reasoning_effort"),
                     metric.get("match_method"), metric.get("source_url"),
                     metric.get("fetched_at") or now),
                )
                connection.execute(
                    """UPDATE benchmark_states SET status='recorded', update_error=NULL, updated_at=?
                       WHERE model_id=? AND benchmark_id=?""", (now, model_id, benchmark_id),
                )

    def start_run(self, mode: str = "scan") -> int:
        with transaction(self.path) as connection:
            cursor = connection.execute(
                "INSERT INTO scan_runs(mode, started_at, status) VALUES (?, ?, 'running')",
                (mode, utc_now()),
            )
            return int(cursor.lastrowid)

    def add_run_model(self, run_id: int, model_id: int, order: int) -> None:
        with transaction(self.path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO scan_run_models(run_id, model_id, discovery_order) VALUES (?, ?, ?)",
                (run_id, model_id, order),
            )

    def finish_run(self, run_id: int, status: str, count: int, error: str | None = None) -> None:
        with transaction(self.path) as connection:
            connection.execute(
                """UPDATE scan_runs SET finished_at=?, status=?, new_model_count=?, error_summary=?
                   WHERE id=?""", (utc_now(), status, count, error, run_id),
            )

    def rows(self, query: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with transaction(self.path) as connection:
            return [dict(row) for row in connection.execute(query, parameters)]
