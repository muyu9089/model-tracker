import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from app.db import TrackerRepository
from app.exporters import JsonExporter
from app.migration.import_legacy import import_legacy
from app.model_tracker import scan_once
from app.qwen_agent import ModelInfo


def profile(name: str) -> dict:
    value = {field: None for field in ModelInfo.model_fields}
    value["model_name"] = name
    return value


class DatabaseTests(unittest.TestCase):
    def test_initialization_is_idempotent_and_new_model_is_atomic(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.db"
            repository = TrackerRepository(path)
            TrackerRepository(path)
            first_id, created = repository.add_model({"name": "Demo-2"})
            second_id, duplicate = repository.add_model({"name": "demo 2"})
            from app.db.connection import transaction
            with transaction(path) as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                state_count = connection.execute(
                    "SELECT COUNT(*) FROM benchmark_states WHERE model_id=?", (first_id,)
                ).fetchone()[0]

        self.assertEqual(version, 1)
        self.assertTrue(created)
        self.assertFalse(duplicate)
        self.assertEqual(first_id, second_id)
        self.assertEqual(state_count, 3)

    def test_recorded_result_without_match_metadata_is_refreshed(self) -> None:
        with TemporaryDirectory() as directory:
            repository = TrackerRepository(Path(directory) / "tracker.db")
            repository.add_model({"name": "Demo"})
            repository.save_benchmark_metrics(
                "intelligence_index", ["Demo"], {"Demo": {
                    "matched": True, "intelligence_index": 42,
                }}
            )
            self.assertEqual(repository.pending_names("intelligence_index"), ["Demo"])
            repository.save_benchmark_metrics(
                "intelligence_index", ["Demo"], {"Demo": {
                    "matched": True, "intelligence_index": 42,
                    "model_name": "Demo (high)", "slug": "demo-high",
                    "match_method": "family", "source_url": "https://example.test/demo-high",
                }}
            )
            self.assertEqual(repository.pending_names("intelligence_index"), [])

    def test_connection_enables_foreign_keys_and_source_change_resets_one_benchmark(self) -> None:
        with TemporaryDirectory() as directory:
            repository = TrackerRepository(Path(directory) / "tracker.db")
            repository.add_model({"name": "Demo"})
            repository.save_benchmark_metrics(
                "intelligence_index", ["Demo"], {"Demo": {
                    "matched": True, "intelligence_index": 42,
                    "output_tokens": 1200, "time_seconds": 30,
                }}
            )
            repository.save_benchmark_metrics(
                "terminalbench-4-0", ["Demo"], {"Demo": {
                    "matched": True, "terminalbench_4_0_score": 50,
                    "output_tokens": 2000, "time_seconds": 60,
                }}
            )
            repository.record_source_version("intelligence_index", "v1")
            repository.save_benchmark_metrics(
                "intelligence_index", ["Demo"], {"Demo": {
                    "matched": True, "intelligence_index": 42,
                    "output_tokens": 1200, "time_seconds": 30,
                }}
            )
            self.assertFalse(repository.record_source_version("intelligence_index", "v1"))
            self.assertTrue(repository.record_source_version("intelligence_index", "v2"))
            self.assertTrue(repository.record_source_version("intelligence_index", "v1"))
            self.assertFalse(repository.record_source_version("intelligence_index", "v1"))
            states = repository.rows(
                """SELECT b.code, s.status FROM benchmark_states s
                   JOIN benchmarks b ON b.id=s.benchmark_id ORDER BY b.id"""
            )
            from app.db.connection import transaction
            with transaction(repository.path) as connection:
                foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]

        self.assertEqual(foreign_keys, 1)
        self.assertEqual({row["code"]: row["status"] for row in states}, {
            "intelligence_index": "pending",
            "terminalbench-4-0": "recorded",
            "tau3-banking": "pending",
        })


class MigrationAndExportTests(unittest.TestCase):
    def test_legacy_import_is_idempotent_and_reports_conflicts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            status = root / "status.json"
            model_info = root / "model_info.json"
            status.write_text(json.dumps({
                "intelligence_index": [
                    {"model_name": "Demo-2", "record_status": "已记录"},
                    {"model_name": "demo 2", "record_status": "待更新"},
                ],
                "tau3-banking": [{"model_name": "Demo-2", "record_status": "已记录"}],
            }), encoding="utf-8")
            (data / "seen_models.json").write_text(json.dumps({
                "demo2": {"name": "Demo-2", "first_seen_at": "2026-01-01T00:00:00+00:00"}
            }), encoding="utf-8")
            model_info.write_text(json.dumps([profile("Demo-2"), {"model_name": "Broken"}]),
                                  encoding="utf-8")
            repository = TrackerRepository(root / "tracker.db")

            first = import_legacy(repository, data, status, model_info)
            second = import_legacy(repository, data, status, model_info)
            names = repository.model_names()

        self.assertEqual(names, ["Demo-2"])
        self.assertEqual(first["conflicts"], 1)
        self.assertEqual(second["models"], 1)
        self.assertEqual(len(first["errors"]), 1)

    def test_compatibility_and_unified_exports_use_database_values(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrackerRepository(root / "data" / "tracker.db")
            repository.add_model({"name": "Demo", "first_seen_at": "2026-01-01T00:00:00+00:00"})
            repository.upsert_profile(profile("Demo"))
            repository.save_benchmark_metrics(
                "terminalbench-4-0", ["Demo"], {"Demo": {
                    "matched": True, "terminalbench_4_0_score": 50.04,
                    "output_tokens": 2000, "time_seconds": 61,
                }}
            )
            exporter = JsonExporter(
                repository, root / "data", root / "status.json",
                root / "model_info.json", root / "data" / "exports",
            )
            exporter.export_all()
            terminal = json.loads(
                (root / "data" / "terminalbench_4-0_model_data.json").read_text(encoding="utf-8")
            )
            unified = json.loads(
                (root / "data" / "exports" / "model_tracker.json").read_text(encoding="utf-8")
            )
            integrity = repository.integrity_check()

        self.assertEqual(terminal[0]["output_tokens_per_task"], "2 K")
        self.assertEqual(terminal[0]["time_per_task_minutes"], 1.0)
        self.assertEqual(unified["schema_version"], "1.0")
        self.assertEqual(unified["models"][0]["benchmarks"]["terminalbench-4-0"]["score"], 50.0)
        self.assertEqual(integrity, "ok")


class ScanTests(unittest.IsolatedAsyncioTestCase):
    async def test_scan_queries_profile_only_for_new_models(self) -> None:
        releases = [
            {"name": "Existing", "item": {"id": "1", "title": "Existing"}},
            {"name": "New Model", "item": {"id": "2", "title": "New"}},
        ]

        class FakeAgent:
            requested: list[str] = []

            def __init__(self, settings: object) -> None:
                pass

            async def discover_model_releases(self, client: object) -> list[dict]:
                return releases

            async def research_model_info(self, name: str) -> dict:
                self.requested.append(name)
                return profile(name)

        class BrokenExporter:
            def export_all(self) -> None:
                raise OSError("disk full")

            def export_run_report(self, run_id: int, run_at: object) -> None:
                raise AssertionError("report export must not run after export_all fails")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "data" / "tracker.db"
            repository = TrackerRepository(database)
            repository.add_model({"name": "Existing"})
            settings = SimpleNamespace(
                tracker_timezone="Asia/Shanghai",
                tracker_database_path=str(database),
                tracker_export_dir=str(root / "data" / "exports"),
                tracker_json_export_enabled=True,
                artificial_analysis_enabled=False,
            )
            with patch("app.model_tracker.get_settings", return_value=settings), patch(
                "app.model_tracker.QwenAgent", FakeAgent
            ), patch(
                "app.model_tracker.make_exporter", return_value=BrokenExporter()
            ):
                models = await scan_once(
                    root / "data", database, root / "status.json",
                    root / "model_info.json", root / "data" / "exports", True,
                )

            profiles = repository.rows(
                """SELECT m.model_name FROM model_profiles p
                   JOIN models m ON m.id=p.model_id ORDER BY m.id"""
            )
            run_status = repository.rows("SELECT status FROM scan_runs")[0]["status"]
            integrity = repository.integrity_check()

        self.assertEqual(FakeAgent.requested, ["New Model"])
        self.assertEqual([model["name"] for model in models], ["New Model"])
        self.assertEqual([row["model_name"] for row in profiles], ["New Model"])
        self.assertEqual(run_status, "completed")
        self.assertEqual(integrity, "ok")


if __name__ == "__main__":
    unittest.main()
