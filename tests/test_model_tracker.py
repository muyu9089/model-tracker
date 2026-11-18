import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.db import TrackerRepository
from app.exporters import JsonExporter
from app.migration.import_legacy import import_legacy
from app.model_tracker import enrich_with_aa_mappings, run_daemon, scan_once
from app.qwen_agent import ModelInfo


def profile(name: str) -> dict:
    value = {field: None for field in ModelInfo.model_fields}
    value["model_name"] = name
    return value


class DatabaseTests(unittest.TestCase):
    def test_old_benchmark_mappings_become_one_model_mapping(self) -> None:
        from app.db.connection import transaction

        with TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.db"
            repository = TrackerRepository(path)
            model_id, _ = repository.add_model({"name": "Local"})
            with transaction(path) as connection:
                connection.execute("DROP TABLE aa_model_mappings")
                connection.execute("""CREATE TABLE aa_model_mappings (
                    model_id INTEGER NOT NULL, benchmark_id INTEGER NOT NULL, slug TEXT NOT NULL,
                    PRIMARY KEY (model_id, benchmark_id))""")
                connection.execute("""INSERT INTO aa_model_mappings VALUES
                    (?, (SELECT id FROM benchmarks WHERE code='aa-briefcase'), 'first-slug'),
                    (?, (SELECT id FROM benchmarks WHERE code='automationbench-aa'), 'second-slug')""",
                    (model_id, model_id))
                connection.execute("PRAGMA user_version = 3")
            upgraded = TrackerRepository(path)
            self.assertEqual(upgraded.aa_mappings(), {"Local": "first-slug"})
            self.assertEqual(upgraded.rows("SELECT COUNT(*) AS count FROM aa_model_mappings")[0]["count"], 1)
            self.assertEqual(upgraded.pending_names("automationbench-aa"), ["Local"])

    def test_existing_models_receive_missing_benchmark_states(self) -> None:
        from app.db.connection import transaction

        with TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.db"
            repository = TrackerRepository(path)
            model_id, _ = repository.add_model({"name": "Muse Spark 1.3"})
            with transaction(path) as connection:
                connection.execute("""DELETE FROM benchmark_states WHERE model_id=? AND benchmark_id IN
                    (SELECT id FROM benchmarks WHERE code IN ('aa-briefcase', 'automationbench-aa'))""", (model_id,))
            repository.set_aa_mapping(model_id, "muse-spark-1-3")
            repaired = TrackerRepository(path)
            self.assertEqual(repaired.pending_names("aa-briefcase"), ["Muse Spark 1.3"])
            self.assertEqual(repaired.pending_names("automationbench-aa"), ["Muse Spark 1.3"])

    def test_existing_database_adds_aa_mapping_table(self) -> None:
        from app.db.connection import transaction

        with TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.db"
            repository = TrackerRepository(path)
            model_id, _ = repository.add_model({"name": "Existing"})
            repository.upsert_profile({"model_name": "Existing"})
            collected_at = repository.rows(
                "SELECT collected_at FROM model_profiles WHERE model_id=?", (model_id,)
            )[0]["collected_at"]
            self.assertEqual(repository.rows(
                "SELECT model_info_updated_at FROM models WHERE id=?", (model_id,)
            )[0]["model_info_updated_at"], collected_at)
            with transaction(path) as connection:
                connection.execute("DROP TABLE aa_model_mappings")
                connection.execute("ALTER TABLE models DROP COLUMN model_info_updated_at")
                connection.execute("PRAGMA user_version = 1")
            upgraded = TrackerRepository(path)
            self.assertTrue(upgraded.set_aa_mapping(model_id, "existing-slug"))
            self.assertEqual(upgraded.aa_mappings(), {"Existing": "existing-slug"})
            self.assertEqual(upgraded.rows(
                "SELECT model_info_updated_at FROM models WHERE id=?", (model_id,)
            )[0]["model_info_updated_at"], collected_at)

    def test_aa_mapping_uses_slug_and_rejects_wrong_match(self) -> None:
        from unittest.mock import AsyncMock

        with TemporaryDirectory() as directory:
            repository = TrackerRepository(Path(directory) / "tracker.db")
            model_id, _ = repository.add_model({"name": "Local Name"})
            self.assertTrue(repository.set_aa_mapping(model_id, "aa-slug"))
            client = SimpleNamespace(enrich_aa_briefcase_models=AsyncMock(return_value={
                "aa-slug": {"matched": True, "slug": "aa-slug", "aa_briefcase_score": 123}
            }))
            with self.assertRaises(StopIteration) as completed:
                enrich_with_aa_mappings(repository, client, "aa-briefcase", ["Local Name"], "html").send(None)
            result = completed.exception.value
            client.enrich_aa_briefcase_models.assert_awaited_once_with(["aa-slug"], "html")
            self.assertEqual(result["Local Name"]["aa_briefcase_score"], 123)
            client.enrich_aa_briefcase_models.return_value = {
                "aa-slug": {"matched": True, "slug": "similar-but-wrong", "aa_briefcase_score": 999}
            }
            with self.assertRaises(StopIteration) as completed:
                enrich_with_aa_mappings(repository, client, "aa-briefcase", ["Local Name"], "html").send(None)
            result = completed.exception.value
            self.assertFalse(result["Local Name"]["matched"])

    def test_initialization_is_idempotent_and_new_model_is_atomic(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.db"
            repository = TrackerRepository(path)
            from app.db.connection import transaction
            with transaction(path) as connection:
                connection.execute("UPDATE benchmarks SET display_name='AA-Briefcase' WHERE code='aa-briefcase'")
            TrackerRepository(path)
            first_id, created = repository.add_model({"name": "Demo-2"})
            second_id, duplicate = repository.add_model({"name": "demo 2"})
            with transaction(path) as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                display_name = connection.execute(
                    "SELECT display_name FROM benchmarks WHERE code='aa-briefcase'"
                ).fetchone()[0]
                state_count = connection.execute(
                    "SELECT COUNT(*) FROM benchmark_states WHERE model_id=?", (first_id,)
                ).fetchone()[0]

        self.assertEqual(version, 4)
        self.assertEqual(display_name, "AA-Briefcase v1.1")
        self.assertTrue(created)
        self.assertFalse(duplicate)
        self.assertEqual(first_id, second_id)
        self.assertEqual(state_count, 4)

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
            metric = {
                "matched": True, "intelligence_index": 42,
                "model_name": "Demo (high)", "slug": "demo-high",
                "match_method": "family", "source_url": "https://example.test/demo-high",
                "output_tokens": 7500,
            }
            repository.save_benchmark_metrics("intelligence_index", ["Demo"], {"Demo": metric})
            self.assertEqual(repository.pending_names("intelligence_index"), ["Demo"])
            metric["time_seconds"] = 150
            repository.save_benchmark_metrics("intelligence_index", ["Demo"], {"Demo": metric})
            self.assertEqual(repository.pending_names("intelligence_index"), [])

    def test_empty_score_clears_previously_saved_metrics_and_logs_no_data(self) -> None:
        with TemporaryDirectory() as directory:
            repository = TrackerRepository(Path(directory) / "tracker.db")
            repository.add_model({"name": "Demo"})
            repository.save_benchmark_metrics("intelligence_index", ["Demo"], {"Demo": {
                "matched": True, "intelligence_index": 42,
                "output_tokens": 7500, "time_seconds": 150,
            }})

            with self.assertLogs("app.db.repositories", level="WARNING") as logs:
                repository.save_benchmark_metrics("intelligence_index", ["Demo"], {"Demo": {
                    "matched": True, "intelligence_index": None,
                    "output_tokens": 9000, "time_seconds": 180,
                }})

            row = repository.rows("""SELECT r.score_value, r.output_tokens, r.time_seconds,
                s.status, s.update_error FROM benchmark_results r
                JOIN benchmark_states s ON s.model_id=r.model_id AND s.benchmark_id=r.benchmark_id""")[0]
            self.assertEqual((row["score_value"], row["output_tokens"], row["time_seconds"]), (None, None, None))
            self.assertEqual((row["status"], row["update_error"]), ("pending", "无数据"))
            self.assertIn("无数据", logs.output[0])

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
            "aa-briefcase": "pending",
            "automationbench-aa": "pending",
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
                "aa-briefcase": [{"model_name": "Demo-2", "record_status": "已记录"}],
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
    async def test_named_scan_uses_only_selected_models(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "tracker.db"
            repository = TrackerRepository(database)
            repository.add_model({"name": "Existing", "title": "Original title"})
            repository.add_model({"name": "Unselected"})
            with (
                patch("app.model_tracker.QwenAgent") as agent,
                patch("app.model_tracker.update_intelligence_index", new_callable=AsyncMock) as index,
                patch("app.model_tracker.update_benchmark", new_callable=AsyncMock) as benchmark,
                patch("app.model_tracker.get_settings", return_value=SimpleNamespace(
                    tracker_timezone="Asia/Shanghai", tracker_export_dir=str(root),
                    artificial_analysis_enabled=True, artificial_analysis_base_url="https://example.test",
                    artificial_analysis_timeout_seconds=10,
                )),
            ):
                agent.return_value.research_model_info = AsyncMock(side_effect=profile)
                models = await scan_once(root, database, export_json=False, model_names=["existing", "New Model"])
                agent.return_value.discover_model_releases.assert_not_called()
                self.assertEqual(agent.return_value.research_model_info.await_count, 2)
                self.assertEqual(index.await_args.args[-1], ["Existing", "New Model"])
                self.assertEqual([call.args[-1] for call in benchmark.await_args_list], [["Existing", "New Model"]] * 3)
            self.assertEqual([row["name"] for row in models], ["Existing", "New Model"])
            self.assertEqual(repository.rows("SELECT mode, new_model_count FROM scan_runs"),
                             [{"mode": "by_name", "new_model_count": 1}])
            self.assertEqual(repository.rows("SELECT title FROM models WHERE model_name='Existing'"),
                             [{"title": "Original title"}])

    async def test_daemon_starts_scheduler_and_web_server(self) -> None:
        class FakeScheduler:
            started = False
            stopped = False

            def add_job(self, *args: object, **kwargs: object) -> None:
                pass

            def start(self) -> None:
                self.started = True

            def shutdown(self, wait: bool) -> None:
                self.stopped = not wait

        class FakeServer:
            served = False

            async def serve(self) -> None:
                self.served = True

        scheduler = FakeScheduler()
        server = FakeServer()
        settings = SimpleNamespace(
            tracker_timezone="Asia/Shanghai", tracker_day_of_week="mon",
            tracker_hour=9, tracker_minute=0, tracker_web_host="0.0.0.0",
            tracker_web_port=8000,
        )
        with TemporaryDirectory() as directory, patch(
            "app.model_tracker.get_settings", return_value=settings
        ), patch(
            "app.model_tracker.AsyncIOScheduler", return_value=scheduler
        ), patch(
            "app.model_tracker.uvicorn.Server", return_value=server
        ) as server_class:
            root = Path(directory)
            await run_daemon(root, root / "tracker.db", root / "status.json",
                             root / "model_info.json", root / "exports", True)

        config = server_class.call_args.args[0]
        self.assertTrue(scheduler.started)
        self.assertTrue(scheduler.stopped)
        self.assertTrue(server.served)
        self.assertEqual((config.host, config.port), ("0.0.0.0", 8000))

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
