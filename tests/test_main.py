import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.db import TrackerRepository
from app.main import app
from app.qwen_agent import ModelInfo


class DatabasePageTests(unittest.TestCase):
    def test_supplement_fills_missing_fields_and_shows_log(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.db"
            repository = TrackerRepository(database)
            repository.add_model({"name": "Demo"})
            repository.upsert_profile({"model_name": "Demo", "vendor_or_organization": "Existing"})
            app.dependency_overrides[get_settings] = lambda: Settings(
                tracker_database_path=str(database), tracker_scan_token="secret"
            )
            researched = {field: None for field in ModelInfo.model_fields}
            researched.update({"model_name": "Demo", "vendor_or_organization": "Replacement",
                               "open_source_attribute": "开源", "region": "中国"})
            try:
                with patch("app.qwen_agent.QwenAgent.research_model_info", new_callable=AsyncMock) as research:
                    research.return_value = researched
                    client = TestClient(app)
                    self.assertEqual(client.post("/models/supplement").status_code, 401)
                    response = client.post("/models/supplement", headers={"X-Scan-Token": "secret"})
                self.assertEqual(response.json(), {"succeeded": 1, "failed": 0})
                self.assertEqual(repository.rows(
                    "SELECT vendor_or_organization, open_source_attribute, region FROM model_profiles"
                )[0], {"vendor_or_organization": "Existing", "open_source_attribute": "开源", "region": "中国"})
                page = client.get("/?table=supplement").text
                self.assertIn("Demo：补充完成", page)
                self.assertNotIn('href="/?table=scan_run_models"', page)
            finally:
                app.dependency_overrides.clear()

    def test_delete_model_requires_token_and_removes_related_data(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.db"
            repository = TrackerRepository(database)
            model_id, _ = repository.add_model({"name": "Delete Me"})
            repository.add_model({"name": "Keep Me"})
            repository.upsert_profile({"model_name": "Delete Me", "region": "中国"})
            repository.save_benchmark_metrics(
                "intelligence_index", ["Delete Me"], {"Delete Me": {
                    "matched": True, "intelligence_index": 42,
                }}
            )
            app.dependency_overrides[get_settings] = lambda: Settings(
                tracker_database_path=str(database), tracker_scan_token="secret"
            )
            try:
                client = TestClient(app)
                self.assertEqual(client.delete(f"/models/{model_id}").status_code, 401)
                page = client.get("/?table=overview").text
                self.assertIn("Delete Me", page)
                self.assertIn(f'data-id="{model_id}"', page)
                self.assertEqual(client.delete(
                    f"/models/{model_id}", headers={"X-Scan-Token": "secret"}
                ).status_code, 200)
                self.assertEqual(client.delete(
                    f"/models/{model_id}", headers={"X-Scan-Token": "secret"}
                ).status_code, 404)
                page = client.get("/?table=overview").text
                self.assertNotIn("Delete Me", page)
                self.assertIn("Keep Me", page)
                self.assertEqual(repository.model_names(), ["Keep Me"])
                for table in ("model_profiles", "benchmark_states", "benchmark_results"):
                    self.assertEqual(repository.rows(
                        f"SELECT COUNT(*) AS n FROM {table} WHERE model_id = ?", (model_id,)
                    )[0]["n"], 0)
            finally:
                app.dependency_overrides.clear()

    def test_overview_shows_all_models_and_benchmarks(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.db"
            repository = TrackerRepository(database)
            repository.add_model({"name": "<Demo>"})
            for number in range(1, 201):
                repository.add_model({"name": f"Model-{number:03}"})
            repository.upsert_profile({
                "model_name": "<Demo>",
                "vendor_or_organization": "<Example>",
                "open_source_attribute": "开源",
                "region": "中国",
            })
            repository.save_benchmark_metrics(
                "intelligence_index", ["<Demo>"], {"<Demo>": {
                    "matched": True, "intelligence_index": 42,
                }}
            )
            repository.save_benchmark_metrics(
                "terminalbench-4-0", ["<Demo>"], {"<Demo>": {
                    "matched": True, "terminalbench_4_0_score": 50,
                }}
            )
            app.dependency_overrides[get_settings] = lambda: Settings(
                tracker_database_path=str(database)
            )
            try:
                response = TestClient(app).get("/?table=overview")
            finally:
                app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text.count("<tr>"), 202)
        self.assertIn("Model-200", response.text)
        self.assertIn("&lt;Demo&gt;", response.text)
        self.assertIn("&lt;Example&gt;", response.text)
        self.assertNotIn("<Demo>", response.text)
        self.assertIn("<td><div>42</div><div>50%</div><div>—</div></td>", response.text)
        for value in ("模型名称", "评测项", "评测结果", "所属组织", "开源状态", "所属地域",
                      "Intelligence Index", "Terminal-Bench 4.0", "τ³-Banking", "42", "开源", "中国", "—"):
            self.assertIn(value, response.text)
        self.assertIn('href="/?table=models"', response.text)

    def test_manual_scan_requires_token_and_returns_new_models(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(tracker_scan_token="test-secret")
            app.dependency_overrides[get_settings] = lambda: settings
            app.state.tracker_scan_args = (root, root / "tracker.db")
            try:
                with patch("app.model_tracker.scan_once", new_callable=AsyncMock) as scan:
                    scan.return_value = [{"name": "New Model"}]
                    client = TestClient(app)
                    self.assertEqual(client.post("/models/scan").status_code, 401)
                    self.assertEqual(scan.await_count, 0)
                    with patch("app.model_tracker.SCAN_LOCK") as lock:
                        lock.locked.return_value = True
                        self.assertEqual(client.post(
                            "/models/scan", headers={"X-Scan-Token": "test-secret"}
                        ).status_code, 409)
                    self.assertEqual(scan.await_count, 0)
                    response = client.post(
                        "/models/scan", headers={"X-Scan-Token": "test-secret"}
                    )
                    scan.assert_awaited_once_with(root, root / "tracker.db")
            finally:
                del app.state.tracker_scan_args
                app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "count": 1, "models": [{"name": "New Model"}]
        })

    def test_page_shows_database_rows_and_escapes_html(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.db"
            TrackerRepository(database).add_model({"name": "<Demo>"})
            app.dependency_overrides[get_settings] = lambda: Settings(
                tracker_database_path=str(database)
            )
            try:
                response = TestClient(app).get("/")
            finally:
                app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        self.assertIn("&lt;Demo&gt;", response.text)
        self.assertNotIn("<Demo>", response.text)

    def test_models_endpoint_returns_profiles_and_benchmarks_for_updated_date(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "tracker.db"
            repository = TrackerRepository(database)
            with patch(
                "app.db.repositories.utc_now", return_value="2026-09-21T01:02:03+00:00"
            ):
                repository.add_model({"name": "Demo"})
                repository.upsert_profile({
                    "model_name": "Demo", "vendor_or_organization": "Example"
                })
                repository.save_benchmark_metrics(
                    "intelligence_index", ["Demo"], {"Demo": {
                        "matched": True, "intelligence_index": 42,
                        "model_name": "Demo High", "slug": "demo-high",
                        "match_method": "exact", "source_url": "https://example.test/demo",
                    }}
                )
            app.dependency_overrides[get_settings] = lambda: Settings(
                tracker_database_path=str(database)
            )
            try:
                response = TestClient(app).get("/models?date=2026-09-21")
            finally:
                app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["models"][0]["profile"]["vendor_or_organization"], "Example")
        self.assertEqual(
            payload["models"][0]["benchmarks"]["intelligence_index"]["score_value"], 42
        )


if __name__ == "__main__":
    unittest.main()
