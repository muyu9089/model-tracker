import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.db import TrackerRepository
from app.main import app


class DatabasePageTests(unittest.TestCase):
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
