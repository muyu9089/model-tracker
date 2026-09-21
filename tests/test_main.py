import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


if __name__ == "__main__":
    unittest.main()
