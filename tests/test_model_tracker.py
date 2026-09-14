import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from app.model_tracker import normalize_name, scan_once, update_model_status_table, write_markdown


class FakeAAClient:
    def __init__(self, description: str, metrics: dict) -> None:
        self.description = description
        self.metrics = metrics
        self.requested: list[str] = []

    async def fetch_models_html(self) -> str:
        return (
            '<span class="bg-brand-purple-dark">Updated</span>'
            f'<span style="text-wrap: pretty;">{self.description}</span>'
        )

    async def enrich_models(self, names: list[str], models_html: str) -> dict:
        self.requested = names
        return {name: self.metrics[name] for name in names}


class ModelTrackerExtractionTests(unittest.TestCase):
    def test_normalize_name_is_only_used_for_deduplication(self) -> None:
        self.assertEqual(normalize_name("Qwen3.8-Flash"), normalize_name("qwen3.8 flash"))

    def test_markdown_uses_integer_metrics_and_one_decimal_minute(self) -> None:
        model = {
            "name": "GPT-6 Astra",
            "published_at": "2026-09-03",
            "title": "OpenAI 发布 GPT-6 Astra",
            "source_url": "https://example.com/news",
            "artificial_analysis": {
                "model_name": "GPT-6 Astra (max)",
                "intelligence_index": 53,
                "output_tokens_per_task": "27 K",
                "time_per_task_minutes": 8.2,
            },
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_markdown(path, "2026-09-11", [model])
            report = path.read_text(encoding="utf-8")
        self.assertIn("| 53 | 27 K | 8.2 分钟 |", report)


class ModelStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_model_is_added_and_description_change_refreshes_all(self) -> None:
        metrics = {
            "Existing": {
                "matched": True,
                "intelligence_index": 40,
                "output_tokens_per_task": "10 K",
                "time_per_task_minutes": 2.5,
            },
            "New Model": {
                "matched": True,
                "intelligence_index": 50,
                "output_tokens_per_task": "20 K",
                "time_per_task_minutes": 3.5,
            },
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            data_path = Path(directory) / "data" / "all_model_data.json"
            path.write_text(
                '[{"model_name":"Existing","record_status":"已记录"}]',
                encoding="utf-8",
            )
            client = FakeAAClient(
                "Artificial Analysis Intelligence Index v4.3 incorporates 10 evaluations",
                metrics,
            )

            await update_model_status_table(path, ["New Model"], client, data_path)

            records = json.loads(path.read_text(encoding="utf-8"))
            all_model_data = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual(client.requested, ["Existing", "New Model"])
        self.assertEqual([record["record_status"] for record in records], ["已记录", "已记录"])
        self.assertEqual(set(records[1]), {"model_name", "record_status"})
        self.assertEqual(all_model_data[1]["intelligence_index"], 50)

    async def test_unchanged_description_only_retries_pending_models(self) -> None:
        description = "Artificial Analysis Intelligence Index v4.3 incorporates 10 evaluations"
        metrics = {
            "Pending": {
                "matched": True,
                "intelligence_index": 42,
                "output_tokens_per_task": "12 K",
                "time_per_task_minutes": 2.0,
            }
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            data_path = Path(directory) / "data" / "all_model_data.json"
            path.write_text(
                '[{"model_name":"Recorded","record_status":"已记录"},'
                '{"model_name":"Pending","record_status":"待更新"}]',
                encoding="utf-8",
            )
            path.with_name("status.intelligence-index.txt").write_text(description, encoding="utf-8")
            client = FakeAAClient(description, metrics)

            await update_model_status_table(path, [], client, data_path)

        self.assertEqual(client.requested, ["Pending"])

    async def test_new_model_is_enriched_when_description_is_unchanged(self) -> None:
        description = "Artificial Analysis Intelligence Index v4.3 incorporates 10 evaluations"
        metric = {
            "matched": True,
            "intelligence_index": 55,
            "output_tokens_per_task": "25 K",
            "time_per_task_minutes": 4.0,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            data_path = Path(directory) / "data" / "all_model_data.json"
            path.write_text("[]", encoding="utf-8")
            path.with_name("status.intelligence-index.txt").write_text(description, encoding="utf-8")
            client = FakeAAClient(description, {"New Model": metric})

            await update_model_status_table(path, ["New Model"], client, data_path)

            status = json.loads(path.read_text(encoding="utf-8"))
            all_model_data = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual(client.requested, ["New Model"])
        self.assertEqual(status, [{"model_name": "New Model", "record_status": "已记录"}])
        self.assertEqual(
            all_model_data,
            [{
                "model_name": "New Model",
                "intelligence_index": 55,
                "output_tokens_per_task": "25 K",
                "time_per_task_minutes": 4.0,
            }],
        )

    async def test_legacy_metrics_are_moved_out_of_status_table(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            data_path = Path(directory) / "data" / "all_model_data.json"
            path.write_text(
                '[{"model_name":"Existing","record_status":"已记录",'
                '"intelligence_index":40,"output_tokens_per_task":"10 K",'
                '"time_per_task_minutes":2.5,"ignored":"value"}]',
                encoding="utf-8",
            )

            await update_model_status_table(path, [], None, data_path)

            status = json.loads(path.read_text(encoding="utf-8"))
            all_model_data = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual(status, [{"model_name": "Existing", "record_status": "已记录"}])
        self.assertEqual(all_model_data[0]["intelligence_index"], 40)

    async def test_scan_uses_status_table_as_new_model_baseline(self) -> None:
        release = {"name": "Existing", "item": {"id": "1", "title": "Existing released"}}

        class FakeAgent:
            def __init__(self, settings: object) -> None:
                pass

            async def discover_model_releases(self, client: object) -> list[dict]:
                return [release]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "status.json"
            status_path.write_text(
                '[{"model_name":"Existing","record_status":"已记录"}]',
                encoding="utf-8",
            )
            settings = SimpleNamespace(artificial_analysis_enabled=False)
            with patch("app.model_tracker.get_settings", return_value=settings), patch(
                "app.model_tracker.QwenAgent", FakeAgent
            ):
                models = await scan_once(root / "data", status_path)

        self.assertEqual(models, [])


if __name__ == "__main__":
    unittest.main()
