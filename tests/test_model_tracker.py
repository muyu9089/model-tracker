import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from app.model_tracker import (
    collect_model_info,
    normalize_name,
    scan_once,
    update_model_status_table,
    update_tau_banking_status_table,
    update_terminalbench_status_table,
    write_markdown,
)
from app.qwen_agent import ModelInfo


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

class ModelInfoTests(unittest.IsolatedAsyncioTestCase):
    async def test_collection_queries_only_new_names_and_preserves_existing(self) -> None:
        def record(name: str) -> dict:
            value = {field: None for field in ModelInfo.model_fields}
            value["model_name"] = name
            return value

        class FakeAgent:
            def __init__(self) -> None:
                self.requested: list[str] = []

            async def research_model_info(self, name: str) -> dict:
                self.requested.append(name)
                return record(name)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "model_info.json"
            output_path.write_text(json.dumps([record("Existing")]), encoding="utf-8")
            agent = FakeAgent()

            result = await collect_model_info(["New Model"], output_path, agent)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(agent.requested, ["New Model"])
        self.assertEqual([item["model_name"] for item in result], ["Existing", "New Model"])
        self.assertEqual(saved, result)


class ModelStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_scan_collects_model_info_only_for_new_models(self) -> None:
        releases = [
            {"name": "Existing", "item": {"id": "1", "title": "Existing released"}},
            {"name": "New Model", "item": {"id": "2", "title": "New released"}},
        ]

        class FakeAgent:
            requested: list[str] = []

            def __init__(self, settings: object) -> None:
                pass

            async def discover_model_releases(self, client: object) -> list[dict]:
                return releases

            async def research_model_info(self, name: str) -> dict:
                self.requested.append(name)
                value = {field: None for field in ModelInfo.model_fields}
                value["model_name"] = name
                return value

        with TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "status.json"
            model_info_path = root / "model_info.json"
            status_path.write_text(json.dumps({
                "intelligence_index": [{"model_name": "Existing", "record_status": "已记录"}],
            }), encoding="utf-8")
            settings = SimpleNamespace(artificial_analysis_enabled=False)
            with patch("app.model_tracker.get_settings", return_value=settings), patch(
                "app.model_tracker.QwenAgent", FakeAgent
            ):
                await scan_once(root / "data", status_path, model_info_path)

            saved = json.loads(model_info_path.read_text(encoding="utf-8"))

        self.assertEqual(FakeAgent.requested, ["New Model"])
        self.assertEqual([record["model_name"] for record in saved], ["New Model"])

    async def test_terminalbench_updates_its_layer_and_separate_data(self) -> None:
        class FakeTerminalClient:
            async def fetch_terminalbench_html(self) -> str:
                return "page"

            async def enrich_terminalbench_models(self, names: list[str], html: str) -> dict:
                return {
                    "Matched": {"matched": True, "terminalbench_4_0_score": 50.0,
                                "output_tokens_per_task": "2 K", "time_per_task_minutes": 0.3},
                    "Missing": {"matched": False, "error": "未匹配到 Terminal-Bench 4.0 模型"},
                }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            data_path = Path(directory) / "terminalbench_data.json"
            original_index = [
                {"model_name": "Matched", "record_status": "已记录"},
                {"model_name": "Missing", "record_status": "已记录"},
            ]
            path.write_text(json.dumps({
                "intelligence_index": original_index,
                "terminalbench-4-0": [
                    {"model_name": "Matched", "record_status": "待更新"},
                    {"model_name": "Missing", "record_status": "待更新"},
                ],
                "tau3-banking": original_index,
            }), encoding="utf-8")

            await update_terminalbench_status_table(path, FakeTerminalClient(), data_path)

            status = json.loads(path.read_text(encoding="utf-8"))
            data = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual(status["intelligence_index"], original_index)
        self.assertEqual(status["terminalbench-4-0"][0]["record_status"], "已记录")
        self.assertEqual(status["terminalbench-4-0"][1]["record_status"], "待更新")
        self.assertEqual(data, [{"model_name": "Matched", "terminalbench_4_0_score": 50.0,
                                 "output_tokens_per_task": "2 K", "time_per_task_minutes": 0.3}])

    async def test_tau_banking_updates_only_its_layer_and_separate_data(self) -> None:
        class FakeTauClient:
            async def fetch_tau_banking_html(self) -> str:
                return "page"

            async def enrich_tau_banking_models(self, names: list[str], html: str) -> dict:
                return {
                    "Matched": {"matched": True, "tau3_banking_score": 51.3,
                                "output_tokens_per_task": "9 K", "time_per_task_minutes": 3.0},
                    "Missing": {"matched": False, "error": "未匹配到 𝜏³-Banking 模型"},
                }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            data_path = Path(directory) / "tau_data.json"
            original_index = [
                {"model_name": "Matched", "record_status": "已记录"},
                {"model_name": "Missing", "record_status": "已记录"},
            ]
            path.write_text(json.dumps({"intelligence_index": original_index,
                                        "tau3-banking": [
                                            {"model_name": "Matched", "record_status": "待更新"},
                                            {"model_name": "Missing", "record_status": "待更新"},
                                        ]}), encoding="utf-8")

            await update_tau_banking_status_table(path, FakeTauClient(), data_path)

            status = json.loads(path.read_text(encoding="utf-8"))
            data = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual(status["intelligence_index"], original_index)
        self.assertEqual(status["tau3-banking"][0]["record_status"], "已记录")
        self.assertEqual(status["tau3-banking"][1]["record_status"], "待更新")
        self.assertEqual(data, [{"model_name": "Matched", "tau3_banking_score": 51.3,
                                 "output_tokens_per_task": "9 K", "time_per_task_minutes": 3.0}])

    async def test_layers_sync_names_without_changing_existing_statuses(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            data_path = Path(directory) / "all_model_data.json"
            path.write_text(json.dumps({
                "intelligence_index": [{"model_name": "Existing", "record_status": "已记录"}],
                "tau3-banking": [
                    {"model_name": "existing", "record_status": "已记录"},
                    {"model_name": "Tau Only", "record_status": "已记录"},
                ],
            }), encoding="utf-8")

            await update_model_status_table(path, ["New Model"], None, data_path)

            status = json.loads(path.read_text(encoding="utf-8"))
        for layer in ("intelligence_index", "terminalbench-4-0", "tau3-banking"):
            self.assertEqual([record["model_name"] for record in status[layer]],
                             ["Existing", "Tau Only", "New Model"])
        self.assertEqual([record["record_status"] for record in status["tau3-banking"]],
                         ["已记录", "已记录", "待更新"])

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

            status = json.loads(path.read_text(encoding="utf-8"))
            records = status["intelligence_index"]
            all_model_data = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual(client.requested, ["Existing", "New Model"])
        self.assertEqual([record["record_status"] for record in records], ["已记录", "已记录"])
        self.assertEqual(set(records[1]), {"model_name", "record_status"})
        self.assertEqual([record["model_name"] for record in status["terminalbench-4-0"]],
                         ["Existing", "New Model"])
        self.assertEqual([record["record_status"] for record in status["terminalbench-4-0"]],
                         ["待更新", "待更新"])
        self.assertEqual([record["model_name"] for record in status["tau3-banking"]],
                         ["Existing", "New Model"])
        self.assertEqual([record["record_status"] for record in status["tau3-banking"]],
                         ["待更新", "待更新"])
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
        self.assertEqual(status["intelligence_index"],
                         [{"model_name": "New Model", "record_status": "已记录"}])
        self.assertEqual(status["tau3-banking"],
                         [{"model_name": "New Model", "record_status": "待更新"}])
        self.assertEqual(status["terminalbench-4-0"],
                         [{"model_name": "New Model", "record_status": "待更新"}])
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
        self.assertEqual(status["intelligence_index"],
                         [{"model_name": "Existing", "record_status": "已记录"}])
        self.assertEqual(status["tau3-banking"],
                         [{"model_name": "Existing", "record_status": "待更新"}])
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
