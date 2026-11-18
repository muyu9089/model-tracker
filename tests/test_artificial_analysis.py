import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.artificial_analysis import (
    format_thousands,
    parse_catalog,
    parse_index_update_text,
    parse_metric_record,
    parse_aa_briefcase_catalog,
    parse_automationbench_catalog,
    parse_terminalbench_catalog,
    round_half_up,
    select_model,
    ArtificialAnalysisClient,
)


def flight_html(value: str) -> str:
    return f"<html><script>self.__next_f.push({json.dumps([1, value])})</script></html>"


class ArtificialAnalysisTests(unittest.TestCase):
    def test_updated_badge_exposes_index_description(self) -> None:
        description = "Artificial Analysis Intelligence Index v4.3 incorporates 10 evaluations"
        html = (
            '<span class="rounded-full bg-brand-purple-dark">Updated</span>'
            f'<span style="text-wrap:pretty">{description}</span>'
        )
        self.assertEqual(parse_index_update_text(html), description)
        self.assertIsNone(parse_index_update_text(html.replace("Updated", "New")))

    def test_metric_rounding_uses_half_up(self) -> None:
        self.assertEqual(round_half_up(52.5), 53)
        self.assertEqual(round_half_up(27205.504), 27206)
        self.assertEqual(round_half_up(8.15, 1), 8.2)
        self.assertIsNone(round_half_up(None))

    def test_output_tokens_are_formatted_in_thousands(self) -> None:
        self.assertEqual(format_thousands(27206), "27 K")
        self.assertEqual(format_thousands(27500), "28 K")
        self.assertIsNone(format_thousands(None))

    def test_catalog_and_metrics_parsing(self) -> None:
        record = {
            "id": "1",
            "slug": "demo-70b-high",
            "name": "Demo 70B (high)",
            "release": {"slug": "demo-70b", "name": "Demo 70B"},
            "effort": {"label": "high", "level": 40},
            "parameters": 70,
            "intelligenceIndex": 42.5,
            "intelligenceIndexTimePerTask": 120.0,
            "intelligenceIndexOutputTokensPerTask": {"output": 1234.0},
        }
        html = flight_html("0:" + json.dumps(record, separators=(",", ":")))
        self.assertEqual(parse_catalog(html)[0]["slug"], "demo-70b-high")
        self.assertEqual(parse_metric_record(html, "demo-70b-high")["intelligenceIndex"], 42.5)

    def test_index_score_without_time_is_usable(self) -> None:
        record = {"id": "1", "slug": "demo", "name": "Demo", "intelligenceIndex": 50,
                  "intelligenceIndexTimePerTask": None}
        html = flight_html("0:" + json.dumps(record, separators=(",", ":")))
        self.assertEqual(parse_metric_record(html, "demo"), record)

    def test_new_benchmark_catalogs(self) -> None:
        record = {
            "id": "1", "slug": "qwen3-8-max", "name": "Qwen3.8 Max",
            "briefcaseElo": 1812,
            "automationBenchPartialScore": 0.51340206185567,
        }
        html = flight_html("0:" + json.dumps(record, separators=(",", ":")))
        self.assertEqual(parse_aa_briefcase_catalog(html), [record])
        self.assertEqual(parse_automationbench_catalog(html), [record])

    def test_terminalbench_payload_contains_score_tokens_and_speed(self) -> None:
        record = {
            "id": "1", "slug": "demo-high", "name": "Demo (high)",
            "terminalBench40": 0.5,
            "canonicalEvalTokenCounts": {
                "terminalBench40": {"answer": 66000, "reasoning": 66000}
            },
            "medianCanonicalAnswerOutputSpeed": 100,
        }
        html = flight_html("0:" + json.dumps(record, separators=(",", ":")))
        self.assertEqual(parse_terminalbench_catalog(html), [record])


class NewBenchmarkClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_mapped_score_slug_uses_model_detail_page(self) -> None:
        other = {"id": "1", "slug": "other-model", "name": "Other Model",
                 "briefcaseElo": 1000, "automationBenchPartialScore": 0.1}
        muse = {"id": "2", "slug": "muse-spark-1-3", "name": "Muse Spark 1.3 (max)",
                "intelligenceIndex": None, "briefcaseBreakdown": {"overall": {"elo": 1673.56}},
                "automationBenchPartialScore": 0.579}
        html = flight_html("0:" + json.dumps(other, separators=(",", ":")))
        filtered_html = flight_html("0:" + json.dumps(muse, separators=(",", ":")))
        with patch("app.artificial_analysis.httpx.AsyncClient") as client_class:
            client = client_class.return_value.__aenter__.return_value
            client.get = AsyncMock(return_value=SimpleNamespace(
                status_code=200, text=filtered_html, raise_for_status=lambda: None))
            aa = ArtificialAnalysisClient()
            briefcase = await aa.enrich_aa_briefcase_models(["muse-spark-1-3"], html)
            automation = await aa.enrich_automationbench_models(["muse-spark-1-3"], html)
        self.assertEqual(briefcase["muse-spark-1-3"]["aa_briefcase_score"], 1674)
        self.assertEqual(automation["muse-spark-1-3"]["automationbench_aa_score"], 57.9)
        self.assertEqual(client.get.await_count, 2)
        client.get.assert_awaited_with("/models/muse-spark-1-3")

    async def test_index_slug_missing_from_catalog_uses_detail_page(self) -> None:
        catalog = {"slug": "other-model", "name": "Other Model",
                   "release": {"slug": "other-model"}}
        detail = {"id": "1", "slug": "deepseek-v4-pro", "name": "DeepSeek V4 Pro 0813",
                  "intelligenceIndex": 23, "intelligenceIndexTimePerTask": 150,
                  "intelligenceIndexOutputTokensPerTask": {"answer": 3000, "reasoning": 4500, "output": 9000}}
        html = flight_html("0:" + json.dumps(catalog, separators=(",", ":")))
        detail_html = flight_html("0:" + json.dumps(detail, separators=(",", ":")))
        with patch("app.artificial_analysis.httpx.AsyncClient") as client_class:
            client = client_class.return_value.__aenter__.return_value
            client.get = AsyncMock(return_value=SimpleNamespace(status_code=200, text=detail_html))
            result = await ArtificialAnalysisClient().enrich_models(["deepseek-v4-pro"], html)
        self.assertEqual(result["deepseek-v4-pro"]["intelligence_index"], 23)
        self.assertEqual(result["deepseek-v4-pro"]["output_tokens"], 7500)
        self.assertEqual(result["deepseek-v4-pro"]["time_per_task_minutes"], 2.5)
        client.get.assert_awaited_once_with("/models/deepseek-v4-pro")

    async def test_enrichment_preserves_elo_and_converts_percent(self) -> None:
        record = {
            "id": "1", "slug": "qwen3-8-max", "name": "Qwen3.8 Max",
            "parameters": 235, "effort": {"label": "max", "level": 60},
            "briefcaseElo": 1812,
            "automationBenchPartialScore": 0.51340206185567,
        }
        html = flight_html("0:" + json.dumps(record, separators=(",", ":")))
        results = await ArtificialAnalysisClient().enrich_aa_briefcase_models(
            ["Qwen3.8 Max", "Qwen3.7 Max", "Unrelated Model"], html
        )
        self.assertEqual(results["Qwen3.8 Max"]["aa_briefcase_score"], 1812)
        percent = await ArtificialAnalysisClient().enrich_automationbench_models(["Qwen3.8 Max"], html)
        self.assertEqual(percent["Qwen3.8 Max"]["automationbench_aa_score"], 51.3)
        self.assertEqual(results["Qwen3.8 Max"]["parameters_billions"], 235.0)
        self.assertEqual(results["Qwen3.8 Max"]["reasoning_effort"], "max")
        self.assertFalse(results["Qwen3.7 Max"]["matched"])
        self.assertFalse(results["Unrelated Model"]["matched"])

    async def test_terminalbench_enrichment_uses_66_tasks(self) -> None:
        record = {
            "id": "1", "slug": "demo-high", "name": "Demo (high)",
            "parameters": 70,
            "terminalBench40": 0.5,
            "canonicalEvalTokenCounts": {
                "terminalBench40": {"answer": 66000, "reasoning": 66000}
            },
            "medianCanonicalAnswerOutputSpeed": 100,
        }
        html = flight_html("0:" + json.dumps(record, separators=(",", ":")))
        with patch("app.artificial_analysis.httpx.AsyncClient") as client:
            client.return_value.__aenter__.return_value.get.return_value.status_code = 404
            results = await ArtificialAnalysisClient().enrich_terminalbench_models(
                ["Demo", "Missing"], html
            )
        self.assertEqual(results["Demo"]["terminalbench_4_0_score"], 50.0)
        self.assertEqual(results["Demo"]["output_tokens_per_task"], "2 K")
        self.assertEqual(results["Demo"]["time_per_task_minutes"], 0.3)
        self.assertEqual(results["Demo"]["parameters_billions"], 70.0)
        self.assertEqual(results["Demo"]["reasoning_effort"], "high")
        self.assertFalse(results["Missing"]["matched"])

    async def test_terminalbench_keeps_score_without_speed(self) -> None:
        record = {"id": "1", "slug": "demo", "name": "Demo", "terminalBench40": 0.071}
        html = flight_html("0:" + json.dumps(record, separators=(",", ":")))
        result = await ArtificialAnalysisClient().enrich_terminalbench_models(["Demo"], html)
        self.assertEqual(result["Demo"]["terminalbench_4_0_score"], 7.1)
        self.assertIsNone(result["Demo"]["time_per_task_minutes"])

    def test_largest_parameters_then_highest_effort(self) -> None:
        catalog = [
            {"slug": "demo-7b-max", "name": "Demo 7B (max)", "release": {"slug": "demo", "name": "Demo"}, "parameters": 7, "effort": {"level": 60}},
            {"slug": "demo-70b-high", "name": "Demo 70B (high)", "release": {"slug": "demo", "name": "Demo"}, "parameters": 70, "effort": {"level": 40}},
            {"slug": "demo-70b-max", "name": "Demo 70B (max)", "release": {"slug": "demo", "name": "Demo"}, "parameters": 70, "effort": {"level": 60}},
        ]
        selected, method = select_model("Demo", catalog)
        self.assertEqual(method, "family_exact")
        self.assertEqual(selected["slug"], "demo-70b-max")

    def test_unrelated_model_is_not_selected(self) -> None:
        catalog = [{"slug": "alpha-70b", "name": "Alpha 70B", "release": {"slug": "alpha", "name": "Alpha"}}]
        selected, method = select_model("Completely Different", catalog)
        self.assertIsNone(selected)
        self.assertEqual(method, "unmatched")

    def test_family_match_prefers_trillion_parameter_model(self) -> None:
        catalog = [
            {"slug": "qwen3-8-27b", "name": "Qwen3.8 27B (xhigh)", "release": {"slug": "qwen3-8-27b", "name": "Qwen3.8 27B"}, "effort": {"level": 50}},
            {"slug": "qwen3-8-2-4t-a95b", "name": "Qwen3.8 2.4T A95B", "release": {"slug": "qwen3-8-2-4t-a95b", "name": "Qwen3.8 2.4T A95B"}, "effort": {"level": 40}},
        ]
        selected, method = select_model("Qwen3.8", catalog)
        self.assertEqual(method, "family")
        self.assertEqual(selected["slug"], "qwen3-8-2-4t-a95b")

if __name__ == "__main__":
    unittest.main()
