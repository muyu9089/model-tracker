import json
import unittest

from app.artificial_analysis import (
    format_thousands,
    parse_catalog,
    parse_index_update_text,
    parse_metric_record,
    parse_tau_banking_catalog,
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

    def test_tau_banking_payload_contains_score_tokens_and_speed(self) -> None:
        record = {
            "id": "1", "slug": "qwen3-8-max", "name": "Qwen3.8 Max",
            "tauBanking": 0.51340206185567,
            "canonicalEvalTokenCounts": {"tauBanking": {"answer": 280112, "reasoning": 639858}},
            "medianCanonicalAnswerOutputSpeed": 52.3251837412224,
        }
        html = flight_html("0:" + json.dumps(record, separators=(",", ":")))
        self.assertEqual(parse_tau_banking_catalog(html), [record])


class TauBankingClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_enrichment_uses_97_tasks_and_marks_missing_models_unmatched(self) -> None:
        record = {
            "id": "1", "slug": "qwen3-8-max", "name": "Qwen3.8 Max",
            "tauBanking": 0.51340206185567,
            "canonicalEvalTokenCounts": {"tauBanking": {"answer": 280112, "reasoning": 639858}},
            "medianCanonicalAnswerOutputSpeed": 52.3251837412224,
        }
        html = flight_html("0:" + json.dumps(record, separators=(",", ":")))
        results = await ArtificialAnalysisClient().enrich_tau_banking_models(
            ["Qwen3.8 Max", "Qwen3.7 Max", "Unrelated Model"], html
        )
        self.assertEqual(results["Qwen3.8 Max"]["tau3_banking_score"], 51.3)
        self.assertEqual(results["Qwen3.8 Max"]["output_tokens_per_task"], "9 K")
        self.assertEqual(results["Qwen3.8 Max"]["time_per_task_minutes"], 3.0)
        self.assertFalse(results["Qwen3.7 Max"]["matched"])
        self.assertFalse(results["Unrelated Model"]["matched"])

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
