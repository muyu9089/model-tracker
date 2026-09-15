import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Any

import httpx


_QUALIFIER_PATTERN = re.compile(r"\s*\([^)]*\)\s*$")
_PARAMETER_PATTERN = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*([TBMK])\b", re.IGNORECASE)


def normalize_model_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def round_half_up(value: Any, digits: int = 0) -> int | float | None:
    """按通常意义的四舍五入处理指标，避免 Python round 的银行家舍入。"""
    if not isinstance(value, (int, float)):
        return None
    quantum = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return int(rounded) if digits == 0 else float(rounded)


def format_thousands(value: Any) -> str | None:
    """将数值换算为千位并四舍五入，例如 27,206 -> 27 K。"""
    if not isinstance(value, (int, float)):
        return None
    return f"{round_half_up(value / 1000)} K"


class _NextFlightParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_script = False
        self.payloads: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.in_script = tag == "script"

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_script = False

    def handle_data(self, data: str) -> None:
        prefix = "self.__next_f.push("
        if not self.in_script or not data.startswith(prefix) or not data.endswith(")"):
            return
        try:
            payload = json.loads(data[len(prefix) : -1])
        except json.JSONDecodeError:
            return
        if len(payload) > 1 and isinstance(payload[1], str):
            self.payloads.append(payload[1])


class _IndexUpdateParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._span: str | None = None
        self._parts: list[str] = []
        self.has_updated_badge = False
        self.descriptions: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "span" or self._span is not None:
            return
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if "bg-brand-purple-dark" in classes:
            self._span = "badge"
        elif "text-wrap:pretty" in (attributes.get("style") or "").replace(" ", ""):
            self._span = "description"
        else:
            return
        self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "span" or self._span is None:
            return
        text = " ".join("".join(self._parts).split())
        if self._span == "badge" and text == "Updated":
            self.has_updated_badge = True
        elif self._span == "description" and text:
            self.descriptions.append(text)
        self._span = None
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._span is not None:
            self._parts.append(data)


def _next_flight_text(html: str) -> str:
    parser = _NextFlightParser()
    parser.feed(html)
    return "".join(parser.payloads)


def parse_index_update_text(html: str) -> str | None:
    """Return the index description only when the page shows its Updated badge."""
    parser = _IndexUpdateParser()
    parser.feed(html)
    if not parser.has_updated_badge:
        return None
    return next(
        (text for text in parser.descriptions if "Artificial Analysis Intelligence Index" in text),
        None,
    )


def _decode_objects(text: str, prefix: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    offset = 0
    while True:
        offset = text.find(prefix, offset)
        if offset < 0:
            break
        try:
            value, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            offset += 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        offset = end
    return objects


def parse_catalog(html: str) -> list[dict[str, Any]]:
    """Extract AA's model catalog from the Next.js flight payload."""
    models: dict[str, dict[str, Any]] = {}
    flight_text = _next_flight_text(html)
    objects = _decode_objects(flight_text, '{"slug":"')
    objects.extend(_decode_objects(flight_text, '{"id":"'))
    for model in objects:
        release = model.get("release")
        if not (
            isinstance(model.get("slug"), str)
            and isinstance(model.get("name"), str)
            and isinstance(release, dict)
            and isinstance(release.get("slug"), str)
        ):
            continue
        # Rich records should replace the lightweight catalog copy.
        current = models.get(model["slug"])
        if current is None or len(model) > len(current):
            models[model["slug"]] = model
    return list(models.values())


def parse_metric_record(html: str, slug: str) -> dict[str, Any] | None:
    """Return the rich model record that contains the three requested metrics."""
    for model in _decode_objects(_next_flight_text(html), '{"id":"'):
        if model.get("slug") != slug:
            continue
        tokens = model.get("intelligenceIndexOutputTokensPerTask")
        if (
            model.get("intelligenceIndex") is not None
            and model.get("intelligenceIndexTimePerTask") is not None
            and isinstance(tokens, dict)
        ):
            return model
    return None


def parse_tau_banking_catalog(html: str) -> list[dict[str, Any]]:
    """Extract evaluated models and their 𝜏³-Banking inputs from the page payload."""
    return [
        model for model in _decode_objects(_next_flight_text(html), '{"id":"')
        if isinstance(model.get("slug"), str)
        and isinstance(model.get("name"), str)
        and isinstance(model.get("tauBanking"), (int, float))
    ]


def _parameter_billions(model: dict[str, Any]) -> float:
    explicit = model.get("parameters")
    if isinstance(explicit, (int, float)):
        return float(explicit)
    text = " ".join(
        str(value)
        for value in (
            model.get("name", ""),
            model.get("slug", ""),
            (model.get("release") or {}).get("name", ""),
        )
    )
    multipliers = {"T": 1000.0, "B": 1.0, "M": 0.001, "K": 0.000001}
    values = [float(number) * multipliers[unit.upper()] for number, unit in _PARAMETER_PATTERN.findall(text)]
    return max(values, default=-1.0)


def _effort_level(model: dict[str, Any]) -> float:
    effort = model.get("effort")
    if isinstance(effort, dict) and isinstance(effort.get("level"), (int, float)):
        return float(effort["level"])
    labels = {"none": 0, "minimal": 10, "low": 20, "medium": 30, "high": 40, "xhigh": 50, "max": 60}
    text = f"{model.get('name', '')} {model.get('slug', '')}".lower()
    return float(max((level for label, level in labels.items() if label in text), default=-1))


def _identity_values(model: dict[str, Any]) -> list[str]:
    release = model.get("release") or {}
    full_name = str(model.get("name", ""))
    return [
        str(model.get("slug", "")),
        full_name,
        _QUALIFIER_PATTERN.sub("", full_name),
        str(release.get("slug", "")),
        str(release.get("name", "")),
    ]


def _match_score(query: str, model: dict[str, Any]) -> tuple[float, str]:
    normalized_query = normalize_model_name(query)
    best_score = 0.0
    method = "unmatched"
    identities = _identity_values(model)
    for index, value in enumerate(identities):
        candidate = normalize_model_name(value)
        if not candidate:
            continue
        if normalized_query == candidate:
            if index in (0, 1):
                return 110.0, "exact"
            best_score, method = max((best_score, method), (100.0, "family_exact"), key=lambda item: item[0])
            continue
        shorter, longer = sorted((normalized_query, candidate), key=len)
        if len(shorter) >= 5 and shorter in longer:
            score = 86.0 - min(15.0, (len(longer) - len(shorter)) * 0.5)
            if score > best_score:
                best_score, method = score, "family"
        ratio = SequenceMatcher(None, normalized_query, candidate).ratio()
        if ratio >= 0.72 and ratio * 80 > best_score:
            best_score, method = ratio * 80, "fuzzy"
    return best_score, method


def select_model(query: str, catalog: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """按原有规则匹配名称；同系列优先最大参数量，其次最高思考深度。"""
    scored = [(model, *_match_score(query, model)) for model in catalog]
    scored = [item for item in scored if item[1] >= 57.6]
    if not scored:
        return None, "unmatched"
    highest_match = max(item[1] for item in scored)
    if highest_match >= 110:
        candidates = [item for item in scored if item[1] == highest_match]
        candidates.sort(key=lambda item: (_parameter_billions(item[0]), _effort_level(item[0])), reverse=True)
        selected, _, method = candidates[0]
        return selected, method
    family_exact = [item for item in scored if item[2] == "family_exact"]
    family = [item for item in scored if item[2] == "family"]
    if family_exact:
        candidates = family_exact
    elif family:
        candidates = family
    else:
        candidates = [item for item in scored if item[1] >= highest_match - 5.0]
    candidates.sort(
        key=lambda item: (
            _parameter_billions(item[0]),
            _effort_level(item[0]),
            item[1],
            not bool(item[0].get("deprecated")),
        ),
        reverse=True,
    )
    selected, _, method = candidates[0]
    return selected, method


@dataclass
class ArtificialAnalysisClient:
    base_url: str = "https://artificialanalysis.ai"
    timeout_seconds: float = 30.0

    async def fetch_models_html(self) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AIHotModelTracker/1.1)"}
        async with httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers=headers,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            response = await client.get("/models")
            response.raise_for_status()
            return response.text

    async def fetch_tau_banking_html(self) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AIHotModelTracker/1.1)"}
        async with httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers=headers,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            response = await client.get("/evaluations/tau3-banking")
            response.raise_for_status()
            return response.text

    async def enrich_tau_banking_models(
        self, names: list[str], html: str
    ) -> dict[str, dict[str, Any]]:
        catalog = parse_tau_banking_catalog(html)
        if not catalog:
            raise RuntimeError("未能从 Artificial Analysis 页面解析 𝜏³-Banking 数据")
        results = {}
        for name in names:
            key = normalize_model_name(name)
            exact_models = [model for model in catalog if key in {
                normalize_model_name(value)
                for value in (
                    model["slug"], model["name"], model.get("shortName") or "",
                    _QUALIFIER_PATTERN.sub("", model["name"]),
                    _QUALIFIER_PATTERN.sub("", model.get("shortName") or ""),
                )
            }]
            selected, method = select_model(name, exact_models)
            if selected is None:
                results[name] = {"matched": False, "error": "未匹配到 𝜏³-Banking 模型"}
                continue
            counts = (selected.get("canonicalEvalTokenCounts") or {}).get("tauBanking") or {}
            answer, reasoning = counts.get("answer"), counts.get("reasoning")
            speed = selected.get("medianCanonicalAnswerOutputSpeed")
            if not all(isinstance(value, (int, float)) for value in (answer, reasoning, speed)) or speed <= 0:
                results[name] = {"matched": False, "error": "已匹配模型，但缺少完整 token 或速度数据"}
                continue
            tokens_per_task = (answer + reasoning) / 97
            results[name] = {
                "matched": True,
                "model_name": selected["name"],
                "slug": selected["slug"],
                "match_method": method,
                "tau3_banking_score": round_half_up(selected["tauBanking"] * 100, 1),
                "output_tokens_per_task": format_thousands(tokens_per_task),
                "time_per_task_minutes": round_half_up(tokens_per_task / speed / 60, 1),
                "source_url": f"{self.base_url.rstrip('/')}/evaluations/tau3-banking",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        return results

    async def enrich_models(
        self,
        names: list[str],
        models_html: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AIHotModelTracker/1.1)"}
        async with httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers=headers,
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            if models_html is None:
                response = await client.get("/models")
                response.raise_for_status()
                models_html = response.text
            catalog = parse_catalog(models_html)
            if not catalog:
                raise RuntimeError("未能从 Artificial Analysis 页面解析模型目录")

            semaphore = asyncio.Semaphore(5)

            async def enrich(name: str) -> tuple[str, dict[str, Any]]:
                selected, method = select_model(name, catalog)
                if selected is None:
                    return name, {"matched": False, "query_name": name, "match_method": method}
                slug = str(selected["slug"])
                metric_record = selected if parse_metric_record(models_html, slug) else None
                if metric_record is None:
                    async with semaphore:
                        detail = await client.get(f"/models/{slug}")
                        detail.raise_for_status()
                    metric_record = parse_metric_record(detail.text, slug)
                if metric_record is None:
                    return name, {
                        "matched": False,
                        "query_name": name,
                        "match_method": method,
                        "candidate_slug": slug,
                        "error": "已匹配模型，但未解析到完整指标",
                    }
                output_tokens = metric_record["intelligenceIndexOutputTokensPerTask"].get("output")
                time_seconds = metric_record.get("intelligenceIndexTimePerTask")
                effort = metric_record.get("effort") or {}
                parameters_billions = _parameter_billions(metric_record)
                return name, {
                    "matched": True,
                    "query_name": name,
                    "match_method": method,
                    "model_name": metric_record.get("name"),
                    "slug": slug,
                    "parameters_billions": parameters_billions if parameters_billions >= 0 else None,
                    "reasoning_effort": effort.get("label") or effort.get("slug"),
                    "reasoning_effort_level": effort.get("level"),
                    "intelligence_index": round_half_up(metric_record.get("intelligenceIndex")),
                    "output_tokens_per_task": format_thousands(output_tokens),
                    "time_per_task_minutes": round_half_up(time_seconds / 60, 1)
                    if isinstance(time_seconds, (int, float))
                    else None,
                    "source_url": f"{self.base_url.rstrip('/')}/models/{slug}",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }

            return dict(await asyncio.gather(*(enrich(name) for name in names)))
