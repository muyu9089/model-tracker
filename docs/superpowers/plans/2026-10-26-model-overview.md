# Model Overview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only “模型总览” menu showing every model, its benchmarks and scores, organization, open-source status, and region.

**Architecture:** Extend the existing `/` FastAPI HTML page and use `TrackerRepository.rows` to join model, profile, benchmark, and result data. Keep the raw-table view intact.

**Tech Stack:** Python, FastAPI, SQLite, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-26-model-overview-design.md`

## Global Constraints

- One row per model; list each benchmark with its corresponding score.
- Missing data displays “—”; show all models, including those without profiles or results.
- Escape database text as HTML; keep raw-table pages limited to 200 rows.
- Add no dependency.

## Review Focus

- Model without a profile remains visible with “—” metadata.
- Model without a result remains visible with its benchmark and “—” score.
- HTML-like model or organization text is escaped.
- More than 200 models are all visible in the overview.
- Raw-table navigation still renders.

---

### Task 1: Model overview in existing page

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `TrackerRepository.rows(query, parameters=())` and `_cell(value)`.
- Produces: `GET /?table=overview` in `database_view`, with the existing default `GET /` and raw table links preserved.

- [x] **Step 1: Write a failing page test**

Add `test_overview_shows_all_models_and_benchmarks` with a temporary database containing 201 models, one profile, one recorded score, and HTML-like text. Assert one row per model, all 201 names, all three benchmark labels, the score, metadata, “—” for missing values, escaped text, and a working raw-table link.

- [x] **Step 2: Run the test and verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_main.DatabasePageTests.test_overview_shows_all_models_and_benchmarks -v`

- [x] **Step 3: Implement the page**

In `app/main.py`, add the overview option to the existing navigation. For `table="overview"`, query all models and their optional profiles plus benchmark states/results; render one HTML table row per model with aligned benchmark and score lists. Reuse `_cell` for data and retain existing raw-table rendering.

- [x] **Step 4: Verify**

Run the new test and `python -m unittest discover -s tests`; confirm all pass. Render the page once and inspect the visible table/navigation.
