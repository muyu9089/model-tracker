# Model Overview Edit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Edit model overview fields inline and save them atomically.

**Architecture:** Add one repository transaction method and a token-protected PUT route. Render row inputs in the existing HTML page and submit JSON with the existing token.

**Tech Stack:** FastAPI, Pydantic, SQLite, vanilla JavaScript.

**Spec:** `docs/superpowers/specs/2026-10-29-model-overview-edit-design.md`

## Global Constraints

- No new dependency or schema migration.
- Preserve profile fields and benchmark metadata outside the overview.

## Review Focus

- Duplicate or blank names must fail without partial updates.
- Unknown benchmark codes and nonfinite scores must fail.
- Blank score must clear the score and mark the state pending.
- Missing profile and result rows must be created when edited.
- Failed saves must leave entered values on screen.

### Task 1: Atomic update

**Files:** `app/db/repositories.py`, `tests/test_main.py`

- [x] Add `TrackerRepository.update_overview(model_id, name, organization, open_source, region, scores)`.
- [x] Validate name, benchmark codes and scores before writes; update all fields in one transaction while preserving unrelated columns.
- [x] Cover successful updates and rejected duplicate/invalid input with a focused test.

### Task 2: Page and route

**Files:** `app/main.py`, `tests/test_main.py`

- [x] Add a token-protected `PUT /models/{model_id}` accepting only overview fields.
- [x] Render an edit button, row inputs and save/cancel behavior on model overview.
- [x] Render server errors without discarding inputs; check UI markup and route behavior.
