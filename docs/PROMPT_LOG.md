# Prompt Log

Log prompts whose output we **kept** (code, analysis, report text, or a decision). Skip trivial Q&A.
This is our evidence for the course GenAI disclosure.

| ID | Date | Who | Tool / Model | Item | Prompt (summary) | Output → where | Human check |
|---|---|---|---|---|---|---|---|
| P-001 | 2026-11-09 | A | Claude Code / Opus 5.5 | — | Follow the HW3 project's development process for HW4; plan the architecture from HW4.pdf | `docs/ARCHITECTURE.md`, D-01..D-08 | A confirmed dataset, explainers, team and folder; B review pending |
| P-002 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-01 | "yes to all": build the rest of the HW3-style scaffold | `AGENTS.md`, `CLAUDE.md`, `README.md`, `docs/*`, `pyproject.toml`, `uv.lock`, `src/humper/device.py`, `tests/test_device.py`, D-09..D-12 | Pending human read-through |
| P-003 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-01b, B-02 | "set up git and GitHub; approved, close B-02" | Git repo, private GitHub repo, D-01..D-12 accepted | A approved decisions and GenAI policy check |
| P-004 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-03 | "go ahead" with MIT1003 download, loader and data check | `scripts/download_data.py`, `src/humper/{data,maps}.py`, `tests/test_data.py`, `notebooks/01_data_check.ipynb`, `results/tables/*`, D-13 | 10 tests pass; agent inspected overlays; σ fit and 1003-image CC asserted in notebook. Human review pending (PR) |
| P-005 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-04 | "approved by me. merge and proceed" | PR #1 merged; `src/humper/maps.py`, `tests/test_maps.py`, D-15 | Tests pass; grid-size effect measured on all 1003 maps. Human review pending (PR) |
| P-006 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-05 | "note the grid-size limit and solution for the report; approved, merge, proceed" | PR #2 merged; `report/report_notes.md`; `src/humper/models.py`, `tests/test_models.py`, D-16 | Agent inspected predictions on 12 images; tests pass. Human review pending (PR) |
| P-007 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-06 | "yes, proceed" | PR #3 merged; `src/humper/explain.py`, `tests/test_explain.py`, D-17, D-18 | Worked examples, independent Grad-CAM cross-check, 6 images inspected by agent; 49 tests pass. Human review pending (PR) |
| P-008 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-07 | "yes, proceed" | PR #4 merged; `src/humper/alignment.py`, `tests/test_alignment.py`, D-19 | Worked examples, SciPy cross-checks, center-prior test; 66 tests pass. Human review pending (PR) |
| P-009 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-08 | "proceed" | PR #5 merged; `scripts/smoke_run.py`, `alignment.other_fixations`, cross-device test | 20-image run on CPU and MPS compared; tests pass. Human review pending (PR) |
| P-010 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-09 | "yes. proceed" | PR #6 merged; `scripts/compute_maps.py`, `src/humper/store.py`, `tests/test_store.py`, `results/predictions.csv`, `results/maps_run_info.json` | Resume tested; all 1003 records checked for shape and finite values; predictions match the smoke run. Human review pending (PR) |
| P-011 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-10 | "ok. proceed" | PR #7 merged; `src/humper/evaluate.py`, `scripts/compute_scores.py`, `tests/test_evaluate.py`, `results/scores.csv`, `results/scores_run_info.json` | Grid references match the B-04 estimates; committed rows reproduce from stored maps; 77 tests pass. Human review pending (PR) |
| P-012 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-11 | "yes" (merge and continue) | PR #8 merged; `evaluate.summary/paired/holm/center_similarity`, `notebooks/02_alignment.ipynb`, `results/tables/*`, `results/figures/alignment/*` | Bootstrap, Wilcoxon and Holm tested; figures inspected by agent; 81 tests pass. Human review pending (PR) |
| P-013 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-12, B-13 | "yes. build labels" | PR #9 merged; `src/humper/{failures,labeling}.py`, `src/humper/label_sheet.html`, `scripts/build_label_sheet.py`, `tests/test_labeling.py`, `annotations/sheet_ids.csv`, D-20 | Selection tests with planted cases; page logic run in a simulated DOM, not yet in a real browser. Labels are human work |
| P-014 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-13 | "it is painful to label, lets drop it" | D-21; labelling code removed; `scripts/select_candidates.py`; docs updated | A tried the sheet and decided; tests pass |

## Template

```
| P-xxx | YYYY-MM-DD | A/B | <tool/model> | B-xx | <1-line summary of prompt> | <file / section> | <how we verified> |
```
