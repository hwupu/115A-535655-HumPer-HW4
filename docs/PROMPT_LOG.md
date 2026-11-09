# Prompt Log

Log prompts whose output we **kept** (code, analysis, report text, or a decision). Skip trivial Q&A.
This is our evidence for the course GenAI disclosure.

| ID | Date | Who | Tool / Model | Item | Prompt (summary) | Output → where | Human check |
|---|---|---|---|---|---|---|---|
| P-001 | 2026-11-09 | A | Claude Code / Opus 5.5 | — | Follow the HW3 project's development process for HW4; plan the architecture from HW4.pdf | `docs/ARCHITECTURE.md`, D-01..D-08 | A confirmed dataset, explainers, team and folder; B review pending |
| P-002 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-01 | "yes to all": build the rest of the HW3-style scaffold | `AGENTS.md`, `CLAUDE.md`, `README.md`, `docs/*`, `pyproject.toml`, `uv.lock`, `src/humper/device.py`, `tests/test_device.py`, D-09..D-12 | Pending human read-through |
| P-003 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-01b, B-02 | "set up git and GitHub; approved, close B-02" | Git repo, private GitHub repo, D-01..D-12 accepted | A approved decisions and GenAI policy check |
| P-004 | 2026-11-09 | A | Claude Code / Opus 5.5 | B-03 | "go ahead" with MIT1003 download, loader and data check | `scripts/download_data.py`, `src/humper/{data,maps}.py`, `tests/test_data.py`, `notebooks/01_data_check.ipynb`, `results/tables/*`, D-13 | 10 tests pass; agent inspected overlays; σ fit and 1003-image CC asserted in notebook. Human review pending (PR) |

## Template

```
| P-xxx | YYYY-MM-DD | A/B | <tool/model> | B-xx | <1-line summary of prompt> | <file / section> | <how we verified> |
```
