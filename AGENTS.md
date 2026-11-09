# AGENTS.md — AI agent context for HW4

This is the canonical agent file, read by Codex, Cursor, Copilot, etc.
`CLAUDE.md` only contains `@AGENTS.md`, so Claude Code loads the same content. **Edit only this file.**

**Task:** NYCU 535655 HW4: Human–Machine Attention Alignment Case Study (spec: `docs/references/HW4.pdf`).
Due **2026-12-01 before class**. Report: 3 pages recommended, 4 maximum, excluding references and the Contribution Statement.
Team of 2: Member A (@hwupu, macOS, Apple Silicon) and Member B (@ALLEN07056, Windows + WSL2).

## Read first
- `README.md`: setup & run commands
- `docs/PROCESS.md`: how we work (Scrum-lite, Definition of Done, AI rules)
- `docs/ARCHITECTURE.md`: layout, data flow, spatial pipeline, interfaces, sanity checks
- `docs/DECISIONS.md`: what is already decided; don't re-litigate `Accepted` items
- `docs/scrum/BOARD.md`: current sprint; `docs/scrum/BACKLOG.md`: all items

## Rules for the agent
- Work on a backlog item (`B-xx`). Reference it in commits.
- After producing kept output, append a row to `docs/PROMPT_LOG.md`.
- For a new non-trivial choice, add a `Proposed` row to `docs/DECISIONS.md`. Don't silently decide.
- Keep reusable logic in `src/humper/`. Notebooks stay thin and import from it.
- All resizing, normalization and smoothing lives in `maps.py`. The report must be able to describe it from that one file.
- Each model uses its own mean/std from `models.py`.
- Numbers are not used until the checks in `docs/ARCHITECTURE.md` §7 pass.

## Interpretation rules
- MIT1003 observers free-viewed; models classified. Low alignment is not automatically a model failure.
- Write "the explanation highlights X", not "the model looks at X because…", unless a probe supports it. Attribution is not a full explanation of model reasoning (spec §6).
- Report alignment scores alongside the `center` baseline.
- Annotators must not see heatmaps or scores while labelling `annotations/labels.csv`. Factor statistics use the random sample only.
- Do not write the perceptual explanations of misalignment cases as final text. Draft observations only; humans own the interpretation.
- Check every citation against the actual paper before it enters the report.

## Cross-platform rules (macOS + WSL)
- Get the device via `humper.get_device()` (cuda > mps > cpu; override with `HUMPER_DEVICE`). Never hard-code `"mps"` or `"cuda"`.
- Use `pathlib.Path` relative to the repo root. No absolute paths, no `\` separators.
- Commands go through `uv run ...`. No OS-specific shell scripts unless they're POSIX `sh`.
- Results must not depend on the device: compare CPU vs MPS/CUDA on a few images when adding an explainer.

## Env
- uv + Python 3.11 (`.python-version`). Dependencies are in `pyproject.toml`; `uv.lock` is committed and universal (macOS + Linux).
- `data/` and `results/maps/` are gitignored.
