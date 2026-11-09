# Process — AI-native Scrum-lite (2 people)

Goal: ship HW4 (report PDF + reproducible code) by **Tue 2026-12-01, before class**,
with the least process that still keeps us honest. Carried over from HW3.

## 1. Team & roles

| Who | Machine | Scrum hat | Dev focus |
|---|---|---|---|
| **Member A** (@hwupu) | macOS, Apple Silicon (MPS) | Product Owner — owns `BACKLOG.md` priority, accepts work | Implements code tickets (with AI agent) |
| **Member B** (@ALLEN07056) | Windows + WSL2 Ubuntu (CUDA or CPU) | Scrum Master — keeps `BOARD.md` current, runs ceremonies | **Code reviewer** for every PR; report & case interpretation |
| **AI agent** (Claude Code, Codex, …) | — | Not a team member — a tool | Drafts code/docs on request; a human owns every output |

@ALLEN07056 reviews all code. @hwupu reviews B's report text. Both annotate images and write the misalignment explanations and the Discussion.

## 2. Cadence

| Sprint | Dates | Goal |
|---|---|---|
| S0 | Mon 11-09 → Sun 11-15 | Env + data + one explainer runs end-to-end; sanity checks |
| S1 | Mon 11-16 → Sun 11-22 | All maps and scores, baselines, candidates, annotations |
| S2 | Mon 11-23 → Sun 11-29 | Probes, final cases, factor analysis, full report draft |
| Release | Mon 11-30 → Tue 12-01 | Freeze, proofread, submit |

HW3 is due 11-17 and the related lecture is also on 11-17, so S0 is deliberately small. Revisit the plan after the lecture.

## 3. Ceremonies (keep it short)

| Ceremony | When | Length | Output |
|---|---|---|---|
| Planning | First day of sprint | 15 min | Items moved Backlog → `BOARD.md` "To Do", owners set |
| Async daily | Any time each day | 1 line each | Update item status on `BOARD.md` (or chat: *done / next / blocked*) |
| Review + Retro | Last day of sprint | 20 min | Demo notebook outputs; 1 keep / 1 change written in `BOARD.md` |

## 4. Workflow for one item

1. Pick an item from **To Do** and put your name on it → **Doing**.
2. If AI is used: work from `AGENTS.md` context, and log significant prompts in `PROMPT_LOG.md`.
3. If a non-trivial choice is made (dataset, explainer config, formula, threshold) → add an entry to `DECISIONS.md`.
4. Work on a branch `b-xx-short-name`. Commit with the item ID in the message, e.g. `B-07: add NSS`.
5. Open a PR to `main` (`gh pr create`). `.github/CODEOWNERS` auto-requests @ALLEN07056 → item moves to **Review**.
6. Reviewer checks the Definition of Done, ideally running `uv run pytest` on WSL, then approves. The author squash-merges → **Done**.

## 5. Definition of Done

- [ ] Runs from a clean clone with `uv sync` + `uv run ...` (no hidden local state)
- [ ] Works on **both** macOS and WSL. The reviewer runs it on the other platform at least once per sprint.
- [ ] Outputs saved under `results/` (CSV/PNG), not only shown in a notebook cell
- [ ] AI-generated code was read and checked by a human (sanity check written down)
- [ ] `PROMPT_LOG.md` / `DECISIONS.md` updated if relevant
- [ ] `uv run pytest` passes
- [ ] PR approved by the reviewer

## 6. AI-native rules

- **Context first.** `AGENTS.md` and `docs/` are the shared memory. `AGENTS.md` is canonical; `CLAUDE.md` just imports it.
- **Humans decide, AI drafts.** Every decision in `DECISIONS.md` has a human owner.
- **Verify, don't trust.** Alignment and explainer code must pass the checks in `ARCHITECTURE.md` §7 before its numbers are used.
- **Log what matters.** Log prompts that produced code, analysis or report text that we kept. Skip trivial Q&A.
- **Policy.** Follow the course Academic Integrity & GenAI policy. Disclose AI use in the report (using `PROMPT_LOG.md` as evidence). Annotations and interpretation of misalignment cases must be our own.

## 7. Git (minimal)

- Repo: **public** at <https://github.com/hwupu/115A-535655-HumPer-HW4>. Never commit secrets/tokens or the dataset. WSL member clones into `~/`, not `/mnt/c/`.
- `.gitattributes` forces LF line endings; WSL member also sets `core.autocrlf input`.
- `main` only changes via reviewed PRs (one per ticket, squash-merge). Docs-only board updates may go straight to `main`.
- `data/` and `results/maps/` are gitignored; `uv.lock` is committed.
- Dependency changes go through `uv add` / `uv remove` only, and the commit includes `uv.lock`. The other member runs `uv sync` after pulling.
- Notebooks: clear heavy outputs before committing. Figures go to `results/figures/`.
