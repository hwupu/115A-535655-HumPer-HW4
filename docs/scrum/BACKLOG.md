# Product Backlog

Owner: Member A (PO). Ordered by priority within each epic.
Code tickets: implemented by A (+AI), reviewed by B via PR. `A→B` = A implements, B reviews.
Size: **S** ≤ 2h, **M** ≈ half day, **L** ≈ 1 day. Rubric points in brackets.

Rubric: Understanding 15, Setup/Repro 15, Quant 20, Misalign 20, Interpretation 20, Clarity 10.

| ID | Item | Size | Sprint | Owner | Rubric |
|---|---|---|---|---|---|
| **Epic 1 — Setup** |||||
| B-01 | Scaffold: uv project, `.gitignore`/`.gitattributes`, layout, README, AGENTS.md, docs | S | S0 | A | Setup [15] |
| B-01b | `git init`, GitHub repo, add B as collaborator | S | S0 | A | Setup [15] |
| B-01c | WSL setup on B's machine: clone in `~/`, `uv sync`, `get_device()`, pytest | S | S0 | B | Setup [15] |
| B-02 | Confirm the course GenAI policy; agree to D-01..D-12 | S | S0 | B | — |
| B-03 | `download_data.py` + `data.py` + `01_data_check` (file formats, fixation scaling overlay, σ in pixels for D-05) | M | S0 | A→B | Setup [15] |
| **Epic 2 — Maps and correctness** |||||
| B-04 | `maps.py` spatial pipeline + tests (D-03) | M | S0 | A→B | Setup [15] |
| B-05 | `models.py` registry, pin timm IDs, prediction sanity after no-crop resize | S | S0 | A→B | Understanding [15] |
| B-06 | `explain.py`: Grad-CAM (ResNet, ViT) + Rollout, with tests (D-02, D-08) | M | S0 | A→B | Understanding [15] |
| B-07 | `alignment.py`: CC, NSS, sAUC + metric tests (D-04) | M | S0 | A→B | Quant [20] |
| B-08 | End-to-end smoke run on 20 images; CPU vs accelerator check | S | S0 | A→B | Setup [15] |
| **Epic 3 — Quantitative evaluation** |||||
| B-09 | `compute_maps.py` full run (resumable) + `predictions.csv` + run info | M | S1 | A | Quant [20] |
| B-10 | `compute_scores.py` + `center`, `shuffled` and `grid` references (D-15) → `scores.csv` | M | S1 | A→B | Quant [20] |
| B-11 | `evaluate.py`: summaries, bootstrap CIs, paired tests; `02_alignment` figure and table | M | S1 | A→B | Quant [20] |
| **Epic 4 — Misalignment cases** |||||
| B-12 | `failures.py` + candidate figures (`03_failure_cases`) | M | S1 | A→B | Misalign [20] |
| B-13 | ~~Blind hand labelling~~ dropped (D-21) | — | — | — | — |
| B-14 | `probes.py`, `run_probes.py`: occlusion on candidates, model randomization (D-07) | M | S2 | A→B | Interpretation [20] |
| B-15 | Pick ≥ 5 final cases, record reasons in `annotations/case_notes.csv`, final figures (`04_selected_cases`) | M | S2 | A+B | Misalign [20] |
| B-16 | Explanation per case, team-written | L | S2 | A+B | Interpretation [20] |
| B-17 | `05_factors`: alignment vs confidence, model agreement, fixation concentration (D-21) | M | S2 | A→B | Interpretation [20] |
| **Epic 5 — Report & submission** |||||
| B-18 | Report skeleton (5 required sections); choose report tool | S | S1 | B | Clarity [10] |
| B-19 | Draft §1 Dataset and Model, §2 Experimental Setup (see `report/report_notes.md`) | M | S2 | A | Setup [15] |
| B-20 | Draft §3 Quantitative, §4 Cases, §5 Discussion (limitation + follow-up experiment; see `report/report_notes.md`) | L | S2 | A+B | Interpretation [20] |
| B-21 | Contribution Statement + GenAI disclosure; verify all citations | S | Rel | B | — |
| B-22 | Clean-clone reproduction on **both** macOS and WSL; README up to date | S | Rel | A+B | Setup [15] |
| B-23 | Page check (≤ 4 pp), proofread, submit PDF + code | S | Rel | A+B | Clarity [10] |
| **Stretch (cut order: `ARCHITECTURE.md` §8)** |||||
| B-24 | Model-randomization wording and figure in the report, beyond the B-14 number | S | S2 | A→B | Understanding [15] |
| B-25 | Center-crop sensitivity on a subset (D-03) | S | S2 | A→B | Setup [15] |
| B-26 | One time-boxed spike: per-subject fixations from `DATA.zip` via Octave → inter-observer `ceiling` | M | S1 | A→B | Quant [20] |
| B-27 | CLIP image encoder as a further model | M | S2 | A→B | Understanding [15] |
