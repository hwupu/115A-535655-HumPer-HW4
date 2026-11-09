# Sprint Board

Status flow: **To Do → Doing → Review → Done**. Move a line between sections and put the owner's name on it.
Async daily: add one dated line under *Daily log* (`done / next / blocked`).

---

## Sprint 0 — Mon 11-09 → Sun 11-15

**Goal:** env + data ready; one explainer scored end-to-end on 20 images; sanity checks pass.

### To Do
- [ ] B-01c WSL setup — B

### Doing

### Review
- [ ] B-01 scaffold — A (awaiting human read-through)

### Done
- [x] B-08 smoke run — PR #6, approved and merged by A; CPU vs CUDA still open
- [x] B-07 `alignment.py` — PR #5, approved and merged by A; WSL run still open
- [x] B-06 `explain.py` — PR #4, approved and merged by A; WSL run still open
- [x] B-05 `models.py` — PR #3, approved and merged by A; WSL run still open
- [x] B-04 `maps.py` — PR #2, approved and merged by A; WSL run still open
- [x] B-03 MIT1003 + `data.py` + notebook 01 — PR #1, approved and merged by A; WSL run still open
- [x] B-01b git + public GitHub repo; @ALLEN07056 invited — A
- [x] B-02 GenAI policy; D-01..D-12 agreed — approved by A 11-09

### Daily log
- 11-09 A: architecture planned (P-001), scaffold built (P-002).

- 11-09 A: B-03 → Review (P-004). Human blur fitted at σ = 24 px; maps cover early attention only.
- 11-09 A: B-03 merged. B-04 → Review (P-005). Model grid size caps CC (7×7: 0.79, 14×14: 0.94), see D-15.
- 11-09 A: B-04 merged, D-15 accepted, report notes started. B-05 → Review (P-006).
- 11-09 A: B-05 merged, D-16 accepted, B-25 kept (D-18). B-06 → Review (P-007).
- 11-09 A: B-06 merged, D-17 accepted. B-07 → Review (P-008).
- 11-09 A: B-07 merged, D-19 accepted. B-08 → Review (P-009). End-to-end run works; CPU and MPS agree.
- 11-09 A: B-08 merged. B-09 → Review (P-010). All 1003 images done on MPS in about 5 min; no all-zero grids.
- 11-09 A: B-09 merged. B-10 → Review (P-011). Center prior beats every explainer on CC and NSS.
- 11-09 A: B-10 merged. B-11 → Review (P-012). Rollout's anti-center pattern confirmed.
- 11-09 A: B-11 merged. B-13 sheet → Review (P-013). Label before opening any candidate figures.

### Retro
- Keep:
- Change:

---

## Sprint 1 — Mon 11-16 → Sun 11-22

**Goal:** all maps and scores, baselines, candidates, annotations. Revisit the plan after the 11-17 lecture; HW3 is also due 11-17.

### Review
- [ ] B-13 labelling sheet — A→B (D-20 proposed). Then both members label
- [ ] B-12 candidate list built; figures wait until labelling is done

### Done
- [x] B-11 summaries, paired tests, figures — PR #9, approved and merged by A
- [x] B-10 scores and references — PR #8, approved and merged by A
- [x] B-09 full maps and predictions — PR #7, approved and merged by A

## Sprint 2 — Mon 11-23 → Sun 11-29

**Goal:** probes, final cases, factor analysis, full report draft.

## Release — Mon 11-30 → Tue 12-01 (before class)

**Goal:** submitted. **Code freeze Mon 11-30 18:00.**
