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
- [ ] B-08 smoke run — A→B
- [ ] B-01 scaffold — A (awaiting human read-through)

### Done
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

### Retro
- Keep:
- Change:

---

## Sprint 1 — Mon 11-16 → Sun 11-22

**Goal:** all maps and scores, baselines, candidates, annotations. Revisit the plan after the 11-17 lecture; HW3 is also due 11-17.

## Sprint 2 — Mon 11-23 → Sun 11-29

**Goal:** probes, final cases, factor analysis, full report draft.

## Release — Mon 11-30 → Tue 12-01 (before class)

**Goal:** submitted. **Code freeze Mon 11-30 18:00.**
