# HW4 — Human–Machine Attention Alignment Case Study

NYCU 535655 *Human Perceptual-Centric Vision and Multimodal Representation Learning*, Fall 2026.
Due **Tue 2026-12-01, before class**. Spec: [`docs/references/HW4.pdf`](docs/references/HW4.pdf).

We compare human eye-tracking attention on MIT1003 with explanations from pretrained vision models (ResNet-50 Grad-CAM, ViT-B/16 Grad-CAM, ViT-B/16 Attention Rollout), measure alignment with CC and NSS against center-bias baselines, and analyse where and why the two disagree.

> **Status:** project scaffold only (B-01). The dataset, explainers and analysis are not implemented yet.

## Setup

Works on **macOS (Apple Silicon)** and **Windows via WSL2 (Ubuntu)**. Only inference is needed, so no cloud GPU is required.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS & WSL
```

### 2. Sync

```bash
uv sync                     # creates .venv from uv.lock (Python 3.11 auto-downloaded)
uv run python -c "import humper; print(humper.get_device())"
uv run pytest
```

**WSL notes**
- Clone inside the Linux filesystem (e.g. `~/code/`), **not** under `/mnt/c/...`. File I/O on `/mnt/c` is very slow.
- On Linux, `uv sync` installs the CUDA build of PyTorch (a large download). Without a GPU, `get_device()` falls back to `cpu`.
- Set `git config --global core.autocrlf input`. `.gitattributes` already forces LF.

**macOS notes**
- `get_device()` returns `mps`. Unsupported ops fall back to CPU automatically.
- Force a device with `HUMPER_DEVICE=cpu uv run ...`.

### 3. Get the data

```bash
uv run python scripts/download_data.py    # MIT1003, about 245 MB; verifies checksums and file counts
```

`data/` is gitignored. Never commit the dataset.

## Labelling (B-13)

```bash
uv run python scripts/build_label_sheet.py
```

Open `results/labeling/label_sheet.html`, label on your own, then save the downloaded CSV as `annotations/labels_<name>.csv`. Don't open `results/failures/` or any heatmap figures until you've finished.

## Repo layout

```
src/humper/    reusable code
scripts/       long-running jobs
notebooks/     analysis; reads from results/
annotations/   human-written labels and case notes
tests/         sanity checks
data/          dataset (gitignored)
results/       scores, tables, figures; maps/ is gitignored
report/        report source + final PDF
docs/          process, architecture, decisions, prompt log, scrum board
```

## How we work

AI-native Scrum-lite. See [`docs/PROCESS.md`](docs/PROCESS.md).
- Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Current sprint: [`docs/scrum/BOARD.md`](docs/scrum/BOARD.md)
- Backlog: [`docs/scrum/BACKLOG.md`](docs/scrum/BACKLOG.md)
- Decisions: [`docs/DECISIONS.md`](docs/DECISIONS.md)
- AI usage log: [`docs/PROMPT_LOG.md`](docs/PROMPT_LOG.md)
- Agent context: [`AGENTS.md`](AGENTS.md) (`CLAUDE.md` imports it)
