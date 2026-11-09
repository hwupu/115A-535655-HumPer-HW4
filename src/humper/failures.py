"""Quantitative shortlist of low- and high-alignment images. Humans make the final choice."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from humper import explain
from humper.evaluate import holm

# Share of images with the fewest fixation points excluded as less reliable.
LOW_FIXATION_QUANTILE = 0.05
SHARED_LOW_PERCENTILE = 0.25


def alignment_percentiles(scores: pd.DataFrame) -> pd.DataFrame:
    """Return one row per image and explainer with its combined CC + NSS percentile.

    Percentiles are within each explainer, so one metric's scale cannot dominate.
    `below_center` marks images where the explanation is below the center prior on CC.
    """
    names = list(explain.EXPLAINERS)
    part = scores[scores.explainer.isin(names)].copy()
    grouped = part.groupby("explainer")
    part["alignment_pct"] = (
        grouped.cc.rank(pct=True) + grouped.nss.rank(pct=True)
    ) / 2
    center = scores[scores.explainer == "center"].set_index("image_id").cc
    part["below_center"] = part.cc < part.image_id.map(center)
    return part[["image_id", "explainer", "cc", "nss", "sauc", "alignment_pct", "below_center"]]


def category_effects(
    scores: pd.DataFrame, flags: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Return, per explainer and category, how alignment inside compares with outside.

    `difference` is mean inside minus mean outside on `alignment_pct`, so each explainer
    is measured against its own performance. Negative means the category is harder.
    Most negative comes first; `holm_p` covers every explainer and category tested.
    """
    table = alignment_percentiles(scores)
    rng = np.random.default_rng(seed)
    rows = []
    for explainer, part in table.groupby("explainer"):
        part = part.set_index("image_id")
        for category in flags.columns:
            mask = flags[category].reindex(part.index).to_numpy(dtype=bool)
            inside, outside = part.alignment_pct[mask].to_numpy(), part.alignment_pct[~mask].to_numpy()
            boot = (
                inside[rng.integers(0, len(inside), (n_boot, len(inside)))].mean(axis=1)
                - outside[rng.integers(0, len(outside), (n_boot, len(outside)))].mean(axis=1)
            )
            rows.append({
                "explainer": explainer,
                "category": category,
                "n_inside": len(inside),
                "mean_inside": inside.mean(),
                "mean_outside": outside.mean(),
                "difference": inside.mean() - outside.mean(),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "mannwhitney_p": float(mannwhitneyu(inside, outside).pvalue),
            })
    effects = pd.DataFrame(rows)
    effects["holm_p"] = holm(effects.mannwhitney_p.to_numpy())
    return effects.sort_values("difference", kind="stable").reset_index(drop=True)


def category_cases(
    scores: pd.DataFrame,
    table: pd.DataFrame,
    flags: pd.DataFrame,
    effects: pd.DataFrame,
    per_pair: int = 3,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Return candidate cases for each category effect that survives Holm correction.

    A negative effect makes that explainer `weak`, paired with the best other explainer in
    the category. A positive effect makes it `strong`, paired with the lowest other one.
    Images are ranked by how far `weak` falls behind `strong`, each used at most once.
    """
    percentiles = alignment_percentiles(scores)
    percentiles = percentiles[percentiles.image_id.isin(reliable_images(table))]
    wide = percentiles.pivot(index="image_id", columns="explainer", values="alignment_pct")
    supported = effects[effects.holm_p < alpha]
    supported = supported.loc[supported.difference.abs().sort_values(ascending=False, kind="stable").index]
    used: set[str] = set()
    seen: set[tuple[str, str, str]] = set()
    rows = []
    for effect in supported.itertuples():
        others = effects[(effects.category == effect.category) & (effects.explainer != effect.explainer)]
        if effect.difference < 0:
            weak, strong = effect.explainer, others.loc[others.mean_inside.idxmax(), "explainer"]
        else:
            weak, strong = others.loc[others.mean_inside.idxmin(), "explainer"], effect.explainer
        if (effect.category, weak, strong) in seen:
            continue
        seen.add((effect.category, weak, strong))
        inside = wide[wide.index.isin(flags.index[flags[effect.category]])]
        gap = (inside[strong] - inside[weak]).sort_values(ascending=False, kind="stable")
        taken = 0
        for image_id, value in gap.items():
            if taken == per_pair:
                break
            if image_id in used:
                continue
            used.add(image_id)
            taken += 1
            rows.append({
                "category": effect.category, "rank": taken, "image_id": image_id,
                "weak": weak, "strong": strong, "effect": effect.difference, "holm_p": effect.holm_p,
                "gap": value, "pct_weak": inside.loc[image_id, weak], "pct_strong": inside.loc[image_id, strong],
            })
    return pd.DataFrame(rows)


def reliable_images(table: pd.DataFrame) -> set[str]:
    """Return images with enough fixation points to trust their human map."""
    cutoff = table.n_fixations.quantile(LOW_FIXATION_QUANTILE)
    return set(table[table.n_fixations >= cutoff].image_id)


def select_candidates(
    scores: pd.DataFrame, table: pd.DataFrame, per_group: int = 10, contrast: int = 5
) -> pd.DataFrame:
    """Shortlist images, each at most once.

    `shared_low` images are low for every explainer, then each explainer gets its own
    lowest images, then `shared_high` images supply the high-alignment contrast.
    """
    percentiles = alignment_percentiles(scores)
    percentiles = percentiles[percentiles.image_id.isin(reliable_images(table))]
    wide = percentiles.pivot(index="image_id", columns="explainer", values="alignment_pct")
    chosen: dict[str, str] = {}

    def take(order: pd.Index, reason: str, count: int) -> None:
        added = 0
        for image_id in order:
            if added == count:
                break
            if image_id not in chosen:
                chosen[image_id] = reason
                added += 1

    take(wide.max(axis=1).sort_values(kind="stable").index, "shared_low", per_group)
    for name in explain.EXPLAINERS:
        take(wide[name].sort_values(kind="stable").index, name, per_group)
    take(wide.min(axis=1).sort_values(ascending=False, kind="stable").index, "shared_high", contrast)

    rows = wide.loc[list(chosen)].add_prefix("pct_").reset_index()
    rows.insert(1, "reason", [chosen[i] for i in rows.image_id])
    low_everywhere = wide.loc[list(chosen)].max(axis=1).to_numpy() < SHARED_LOW_PERCENTILE
    rows.insert(2, "pattern", ["shared" if low else "explainer_specific" for low in low_everywhere])
    rows.loc[rows.reason == "shared_high", "pattern"] = "contrast"
    return rows
