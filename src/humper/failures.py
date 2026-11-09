"""Quantitative shortlist of low- and high-alignment images. Humans make the final choice."""

import pandas as pd

from humper import explain

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
