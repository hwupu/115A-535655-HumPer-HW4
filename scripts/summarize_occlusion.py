"""Write paired occlusion comparisons and the occlusion results for the final cases."""

import pandas as pd

from humper import data, evaluate, explain, probes

OUT = data.REPO_ROOT / "results"


def main() -> None:
    occlusion = pd.read_csv(OUT / "occlusion.csv")
    rows = []
    for name, spec in explain.EXPLAINERS.items():
        for fill in probes.FILLS:
            part = occlusion[(occlusion.model == spec.model) & (occlusion.fill == fill)]
            wide = part.pivot(index="image_id", columns="mask", values="drop")
            for a, b in ((name, "human"), (name, "random"), ("human", "random")):
                difference = (wide[a] - wide[b]).to_numpy()
                low, high = evaluate.bootstrap_ci(difference)
                rows.append({
                    "explainer": name, "fill": fill, "a": a, "b": b, "n": len(difference),
                    "mean_difference": difference.mean(), "ci_low": low, "ci_high": high,
                    "share_a_larger": float((difference > 0).mean()),
                })
    paired = pd.DataFrame(rows)
    paired.to_csv(OUT / "tables" / "occlusion_paired.csv", index=False)
    print(paired.round(3).to_string())

    notes = pd.read_csv(data.REPO_ROOT / "annotations" / "case_notes.csv")
    rows = []
    for case in notes.itertuples():
        for role, explainer in (("weak", case.weak), ("strong", case.strong)):
            model = explain.EXPLAINERS[explainer].model
            part = occlusion[(occlusion.image_id == case.image_id) & (occlusion.model == model)]
            for fill in probes.FILLS:
                drops = part[part.fill == fill].set_index("mask")["drop"]
                rows.append({
                    "case": case.case, "category": case.category, "image_id": case.image_id, "role": role,
                    "explainer": explainer, "fill": fill, "baseline_prob": part.baseline_prob.iloc[0],
                    "drop_own_map": drops[explainer], "drop_human": drops["human"], "drop_random": drops["random"],
                })
    cases = pd.DataFrame(rows)
    cases.to_csv(OUT / "tables" / "occlusion_cases.csv", index=False)
    print(cases.round(3).to_string())


if __name__ == "__main__":
    main()
