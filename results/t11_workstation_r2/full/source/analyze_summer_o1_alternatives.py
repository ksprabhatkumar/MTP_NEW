"""Summarize source pilots or full matched O1 graph follow-up runs."""

import argparse
import json
from pathlib import Path
from statistics import mean

from run_summer_o1_alternatives import ARMS, SUBJECTS, SEEDS


def analyze(root, mode, arms=ARMS):
    root = Path(root).resolve()
    subjects = (2, 5) if mode == "pilot" else SUBJECTS
    seeds = (13,) if mode == "pilot" else SEEDS
    rows = {}
    for arm in arms:
        for subject in subjects:
            for seed in seeds:
                job = root / "jobs" / f"mag_dot_{arm}" / f"subject_{subject:02}" / f"seed_{seed}"
                path = job / "result.json"
                if not path.is_file():
                    raise ValueError(f"Missing {path}")
                rows[(arm, subject, seed)] = json.loads(path.read_text())
                if mode == "pilot" and not (job / "source_routing.json").is_file():
                    raise ValueError(f"Missing source routing audit for {job}")
    summary = {}
    for arm in arms:
        selected = [rows[(arm, subject, seed)] for subject in subjects for seed in seeds]
        summary[arm] = {"source_ideal_f1": mean(r["best_validation_f1"] for r in selected),
                        "source_self_f1": mean(r["metrics"]["val_self"]["macro_f1"] for r in selected)
                        if mode == "pilot" else None}
        if mode == "pilot":
            routing = [json.loads((root / "jobs" / f"mag_dot_{arm}" /
                       f"subject_{subject:02}" / f"seed_{seed}" / "source_routing.json").read_text())["ideal"]
                       for subject in subjects for seed in seeds]
            summary[arm]["source_adjacency_std"] = mean(r["adjacency_element_std"] for r in routing)
            if arm not in ("physical", "none"):
                summary[arm]["source_uniform_flip"] = mean(
                    r["uniform_changed_prediction_fraction"] for r in routing)
                summary[arm]["source_shuffled_flip"] = mean(
                    r["shuffled_changed_prediction_fraction"] for r in routing)
        else:
            for scenario in ("ideal", "self", "induced"):
                summary[arm][scenario] = mean(mean(rows[(arm, subject, seed)]["metrics"][scenario]["macro_f1"]
                                                    for seed in seeds) for subject in subjects)
            summary[arm]["latency_ms"] = mean(r["efficiency"]["batch_one_median_ms"] for r in selected)
            summary[arm]["dynamic_branch_mass"] = mean(
                r["routing"]["induced"]["effective_branch_mass"][2] for r in selected)
    comparisons = {}
    if mode == "full":
        for arm in ARMS[3:]:
            for baseline in ("physical", "residual", "none"):
                values = [rows[(arm, subject, seed)]["metrics"]["induced"]["macro_f1"]
                          - rows[(baseline, subject, seed)]["metrics"]["induced"]["macro_f1"]
                          for subject in subjects for seed in seeds]
                comparisons[f"{arm}_minus_{baseline}"] = {
                    "mean_difference": mean(values),
                    "positive_pairs": sum(value > 0 for value in values), "pairs": len(values)}
    report = {"mode": mode, "jobs": len(rows), "summary": summary,
              "paired_induced": comparisons, "confirmatory": False}
    (root / "ANALYSIS.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"# O1 alternative graph {mode} analysis", "",
             "The summer backbone and magnitude+dot input are fixed across all arms. "
             "Checkpoints use source ideal validation only. Recorded displacement subjects "
             "were previously inspected, so full placement results are exploratory.", ""]
    if mode == "pilot":
        lines += ["| Arm | Source ideal | Source self | Adjacency SD | Uniform flip | Shuffled flip |",
                  "|---|---:|---:|---:|---:|---:|"]
        for arm in arms:
            item = summary[arm]
            lines.append(f"| {arm} | {100*item['source_ideal_f1']:.2f}% | "
                         f"{100*item['source_self_f1']:.2f}% | {item['source_adjacency_std']:.3g} | "
                         f"{100*item.get('source_uniform_flip', 0):.2f}% | "
                         f"{100*item.get('source_shuffled_flip', 0):.2f}% |")
        lines += ["", "A pilot establishes learning and routing activity, not placement superiority. "
                  "Review these source-only results before starting full training."]
    else:
        lines += ["| Arm | Ideal | Self | Induced | Latency | Dynamic mass |",
                  "|---|---:|---:|---:|---:|---:|"]
        for arm in arms:
            item = summary[arm]
            lines.append(f"| {arm} | {100*item['ideal']:.2f}% | {100*item['self']:.2f}% | "
                         f"{100*item['induced']:.2f}% | {item['latency_ms']:.2f} ms | "
                         f"{100*item['dynamic_branch_mass']:.2f}% |")
        lines += ["", "## Paired induced differences", ""]
        for name, item in comparisons.items():
            lines.append(f"- {name}: {100*item['mean_difference']:+.2f} points; "
                         f"{item['positive_pairs']}/{item['pairs']} positive pairs.")
        lines += ["", "Use the saved job-level routing interventions and per-recording "
                  "metrics before making any graph claim. A larger adjacency variation by "
                  "itself does not establish useful sample-conditioned routing."]
    (root / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"mode": mode, "jobs": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--arms", help="Comma-separated pilot subset")
    args = parser.parse_args()
    arms = tuple(args.arms.split(",")) if args.arms else ARMS
    if any(arm not in ARMS for arm in arms):
        parser.error("Unknown arm")
    analyze(args.run, args.mode, arms)
