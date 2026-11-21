"""Create explanatory, reproducible figures from completed O1/O2 JSON results.

This script reads result.json and summary.json files only. It does not load
checkpoints, run evaluation, or combine the two different model pipelines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


O1 = Path("results/summer_o1_session_20261017_232945/training")
O2 = Path("results/o2_full_20261017_075315")
COLORS = {
    "static_sum": "#868e96",
    "physical": "#5271a5",
    "residual": "#dc6839",
    "none": "#36a17d",
    "physical_rotate": "#7595c7",
    "residual_rotate": "#ec9a72",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=190, bbox_inches="tight")
    plt.close(fig)


def grouped_jobs(root: Path, expected: dict[str, int]):
    rows = [read_json(p) for p in (root / "jobs").glob("*/*/*/result.json")]
    rows = [row for row in rows if row["subject"] in (2, 5, 15)]
    groups = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    for name, count in expected.items():
        if len(groups.get(name, [])) != count:
            raise ValueError(f"{root}: {name} has {len(groups.get(name, []))}/{count} jobs")
        pairs = {(r["subject"], r["seed"]) for r in groups[name]}
        if len(pairs) != count:
            raise ValueError(f"{root}: duplicate subject/seed in {name}")
    return groups


def subject_means(rows, metric):
    subjects = sorted({r["subject"] for r in rows})
    return {s: mean(metric(r) for r in rows if r["subject"] == s) for s in subjects}


def decorate(ax, title, ylabel="33-class macro-F1 (%)"):
    ax.set_title(title, loc="left", weight="bold", fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def plot_o1(root: Path):
    if read_json(root / "completed.json")["completed_jobs"] != 126:
        raise ValueError("O1 is not complete")
    names = ["static_sum", "physical", "residual", "none", "physical_rotate", "residual_rotate"]
    groups = grouped_jobs(root, {name: 9 for name in names})
    out = root / "figures"
    labels = ["Original fixed", "Normalized fixed", "Residual", "No graph", "Fixed + rotation", "Residual + rotation"]

    # Raw O1 placement trade-off. A single axis preserves the large clean/shift gap.
    fig, ax = plt.subplots(figsize=(8.8, 6.6), layout="constrained")
    for name, label in zip(names, labels):
        ideal = mean(subject_means(groups[name], lambda r: r["metrics"]["ideal"]["macro_f1"]).values()) * 100
        induced = mean(subject_means(groups[name], lambda r: r["metrics"]["induced"]["macro_f1"]).values()) * 100
        ax.scatter(ideal, induced, s=115, color=COLORS[name], zorder=3)
        dx, dy = {"static_sum": (-5, -7), "physical": (-9, 4), "residual": (5, -2),
                  "none": (5, 2), "physical_rotate": (-10, -3), "residual_rotate": (4, 2)}[name]
        ax.annotate(label, (ideal, induced), xytext=(dx, dy), textcoords="offset points", fontsize=9,
                    ha="right" if dx < 0 else "left", va="center")
    ax.set(xlabel="Ideal placement: macro-F1 (%)", ylabel="Recorded induced placement: macro-F1 (%)",
           xlim=(57, 100), ylim=(0, 76))
    decorate(ax, "O1 | Ideal performance versus recorded displacement", "Recorded induced macro-F1 (%)")
    fig.text(.02, -.01, "Summer GIN-attention pipeline · 3 held-out subjects × 3 seeds · subject-mean scores",
             fontsize=9, color="#555")
    save(fig, out / "explain_o1_placement_tradeoff")

    # Paired differences use the same subject and seed, so baseline drift is not hidden.
    fig, ax = plt.subplots(figsize=(8.8, 5.5), layout="constrained")
    refs = [("physical", "Residual − normalized fixed", "#5271a5"),
            ("none", "Residual − no graph", "#36a17d")]
    x = np.arange(3)
    for k, (reference, label, color) in enumerate(refs):
        left = {(r["subject"], r["seed"]): r for r in groups["residual"]}
        right = {(r["subject"], r["seed"]): r for r in groups[reference]}
        values = [100 * mean(left[s, seed]["metrics"]["induced"]["macro_f1"]
                             - right[s, seed]["metrics"]["induced"]["macro_f1"]
                             for seed in (13, 29, 47)) for s in (2, 5, 15)]
        bars = ax.bar(x + (k - .5) * .34, values, width=.32, color=color, label=label)
        ax.bar_label(bars, fmt="%+.2f", padding=3, fontsize=9)
    ax.axhline(0, color="#333", linewidth=.9)
    ax.set(xticks=x, xticklabels=["Subject 2", "Subject 5", "Subject 15"],
           ylim=(-4.6, 3.5))
    decorate(ax, "O1 | Paired induced-placement differences", "Residual advantage (F1 points)")
    ax.legend(frameon=False, loc="lower left")
    fig.text(.02, -.01, "Each bar averages 3 matched seed differences; positive favors residual.", fontsize=9, color="#555")
    save(fig, out / "explain_o1_paired_induced")

    # Show effective branch contribution and the intervention evidence together.
    residual = groups["residual"]
    masses = np.array([r["routing"]["induced"]["effective_branch_mass"] for r in residual], dtype=float)
    changes = np.array([[r["routing"]["induced"][f"{kind}_changed_prediction_fraction"]
                         for kind in ("uniform", "shuffled")] for r in residual], dtype=float)
    variation = np.array([r["routing"]["induced"]["adjacency_element_std"] for r in residual], dtype=float)
    fig, (ax, note) = plt.subplots(1, 2, figsize=(10.2, 4.8), gridspec_kw={"width_ratios": [1.4, 1]}, layout="constrained")
    mass = masses.mean(0) * 100
    ax.bar(["Physical", "Global", "Input-dependent"], mass, color=["#5271a5", "#9297aa", "#dc6839"])
    for i, value in enumerate(mass):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 75)
    decorate(ax, "Effective routing mass", "Mean contribution (%)")
    note.axis("off")
    note.text(0, .88, "Dynamic-branch intervention", weight="bold", fontsize=12)
    note.text(0, .68, f"Uniform affinity: {np.count_nonzero(changes[:, 0])}/9 runs changed sampled predictions", fontsize=10)
    note.text(0, .53, f"Shuffled affinity: {np.count_nonzero(changes[:, 1])}/9 runs changed sampled predictions", fontsize=10)
    note.text(0, .34, f"Largest measured adjacency variation: {variation.max():.2g}", fontsize=10)
    note.text(0, .13, "This diagnostic does not establish useful\ninput-dependent routing.", fontsize=10, color="#9c4e2e")
    fig.suptitle("O1 | What did the trained residual graph use?", x=.04, ha="left", weight="bold")
    save(fig, out / "explain_o1_routing_diagnostic")
    return ["explain_o1_placement_tradeoff", "explain_o1_paired_induced", "explain_o1_routing_diagnostic"]


def plot_o2(root: Path):
    if read_json(root / "status.json")["completed_jobs"] != 128:
        raise ValueError("Historical O2 is not complete")
    names = ["raw", "raw_aug", "magnitude", "mag_dot"]
    conditions = [f"{name}_{graph}" for name in names for graph in ("none", "residual")]
    groups = grouped_jobs(root, {name: 9 for name in conditions})
    summary = read_json(root / "summary.json")
    out = root / "figures"
    labels = ["Raw axes", "Raw + rotation", "Magnitudes", "Magnitudes + dot"]
    x = np.arange(4)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True, layout="constrained")
    for ax, scenario, title in zip(axes, ("ideal", "induced"), ("Ideal placement", "Recorded induced placement")):
        for offset, graph, color in ((-.18, "none", "#36a17d"), (.18, "residual", "#dc6839")):
            values = [100 * summary[f"{name}_{graph}"][f"{scenario}_macro_f1_33"]["subject_mean"] for name in names]
            bars = ax.bar(x + offset, values, .34, label="No graph" if graph == "none" else "Residual graph", color=color)
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 105))
        ax.tick_params(axis="x", rotation=25)
        decorate(ax, title)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.035), ncol=2)
    fig.suptitle("Earlier O2 pipeline | Orientation handling versus graph choice", x=.03, ha="left", weight="bold")
    save(fig, out / "explain_o2_feature_graph")

    fig, ax = plt.subplots(figsize=(8.8, 5.5), layout="constrained")
    selected = [("raw_aug_none", "Raw + rotation, no graph", "#6b83b7"),
                ("mag_dot_none", "Magnitude + dot, no graph", "#36a17d"),
                ("mag_dot_residual", "Magnitude + dot, residual", "#dc6839")]
    for k, (name, label, color) in enumerate(selected):
        subjects = subject_means(groups[name], lambda r: r["metrics"]["induced"]["macro_f1"])
        bars = ax.bar(np.arange(3) + (k - 1) * .25, [100 * subjects[s] for s in (2, 5, 15)], .24,
                      color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set(xticks=np.arange(3), xticklabels=["Subject 2", "Subject 5", "Subject 15"], ylim=(0, 108))
    decorate(ax, "Earlier O2 pipeline | Induced-placement variation by subject")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3)
    save(fig, out / "explain_o2_subjects")

    fig, ax = plt.subplots(figsize=(8.8, 5.2), layout="constrained")
    for offset, metric, label, color in ((-.18, "ideal", "Original ideal", "#5271a5"),
                                         (.18, "rotated_ideal", "Synthetic rotated ideal", "#dc6839")):
        key = "ideal_macro_f1_33" if metric == "ideal" else "rotated_ideal_macro_f1_33"
        values = [100 * summary[f"{name}_none"][key]["subject_mean"] for name in names]
        bars = ax.bar(x + offset, values, .34, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 108))
    ax.tick_params(axis="x", rotation=20)
    decorate(ax, "Earlier O2 pipeline | Independent IMU rotations (no-graph control)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2)
    fig.text(.02, -.13, "Synthetic rotation tests orientation stability; it is distinct from recorded physical displacement.", fontsize=9, color="#555")
    save(fig, out / "explain_o2_rotation")
    return ["explain_o2_feature_graph", "explain_o2_subjects", "explain_o2_rotation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o1", type=Path, default=O1)
    parser.add_argument("--o2", type=Path, default=O2)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "#24292e", "text.color": "#24292e"})
    o1 = plot_o1(args.o1)
    o2 = plot_o2(args.o2)
    for root, names in ((args.o1, o1), (args.o2, o2)):
        for name in names:
            print(root / "figures" / (name + ".png"))


if __name__ == "__main__":
    main()
