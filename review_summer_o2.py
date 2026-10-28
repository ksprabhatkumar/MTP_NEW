"""Audit completed summer O2 artifacts and write a compact scientific review."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


EXPECTED = (
    "magnitude_physical", "magnitude_residual", "magnitude_none",
    "mag_dot_physical", "mag_dot_residual", "mag_dot_none",
    "raw_aug_none",
)
SUBJECTS = (2, 5, 15)
SEEDS = (13, 29, 47)


def percent(value):
    return f"{100 * value:.2f}%"


def points(value):
    return f"{100 * value:+.2f} points"


def run(root):
    root = Path(root).resolve()
    if json.loads((root / "status.json").read_text())["state"] != "complete":
        raise ValueError("Outer O2 session is incomplete")
    training = root / "training"
    if json.loads((training / "completed.json").read_text())["completed_jobs"] != 63:
        raise ValueError("Training completion marker is not 63")
    rows = {}
    missing = []
    for condition in EXPECTED:
        for subject in SUBJECTS:
            for seed in SEEDS:
                job = training / "jobs" / condition / f"subject_{subject:02}" / f"seed_{seed}"
                needed = ["result.json", "best.pt", "latest.pt", "epochs.jsonl",
                          "predictions_ideal.npz", "predictions_self.npz",
                          "predictions_induced.npz", "predictions_rotated_0.npz",
                          "predictions_rotated_1.npz", "predictions_rotated_2.npz"]
                missing.extend(str(job / name) for name in needed if not (job / name).is_file())
                if not (job / "result.json").is_file():
                    continue
                row = json.loads((job / "result.json").read_text())
                if (row["condition"], row["subject"], row["seed"]) != (condition, subject, seed):
                    raise ValueError(f"Result identity mismatch: {job}")
                if row["epochs"] != 31 or len(row["history"]) != 31:
                    raise ValueError(f"Incomplete epoch history: {job}")
                rows[(condition, subject, seed)] = row
    if missing:
        raise ValueError(f"Missing {len(missing)} artifacts; first: {missing[0]}")
    source = {condition: mean(rows[(condition, s, seed)]["best_validation_f1"]
              for s in SUBJECTS for seed in SEEDS) for condition in EXPECTED}
    selection = max(("magnitude", "mag_dot"),
                    key=lambda kind: (source[f"{kind}_residual"], kind == "mag_dot"))
    scenarios = {}
    for condition in EXPECTED:
        scenarios[condition] = {}
        for scenario in ("ideal", "self", "induced"):
            by_subject = {str(s): mean(rows[(condition, s, seed)]["metrics"][scenario]["macro_f1"]
                                       for seed in SEEDS) for s in SUBJECTS}
            scenarios[condition][scenario] = {"mean": mean(by_subject.values()),
                                               "by_subject": by_subject}
    pairs = {}
    for kind in ("magnitude", "mag_dot"):
        for reference in ("physical", "none"):
            target = f"{kind}_residual"
            base = f"{kind}_{reference}"
            by_subject = {str(s): mean(rows[(target, s, seed)]["metrics"]["induced"]["macro_f1"]
                                       - rows[(base, s, seed)]["metrics"]["induced"]["macro_f1"]
                                       for seed in SEEDS) for s in SUBJECTS}
            all_pairs = [rows[(target, s, seed)]["metrics"]["induced"]["macro_f1"]
                         - rows[(base, s, seed)]["metrics"]["induced"]["macro_f1"]
                         for s in SUBJECTS for seed in SEEDS]
            pairs[f"{target}_minus_{base}"] = {"mean": mean(by_subject.values()),
                "positive_pairs": sum(x > 0 for x in all_pairs),
                "by_subject": by_subject}
    rotation = {}
    routing = {}
    latency = {}
    per_recording = defaultdict(lambda: defaultdict(list))
    for condition in EXPECTED:
        values = [rows[(condition, s, seed)] for s in SUBJECTS for seed in SEEDS]
        rotation[condition] = mean(mean(v["macro_f1"] for v in r["rotated_ideal"])
                                   for r in values)
        latency[condition] = mean(r["efficiency"]["batch_one_median_ms"] for r in values)
        if condition.endswith("_residual"):
            routing[condition] = {
                "mean_dynamic_mass_induced": mean(r["routing"]["induced"]["effective_branch_mass"][2]
                                                  for r in values),
                "mean_adjacency_std_induced": mean(r["routing"]["induced"]["adjacency_element_std"]
                                                   for r in values),
                "mean_uniform_prediction_flip": mean(r["routing"]["induced"]["uniform_changed_prediction_fraction"]
                                                     for r in values),
                "mean_shuffled_prediction_flip": mean(r["routing"]["induced"]["shuffled_changed_prediction_fraction"]
                                                      for r in values),
            }
        for row in values:
            for name, detail in row["per_recording"]["induced"].items():
                per_recording[condition][name].append(detail["fixed33"]["macro_f1"])
    recording_summary = {condition: {name: mean(scores) for name, scores in sorted(recordings.items())}
                         for condition, recordings in per_recording.items()}
    report = {"artifact_check": "63/63 complete with checkpoints, histories and predictions",
              "source_validation_f1": source, "selected_invariant": selection,
              "selection_basis": "mean best source ideal validation macro-F1 over nine residual runs",
              "scenarios": scenarios, "paired_induced": pairs,
              "rotated_ideal_f1": rotation, "batch_one_median_ms": latency,
              "routing": routing, "induced_by_recording": recording_summary,
              "confirmatory": False}
    (training / "O2_REVIEW.json").write_text(json.dumps(report, indent=2) + "\n")
    text = ["# O2 result audit", "", "All 63 new jobs have 31-epoch histories, both checkpoints, and ideal/self/induced and three rotated prediction files.",
            "The outer supervisor and final analysis completed successfully.", "",
            "## Source-only O3 representation choice", "",
            f"Magnitude residual: {percent(source['magnitude_residual'])}; magnitude+dot residual: {percent(source['mag_dot_residual'])}.",
            f"The predeclared source-validation rule selects **{selection}**. Held-out placement scores were not used to choose it.", "",
            "## Recorded placement", "",
            "| Condition | Ideal | Self | Induced | Synthetic rotated ideal | Batch-one median |",
            "|---|---:|---:|---:|---:|---:|"]
    for condition in EXPECTED:
        text.append("| " + condition + " | " + " | ".join(
            percent(scenarios[condition][scenario]["mean"]) for scenario in ("ideal", "self", "induced"))
            + f" | {percent(rotation[condition])} | {latency[condition]:.2f} ms |")
    text += ["", "Paired induced differences, averaging seeds within subject and subjects equally:"]
    for name, value in pairs.items():
        text.append(f"- {name}: {points(value['mean'])}; {value['positive_pairs']}/9 positive pairs; "
                    + ", ".join(f"S{s} {points(v)}" for s, v in value["by_subject"].items()) + ".")
    text += ["", "## Dynamic routing evidence", ""]
    for condition, value in routing.items():
        text.append(f"- {condition}: induced dynamic branch mass {percent(value['mean_dynamic_mass_induced'])}; "
                    f"adjacency element SD {value['mean_adjacency_std_induced']:.3g}; "
                    f"uniform/shuffled intervention prediction flip rates "
                    f"{percent(value['mean_uniform_prediction_flip'])}/{percent(value['mean_shuffled_prediction_flip'])}.")
    text += ["", "The invariant representation yields the large placement improvement. Residual routing has a small advantage over the physical graph but trails no graph for both feature families. The near-zero affinity variation and prediction interventions do not support a useful sample-conditioned routing claim. The previously inspected placement subjects make this exploratory, not confirmatory evidence.",
             "", "Per-recording fixed-33-class results and subject breakdowns are in O2_REVIEW.json. A future placement claim needs untouched recorded data. O3 may test federation of the selected representation with residual, physical, and no-graph controls; it must not call the residual graph superior on these O2 results."]
    (training / "O2_REVIEW.md").write_text("\n".join(text) + "\n")
    print(json.dumps({"jobs": len(rows), "selected_invariant": selection,
                      "source_validation_f1": {k: source[k] for k in ("magnitude_residual", "mag_dot_residual")},
                      "paired_induced": pairs}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    run(parser.parse_args().run)
