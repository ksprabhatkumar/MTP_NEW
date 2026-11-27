"""Train one full O3 centralized source bridge and compare source validation to O2."""

import argparse
import json
from pathlib import Path

import torch

from o2_features import O2Corpus
from summer_o3_experiments import O3Job, run_job
from workstation_experiments import write_json


def run(o2_run, output):
    o2_run, output = Path(o2_run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for O3 bridge")
    corpus = O2Corpus()
    fold = corpus.fold(2, "mag_dot")
    job = O3Job("mag_dot", "residual", "centralized", 2, 13)
    current = run_job(corpus, fold, job, 31, 64, output / "job")
    old_path = (o2_run / "training" / "jobs" / "mag_dot_residual" /
                "subject_02" / "seed_13" / "best.pt")
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    report = {
        "comparison": "O2 original model versus O3 projection/BatchNorm bridge",
        "subject": 2, "seed": 13, "representation": "mag_dot",
        "graph": "residual", "checkpoint_rule": "source ideal validation only",
        "o2_source_best_f1": old["validation_f1"],
        "o3_source_best_f1": current["best_validation_f1"],
        "o3_minus_o2": current["best_validation_f1"] - old["validation_f1"],
        "o3_best_epoch": current["best_epoch"],
        "o3_seconds": current["seconds_this_invocation"],
        "held_out_used_for_decision": False,
    }
    write_json(output / "BRIDGE_REPORT.json", report)
    (output / "BRIDGE_REPORT.md").write_text(
        "# O3 centralized architecture bridge\n\n"
        "One subject-2/seed-13 run, 31 epochs, source ideal checkpoint selection.\n\n"
        f"O2 source validation F1: {100 * report['o2_source_best_f1']:.2f}%; "
        f"O3 source validation F1: {100 * report['o3_source_best_f1']:.2f}%; "
        f"difference: {100 * report['o3_minus_o2']:+.2f} points.\n\n"
        "The O3 normalization bridge changes the model, so O2 scores are not "
        "a direct federated baseline. Full O3 comparisons must use the same "
        "bridged architecture in every method.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o2-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.o2_run, args.output)
