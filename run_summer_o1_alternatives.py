"""Sequential, resumable O1 graph follow-up on locked magnitude+dot inputs."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parent
SOURCES = ["run_summer_o1_alternatives.py", "models/summer_alternatives.py",
           "models/summer_dynamic.py", "models/mamba_whar.py",
           "summer_o2_experiments.py", "summer_o1_experiments.py",
           "o2_features.py", "o2_experiments.py", "workstation_experiments.py",
           "run_training.py", "results/o1_alternatives_pilot_decision_20261021.json"]
ARMS = ("physical", "residual", "none", "masked_balanced", "masked_floor",
        "pairwise_balanced", "pairwise_floor", "gated_masked", "gated_pairwise")
SUBJECTS = (2, 5, 15)
SEEDS = (13, 29, 47)


def model_factory(variant):
    from models.summer_alternatives import MODES, SummerAlternativeGraph
    from summer_o2_experiments import model_for
    if variant.graph in MODES:
        return SummerAlternativeGraph(variant.graph).cuda()
    return model_for(variant)


def routing_fn(model, fold, variant):
    from summer_o1_experiments import Experiment, routing_audit
    graph = "residual" if variant.graph not in ("physical", "residual", "none") else variant.graph
    return routing_audit(model, fold, Experiment(variant.name, graph))


def publish(output, state):
    from workstation_experiments import write_json
    state = dict(state, pid=os.getpid(), updated=datetime.now().astimezone().isoformat())
    write_json(output / "status.json", state)
    (output / "TRAINING_HANDOFF.txt").write_text(
        "O1 ALTERNATIVE GRAPH SESSION\n" + json.dumps(state, indent=2) + "\n\n"
        "This run uses the frozen O2-selected mag_dot input to isolate graph routing. "
        "Resume in the SAME output directory only after the old process has exited. "
        "Inspect status.json, completed.json, jobs/*/subject_*/seed_*/result.json "
        "and run.log. Source-only validation chooses checkpoints; held-out "
        "placement results are exploratory.\n", encoding="utf-8")


def source_snapshot(output):
    from workstation_experiments import write_json
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    cache = hashlib.sha256((ROOT / "dataset/workstation_o1/audit.json").read_bytes()).hexdigest()
    fingerprint = {"sources": hashes, "cache_audit": cache,
                   "arms": list(ARMS), "subjects": SUBJECTS, "seeds": SEEDS,
                   "representation": "mag_dot", "epochs_full": 31}
    path = output / "fingerprint.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(fingerprint)):
        raise ValueError("Sources, cache or schedule changed during resume")
    write_json(path, fingerprint)
    if not (output / "source").exists():
        for name in SOURCES:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        shutil.copy2(ROOT / "O1_ALTERNATIVES_PLAN.md", output / "plan_at_start.md")


def update_summary(output, jobs):
    from statistics import mean
    from workstation_experiments import write_json
    summary = {}
    for arm in ARMS:
        rows = []
        for subject, seed in jobs:
            path = output / "jobs" / f"mag_dot_{arm}" / f"subject_{subject:02}" / f"seed_{seed}" / "result.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
        summary[arm] = {"completed_jobs": len(rows), "expected_jobs": len(jobs),
                        "source_ideal_f1": mean(r["best_validation_f1"] for r in rows) if rows else None}
        if rows and "induced" in rows[0]["metrics"]:
            summary[arm]["induced_f1_observed"] = mean(r["metrics"]["induced"]["macro_f1"] for r in rows)
    write_json(output / "summary.json", summary)


def run(args):
    import msvcrt
    import torch
    from o2_features import O2Corpus
    from summer_o2_experiments import Variant, run_job
    from run_training import Tee, telemetry
    from workstation_experiments import write_json

    output = Path(args.output).resolve()
    active_arms = tuple(args.arms.split(",")) if args.arms else ARMS
    if not active_arms or len(active_arms) != len(set(active_arms)) or any(arm not in ARMS for arm in active_arms):
        raise ValueError("Invalid arm selection")
    if args.mode == "full" and active_arms != ARMS:
        raise ValueError("Full mode requires all nine matched arms")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"0"); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    log = (output / "run.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    stop = threading.Event()
    monitor = None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 GPU required")
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        source_snapshot(output)
        write_json(output / "system.json", {"python": sys.version,
            "executable": sys.executable, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()})
        monitor = threading.Thread(target=telemetry, args=(output, stop), daemon=True)
        monitor.start()
        corpus = O2Corpus()
        if args.mode == "smoke":
            jobs = [(2, 13)]
            epochs, limit, validation_only = 2, 2, True
        elif args.mode == "pilot":
            jobs = [(2, 13), (5, 13)]
            epochs, limit, validation_only = 8, None, True
        else:
            if not args.pilot_dir or not (Path(args.pilot_dir) / "completed.json").exists():
                raise ValueError("Complete the source-only pilot and provide --pilot-dir")
            if not args.gate_pilot_dir or not (Path(args.gate_pilot_dir) / "completed.json").exists():
                raise ValueError("Complete the gated source pilot and provide --gate-pilot-dir")
            decision_file = ROOT / "results" / "o1_alternatives_pilot_decision_20261021.json"
            if not decision_file.is_file() or json.loads(decision_file.read_text()).get("decision") != "proceed":
                raise ValueError(f"O1 full-run scientific gate is not open: {decision_file}")
            jobs = [(subject, seed) for subject in SUBJECTS for seed in SEEDS]
            epochs, limit, validation_only = 31, None, False
        schedule = {"mode": args.mode, "arms": list(active_arms), "jobs": jobs,
                    "epochs": epochs, "batch_size": 64,
                    "selection": "source ideal validation only",
                    "representation": "mag_dot"}
        schedule_path = output / "schedule.json"
        if schedule_path.exists() and json.loads(schedule_path.read_text()) != json.loads(json.dumps(schedule)):
            raise ValueError("Schedule changed during resume")
        write_json(schedule_path, schedule)
        count = 0
        for subject, seed in jobs:
            fold = corpus.fold(subject, "mag_dot")
            fold["sources"] = {key: corpus.sources[indices] for key, indices in fold["indices"].items()}
            for arm in active_arms:
                count += 1
                variant = Variant("mag_dot", arm)
                destination = output / "jobs" / variant.name / f"subject_{subject:02}" / f"seed_{seed}"
                if not (destination / "result.json").exists():
                    publish(output, {"state": "training", "plan_step": "O1-alt",
                                     "current_job": count, "total_jobs": len(jobs) * len(active_arms),
                                     "subject": subject, "seed": seed, "arm": arm})
                    run_job(fold, variant, subject, seed, epochs, 64, destination,
                            limit=limit, validation_only=validation_only,
                            model_factory=model_factory, routing_fn=routing_fn)
                if args.mode == "pilot" and not (destination / "source_routing.json").exists():
                    source_model = model_factory(variant)
                    checkpoint = torch.load(destination / "best.pt", map_location="cpu",
                                            weights_only=False)
                    source_model.load_state_dict(checkpoint["model"])
                    source_fold = {"data": {"ideal": fold["data"]["val_ideal"]}}
                    write_json(destination / "source_routing.json",
                               routing_fn(source_model, source_fold, variant))
                    del source_model, checkpoint
                update_summary(output, jobs)
            del fold
            torch.cuda.empty_cache()
        write_json(output / "completed.json", {"completed_jobs": count,
            "expected_jobs": len(jobs) * len(active_arms),
            "finished": datetime.now().astimezone().isoformat()})
        publish(output, {"state": "complete", "plan_step": "O1-alt",
                         "completed_jobs": count, "total_jobs": count})
    except BaseException as error:
        publish(output, {"state": "failed", "error": repr(error),
                         "traceback": traceback.format_exc()})
        raise
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=12)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "pilot", "full"), required=True)
    parser.add_argument("--output")
    parser.add_argument("--pilot-dir")
    parser.add_argument("--gate-pilot-dir")
    parser.add_argument("--arms", help="Comma-separated subset for smoke/pilot only")
    args = parser.parse_args()
    if args.mode == "plan":
        print(json.dumps({"arms": list(ARMS), "smoke_jobs": 9,
                          "pilot_jobs": 18, "full_jobs": 81,
                          "full_epochs": 31, "full_requires_completed_pilot": True}, indent=2))
    else:
        if not args.output:
            parser.error("--output is required")
        run(args)
