"""Durable, fail-closed raw then magnitude T12 full launcher."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.t12_representation.common import (  # noqa: E402
    configure, fingerprint, queue_lock, selected, sha, atomic_json,
)

RAW_BASE_REL = Path("results/t12_representation_workstation_r2/raw")
MAG_BASE_REL = Path("results/t12_representation_workstation_r3/magnitude")
SPEC_PATH = ROOT / "docs/experiments/t12_representation_r1.txt"
MATRIX_PATH = ROOT / "docs/experiments/t12_representation_r1_matrix.json"
PAIR_PATH = ROOT / "results/t12_representation_workstation_r3/smoke_pair.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def gate_smoke(rep, base_rel):
    os.environ["MTP_T12_REPRESENTATION"] = rep
    fp = fingerprint()
    base = ROOT / base_rel
    pre = read_json(base / "preflight/preflight.json")
    receipt = read_json(base / "smoke/receipt.json")
    audit = read_json(base / "smoke/audit.json")
    status = read_json(base / "smoke_supervisor.json")
    assert pre["passed"] and pre["cases"] == 20
    assert receipt["passed"] and receipt["stage"] == "smoke" and receipt["jobs_passed"] == 20
    assert audit["passed"] and audit["audited_jobs"] == 20 and audit["paired_contrasts"] == 36
    assert receipt["fingerprint"] == pre["fingerprint"] == fp, f"{rep} smoke/preflight fingerprint mismatch"
    assert set(receipt["job_ids"]) == {j["id"] for j in selected("smoke")}
    assert status["state"] == "complete" and status["stage"] == "smoke"
    return fp


def validate_gates(output_root):
    configure()
    spec = SPEC_PATH.read_text(encoding="utf8")
    assert "New: raw90 + magnitude90 =180 jobs." in spec
    assert "31epochs from scratch; batch64; explicit FP32" in spec
    matrix = read_json(MATRIX_PATH)
    assert matrix["new_jobs"] == 180 and matrix["reused_mag_dot_jobs"] == 90
    assert matrix["epochs_full"] == 31 and matrix["batch_size"] == 64
    pair = read_json(PAIR_PATH)
    assert pair["passed"] and pair["jobs"] == 40 and not pair["full_authorized"]
    assert Path(pair["raw_receipt"]).resolve() == ROOT / RAW_BASE_REL / "smoke/receipt.json"
    assert Path(pair["magnitude_receipt"]).resolve() == ROOT / MAG_BASE_REL / "smoke/receipt.json"
    raw_fp = gate_smoke("raw", RAW_BASE_REL)
    mag_fp = gate_smoke("magnitude", MAG_BASE_REL)
    assert raw_fp["sources"] == mag_fp["sources"] and raw_fp["data"] == mag_fp["data"]
    raw_env = dict(raw_fp["environment"])
    mag_env = dict(mag_fp["environment"])
    assert raw_env.pop("representation") == "raw"
    assert mag_env.pop("representation") == "magnitude"
    assert raw_env == mag_env
    assert len(selected("full")) == 90
    assert all(j["representation"] in ("raw", "magnitude") for j in matrix["jobs"])
    assert sum(j["representation"] == "raw" for j in matrix["jobs"]) == 90
    assert sum(j["representation"] == "magnitude" for j in matrix["jobs"]) == 90
    assert not (ROOT / RAW_BASE_REL / "full").exists(), "Raw full output already exists"
    assert not (ROOT / MAG_BASE_REL / "full").exists(), "Magnitude full output already exists"
    return {"raw_fingerprint": raw_fp, "magnitude_fingerprint": mag_fp,
            "raw_receipt_sha256": sha(ROOT / RAW_BASE_REL / "smoke/receipt.json"),
            "magnitude_receipt_sha256": sha(ROOT / MAG_BASE_REL / "smoke/receipt.json")}


def full_result(rep, base_rel, expected_fp):
    os.environ["MTP_T12_REPRESENTATION"] = rep
    base = ROOT / base_rel
    out = base / "full"
    receipt = read_json(out / "receipt.json")
    audit = read_json(out / "audit.json")
    pre = read_json(base / "preflight/preflight.json")
    smoke = read_json(base / "smoke/receipt.json")
    supervisor = read_json(base / "full_supervisor.json")
    progress = read_json(out / "progress.json")
    assert supervisor["state"] == "complete" and supervisor["stage"] == "full"
    assert receipt["passed"] and receipt["stage"] == "full" and receipt["jobs_passed"] == 90
    assert receipt["fingerprint"] == expected_fp == pre["fingerprint"] == smoke["fingerprint"]
    assert audit["passed"] and audit["expected_jobs"] == 90 and audit["audited_jobs"] == 90
    assert audit["paired_contrasts"] == 162
    assert set(receipt["job_ids"]) == {j["id"] for j in selected("full")}
    assert len(progress["completed"]) == 90 and progress["total"] == 90
    exits = [json.loads(line) for line in (base / "child_exits.jsonl").read_text().splitlines() if line.strip()]
    assert exits and exits[-1]["returncode"] == 0
    assert "experiments.t12_representation.run" in exits[-1]["command"]
    assert read_json(out / "command.json")["smoke_receipt_sha256"] == sha(base / "smoke/receipt.json")
    with (out / "metrics.csv").open(newline="", encoding="utf8") as f:
        assert sum(1 for _ in csv.DictReader(f)) == 180
    with (out / "paired_contrasts.json").open(encoding="utf8") as f:
        assert len(json.load(f)) == 162

    import torch
    jobs = {j["id"]: j for j in selected("full")}
    for job_id, job in jobs.items():
        path = out / "jobs" / job_id
        heartbeat = read_json(path / "heartbeat.json")
        result = read_json(path / "result.json")
        latest = torch.load(path / "latest.pt", map_location="cpu", weights_only=False)
        best = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
        assert heartbeat["epoch"] == latest["epoch"] == 31
        assert len(latest["history"]) == len(result["history"]) == 31
        assert all(k in latest for k in ("model", "teacher", "optimizer", "scheduler", "cpu_rng", "cuda_rng", "streams", "mean", "std", "raw_scale", "history"))
        assert best["epoch"] == result["best_epoch"]
        assert result["best_epoch"] == max(result["history"], key=lambda row: row["val_ideal"]["macro_f1"])["epoch"]
        assert all(k in best for k in ("model", "config", "mean", "std"))
        assert latest["config"] == best["config"] == result["config"]
        norm = torch.load(out / f"normalization_{job['subject']:02d}.pt", map_location="cpu", weights_only=False)
        assert torch.equal(norm["mean"], latest["mean"]) and torch.equal(norm["std"], latest["std"])
        with (path / "epochs.jsonl").open(encoding="utf8") as f:
            assert sum(1 for line in f if line.strip()) == 31

    target = read_json(out / "target_evaluation/audit.json")
    assert target["passed"] and target["jobs"] == 90 and target["primary_rows"] == 270
    assert target["paired_contrasts"] == 243 and not target["test_used_for_selection"]
    with (out / "target_evaluation/metrics.csv").open(newline="", encoding="utf8") as f:
        assert sum(1 for _ in csv.DictReader(f)) == 270
    with (out / "target_evaluation/paired_contrasts.json").open(encoding="utf8") as f:
        assert len(json.load(f)) == 243
    return {"receipt": str(out / "receipt.json"), "audit": str(out / "audit.json"),
            "target_audit": str(out / "target_evaluation/audit.json"), "jobs": 90}


def reserve_output(output_root):
    output_root=Path(output_root).resolve()
    output_root.mkdir(parents=True,exist_ok=False)
    token=uuid.uuid4().hex
    atomic_json(output_root/'reservation.json',dict(token=token))
    return token


def claim_output(output_root,token):
    output_root=Path(output_root).resolve()
    assert read_json(output_root/'reservation.json')['token']==token, 'Invalid launcher reservation'
    assert not (output_root/'status.json').exists(), 'Existing status; do not overwrite'
    # Exclusive creation prevents a second child from consuming this reservation.
    with (output_root/'started.json').open('x',encoding='utf8') as f:
        json.dump(dict(pid=os.getpid()),f)


def run_sequence(output_root,token):
    output_root = Path(output_root).resolve()
    claim_output(output_root,token)
    state = {"state": "running", "pid": os.getpid(), "completed": [],
             "full_started": False, "magnitude_started": False,
             "commands": {}, "logs": {}, "child_exit_codes": {}}
    try:
        with queue_lock(ROOT / "results/.t12_batch.lock"):
            atomic_json(output_root / "status.json", state)
            gates = validate_gates(output_root)
            state["verified_gates"] = True
            state["smoke_pair"] = str(PAIR_PATH)
            state["raw_smoke_receipt_sha256"] = gates["raw_receipt_sha256"]
            state["magnitude_smoke_receipt_sha256"] = gates["magnitude_receipt_sha256"]
            atomic_json(output_root / "full_gate.json", {"passed": True, "user_authorized": True,
                        "new_full_jobs": 180, "reused_t11_mag_dot_jobs": 90,
                        "raw_full_base": str(ROOT / RAW_BASE_REL),
                        "magnitude_full_base": str(ROOT / MAG_BASE_REL)})
            for rep, base_rel, fp in (("raw", RAW_BASE_REL, gates["raw_fingerprint"]),
                                      ("magnitude", MAG_BASE_REL, gates["magnitude_fingerprint"])):
                if rep == "magnitude":
                    assert state.get("completed") == ["raw_full"]
                    state["magnitude_started"] = True
                else:
                    state["full_started"] = True
                base_arg = base_rel.as_posix()
                command = [sys.executable, "-m", "experiments.t12_representation.launch",
                           "--representation", rep, "--stage", "full",
                           "--output-base", base_arg,
                           "--confirm-full", "SMOKE_REVIEWED_BY_USER"]
                log_path = output_root / f"{rep}_launch.log"
                state["commands"][rep] = command
                state["logs"][rep] = str(log_path)
                atomic_json(output_root / "status.json", state)
                with log_path.open("x", encoding="utf8") as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    state['child_pid']=child.pid
                    atomic_json(output_root/'status.json',state)
                    rc=child.wait()
                state['child_exit_codes'][rep+'_supervisor']=rc
                atomic_json(output_root/'status.json',state)
                if rc!=0:raise RuntimeError(f'{rep} supervisor failed with exit {rc}; no next job')
                result=full_result(rep,base_rel,fp)
                state['completed'].append(rep+'_full')
                state.setdefault('results',{})[rep]=result
                atomic_json(output_root/'status.json',state)
            state["state"] = "complete"
            state["full_authorized"] = True
            atomic_json(output_root / "status.json", state)
    except BaseException:
        state["state"] = "failed"
        state["error"] = traceback.format_exc()
        atomic_json(output_root / "status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--child-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if args.check_only:
        assert not output_root.exists(), "Check-only requires a fresh output path"
        gates = validate_gates(output_root)
        print(json.dumps({"passed": True, "new_full_jobs": 180,
                          "reused_t11_mag_dot_jobs": 90,
                          "raw_receipt_sha256": gates["raw_receipt_sha256"],
                          "magnitude_receipt_sha256": gates["magnitude_receipt_sha256"],
                          "raw_full_exists": False, "magnitude_full_exists": False}, indent=2))
        return
    if args.detach:
        token=reserve_output(output_root)
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--output-root", str(output_root), "--child-token", token]
        options = {"cwd": ROOT, "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        log_path = output_root / "sequence_supervisor.log"
        with log_path.open("x", encoding="utf8") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **options)
        atomic_json(output_root / "launch.json", {"pid": child.pid, "command": command,
                    "log": str(log_path), "state": "launched"})
        print(json.dumps({"pid": child.pid, "log": str(log_path),
                          "status": str(output_root / "status.json")}))
        return
    token=args.child_token if args.child_token else reserve_output(output_root)
    run_sequence(output_root,token)


if __name__ == "__main__":
    main()
