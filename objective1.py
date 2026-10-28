"""Reproducible Objective 1 experiment runner for REALDISP."""

import argparse
from datetime import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from dataset.data_loader import get_objective1_loso_loaders
from models.mamba_whar import GAE_Mamba_WHAR


RESULTS_LOG = Path("results_log.txt")


def append_result_log(result, args):
    """Persist only completed folds; interrupted runs never create log entries."""
    ideal, induced, efficiency = result["ideal"], result["induced"], result["efficiency"]
    with RESULTS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] OBJECTIVE 1 COMPLETE | "
            f"Variant: {args.variant} | Held-out subject: {result['subject']} | Seed: {result['seed']} | "
            f"Epochs: {args.epochs} | Sensor dropout: {args.sensor_dropout:.2f} | "
            f"GAE: {args.gae} | Device: {efficiency['device']}\n"
        )
        handle.write(
            f"   --> Best Val Macro-F1: {result['validation_macro_f1'] * 100:.2f}% | "
            f"Ideal: Loss={ideal['loss']:.4f}, Acc={ideal['accuracy'] * 100:.2f}%, Macro-F1={ideal['macro_f1'] * 100:.2f}% | "
            f"Induced: Loss={induced['loss']:.4f}, Acc={induced['accuracy'] * 100:.2f}%, Macro-F1={induced['macro_f1'] * 100:.2f}%\n"
        )
        handle.write(
            f"   --> F1 degradation: {result['induced_degradation_macro_f1'] * 100:.2f} percentage points | "
            f"Batch-one median latency: {efficiency['batch_one_median_ms']:.2f} ms | "
            f"Parameters: {efficiency['parameters']:,}\n"
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sensor_dropout(x, availability, probability):
    if probability <= 0:
        return x, availability
    mask = availability.clone()
    for index in range(mask.size(0)):
        if torch.rand(()) < probability:
            candidates = torch.where(mask[index])[0]
            if len(candidates):
                mask[index, candidates[torch.randint(len(candidates), ())]] = False
    return x * mask[:, None, :, None].to(x.dtype), mask


def evaluate(model, loader, device, criterion=None):
    if loader is None:
        return None
    model.eval()
    labels, predictions, losses = [], [], []
    with torch.no_grad():
        for x, y, availability in loader:
            x, y, availability = x.to(device), y.to(device), availability.to(device)
            logits, _ = model(x, availability)
            if criterion is not None:
                losses.append(criterion(logits, y).item())
            labels.extend(y.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if not labels:
        return None
    return {
        "windows": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "loss": float(np.mean(losses)) if losses else None,
        "labels": labels,
        "predictions": predictions,
    }


def build_model(variant, use_reconstruction, device):
    if variant == "dynamicwhar":
        from baselines.dynamicwhar_adapter import Objective1DynamicWHAR
        return Objective1DynamicWHAR().to(device)
    return GAE_Mamba_WHAR(graph_variant=variant, use_reconstruction=use_reconstruction).to(device)


def profile_batch_one(model, loader, device, repeats=30):
    """Measure the complete model path for one window after a short warm-up."""
    x, _, availability = next(iter(loader))
    x, availability = x[:1].to(device), availability[:1].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x, availability)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x, availability)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_one_median_ms": float(np.median(timings)),
        "batch_one_p95_ms": float(np.percentile(timings, 95)),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "device": str(device),
    }


def train_fold(args, subject, seed, device):
    seed_everything(seed)
    loaders = get_objective1_loso_loaders(subject, args.batch_size)
    model = build_model(args.variant, args.gae, device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None
    print(
        f"\nTRAINING STARTED | subject={subject} | validation_subject={loaders['validation_subject']} | "
        f"seed={seed} | device={device} | epochs={args.epochs} | batch_size={args.batch_size} | "
        f"sensor_dropout={args.sensor_dropout:.2f} | gae={args.gae}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_losses, train_labels, train_predictions = [], [], []
        for x, y, availability in loaders["train"]:
            x, y, availability = x.to(device), y.to(device), availability.to(device)
            clean = x
            x, augmented_availability = sensor_dropout(x, availability, args.sensor_dropout)
            logits, reconstruction = model(x, augmented_availability)
            loss = criterion(logits, y)
            if args.gae and reconstruction is not None:
                masked = (~augmented_availability & availability)[:, None, :, None].expand_as(clean)
                if masked.any():
                    loss = loss + args.reconstruction_weight * torch.nn.functional.mse_loss(reconstruction[masked], clean[masked])
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())
            train_labels.extend(y.detach().cpu().tolist())
            train_predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        train_loss = float(np.mean(train_losses))
        train_accuracy = accuracy_score(train_labels, train_predictions)
        validation = evaluate(model, loaders["validation"], device, criterion)
        elapsed = time.perf_counter() - epoch_started
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"LR={optimiser.param_groups[0]['lr']:.2e} | "
            f"Train Loss={train_loss:.4f} Acc={train_accuracy * 100:.2f}% | "
            f"Val Loss={validation['loss']:.4f} Acc={validation['accuracy'] * 100:.2f}% Macro-F1={validation['macro_f1'] * 100:.2f}% | "
            f"Time={elapsed:.1f}s",
            flush=True,
        )
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    ideal = evaluate(model, loaders["ideal_test"], device, criterion)
    induced = evaluate(model, loaders["induced_test"], device, criterion)
    result = {
        "subject": subject,
        "seed": seed,
        "variant": args.variant,
        "gae": args.gae,
        "sensor_dropout": args.sensor_dropout,
        "validation_macro_f1": best_f1,
        "split": loaders["manifest_summary"],
        "ideal": ideal,
        "induced": induced,
        "efficiency": profile_batch_one(model, loaders["ideal_test"], device),
    }
    if ideal and induced:
        result["induced_degradation_macro_f1"] = ideal["macro_f1"] - induced["macro_f1"]
    print(
        f"FOLD COMPLETE | subject={subject} | seed={seed} | "
        f"Ideal Macro-F1={ideal['macro_f1'] * 100:.2f}% | "
        f"Induced Macro-F1={(induced['macro_f1'] * 100 if induced else float('nan')):.2f}% | "
        f"Latency={result['efficiency']['batch_one_median_ms']:.2f}ms",
        flush=True,
    )
    return result, model


def main():
    parser = argparse.ArgumentParser(description="Objective 1 REALDISP nested-LOSO experiment")
    parser.add_argument("--variant", choices=["none", "physical", "global", "dynamic", "residual", "dynamicwhar"], default="residual")
    parser.add_argument("--subjects", default="all", help="Comma-separated held-out subjects, or all")
    parser.add_argument("--seeds", default="13,29,47")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--sensor-dropout", type=float, default=0.0)
    parser.add_argument("--gae", action="store_true")
    parser.add_argument("--reconstruction-weight", type=float, default=0.01)
    parser.add_argument("--output", default="results/objective1")
    args = parser.parse_args()
    subjects = list(range(1, 18)) if args.subjects == "all" else [int(value) for value in args.subjects.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"OBJECTIVE 1 RUN | device={device} | variant={args.variant} | subjects={subjects} | seeds={seeds}", flush=True)
    results = []
    for subject in subjects:
        for seed in seeds:
            result, _ = train_fold(args, subject, seed, device)
            results.append(result)
            append_result_log(result, args)
            print(json.dumps({key: value for key, value in result.items() if key not in {"ideal", "induced"}}, sort_keys=True))
    with open(output / f"{args.variant}_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
