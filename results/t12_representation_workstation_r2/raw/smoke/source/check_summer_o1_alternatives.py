"""CPU graph correctness checks for the O1 routing follow-up."""

import argparse
import json
from pathlib import Path

import torch

from models.summer_alternatives import MODES, SummerAlternativeGraph
from workstation_experiments import write_json


def check():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    x = torch.randn(2, 8, 9, 5)
    availability = torch.ones(2, 9, dtype=torch.bool)
    availability[1, 2] = False
    report = {}
    for mode in sorted(MODES):
        model = SummerAlternativeGraph(mode)
        model.eval()
        adjacency, detail = model.graph(x, availability)
        assert adjacency.shape == (16, 9, 9)
        assert torch.isfinite(adjacency).all()
        assert torch.allclose(adjacency.diagonal(dim1=-2, dim2=-1),
                              torch.zeros(16, 9), atol=1e-7)
        reshaped = adjacency.reshape(2, 8, 9, 9)
        assert reshaped[1, :, :, 2].abs().max() == 0
        assert reshaped[1, :, 2, :].abs().max() == 0
        if mode.startswith("gated"):
            assert (reshaped[0].sum(-1) > 0).all()
            assert (reshaped[0].sum(-1) < 1).all()
        else:
            assert torch.allclose(reshaped[0].sum(-1), torch.ones(8, 9), atol=1e-6)
        assert torch.allclose(reshaped[1, :, 2, :].sum(-1), torch.zeros(8), atol=1e-7)
        singleton = torch.zeros(1, 9, dtype=torch.bool)
        singleton[0, 4] = True
        single_adj, _ = model.graph(x[:1], singleton)
        assert single_adj.abs().max() == 0
        logits, _ = model(x, availability)
        assert logits.shape == (2, 33) and torch.isfinite(logits).all()
        model.train()
        logits, _ = model(x, availability)
        logits.square().mean().backward()
        trainable = (model.pairwise[0].weight if "pairwise" in mode
                     else model.q.weight)
        assert trainable.grad is not None and torch.isfinite(trainable.grad).all()
        assert trainable.grad.norm() > 0
        report[mode] = {"passed": True,
                        "initial_dynamic_mass": float(detail["effective_branch_mass"][2]),
                        "router_grad_norm": float(trainable.grad.norm()),
                        "available_row_sum_mean": float(reshaped[0].sum(-1).mean())}
    return {"passed": True, "modes": report,
            "checks": ["eligible sender masking", "self edge masking", "empty row",
                       "normalized nonempty rows", "finite logits", "router gradients"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = check()
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
