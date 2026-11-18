"""CPU-only checks for O3 aggregation, local normalization and model routing."""

import argparse
import json
from pathlib import Path

import torch

from summer_o3_federation import (FederatedSummerModel, apply_local_state,
    local_normalization_keys, round_communication_bytes,
    weighted_state_average)
from workstation_experiments import write_json


def check():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    model = FederatedSummerModel('mag_dot', 'residual').eval()
    keys = local_normalization_keys(model)
    assert any(key.startswith('input_bn.') for key in keys)
    assert any(key.startswith('backbone.encoder.gin_agg.nn.1.') for key in keys)
    base = model.state_dict()
    first = {key: value.clone() for key, value in base.items()}
    second = {key: value.clone() for key, value in base.items()}
    first['input_projection.weight'].fill_(1)
    second['input_projection.weight'].fill_(3)
    first['input_bn.running_mean'].fill_(10)
    second['input_bn.running_mean'].fill_(20)
    shared = weighted_state_average([first, second], [1, 3], keys, base)
    assert torch.allclose(shared['input_projection.weight'],
                          torch.full_like(shared['input_projection.weight'], 2.5))
    assert torch.equal(shared['input_bn.running_mean'], base['input_bn.running_mean'])
    merged = apply_local_state(shared, first, keys)
    assert torch.equal(merged['input_bn.running_mean'], first['input_bn.running_mean'])
    assert torch.equal(merged['input_projection.weight'], shared['input_projection.weight'])
    expected_bytes = 2 * 2 * sum(value.numel() * value.element_size()
        for key, value in base.items() if key not in keys)
    assert round_communication_bytes(base, 2, keys) == expected_bytes
    with torch.inference_mode():
        x = torch.randn(2, 50, 9, 5)
        availability = torch.ones(2, 9, dtype=torch.bool)
        logits, reconstruction = model(x, availability)
    assert logits.shape == (2, 33) and reconstruction is None
    no_graph = FederatedSummerModel('mag_dot', 'none').eval()
    with torch.inference_mode():
        no_graph_logits, no_graph_reconstruction = no_graph(x, availability)
    assert no_graph_logits.shape == (2, 33) and no_graph_reconstruction is None
    return {'passed': True, 'model': 'summer GIN-attention + projection/BN bridge',
            'local_normalization_keys': sorted(keys),
            'communication_bytes_two_clients': expected_bytes,
            'checks': ['weighted FedAvg', 'local BN retention', 'state merge',
                       'communication accounting', 'CPU model forward',
                       'no-graph control forward']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = check()
    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))
