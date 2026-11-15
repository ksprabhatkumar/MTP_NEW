"""Source-only causal interventions on saved correction checkpoints; no training."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from run_local_message_pilot import build_fold, batches, precision_context
from models.summer_message_correction import SummerMessageCorrection
from o2_features import O2Corpus
from workstation_experiments import metric_values, write_json

ROOT = Path(__file__).resolve().parent

def audit(model, part, batch_size=64):
    model.eval()
    rows = {}
    reference = None
    for mode in ('normal', 'zero_message', 'window_shuffle', 'zero_correction'):
        def change(module, args):
            z = args[0].clone()
            h = z.shape[-1] // 2
            if mode == 'zero_message':
                z[..., h:] = 0
            elif mode == 'window_shuffle':
                m = z[..., h:].reshape(-1, 50, 9, h)
                z[..., h:] = m.roll(1, 0).reshape(-1, 9, h)
            return (z,)
        handle = model.correction.register_forward_pre_hook(change)
        out_handle = None
        if mode == 'zero_correction':
            out_handle = model.correction.register_forward_hook(lambda m, a, out: torch.zeros_like(out))
        outputs, labels = [], []
        with torch.inference_mode():
            for batch in batches(part, batch_size):
                with precision_context():
                    logits, _ = model(batch['x'], batch['a'])
                outputs.append(logits.float().cpu())
                labels.append(batch['y'].cpu())
        handle.remove()
        if out_handle: out_handle.remove()
        logits, y = torch.cat(outputs), torch.cat(labels)
        if reference is None: reference = logits
        centered = logits-logits.mean(-1, keepdim=True)
        ref_centered = reference-reference.mean(-1, keepdim=True)
        rows[mode] = {**metric_values(y.numpy(), logits.argmax(-1).numpy()),
            'cross_entropy': F.cross_entropy(logits, y).item(),
            'label_flip_fraction': (logits.argmax(-1)!=reference.argmax(-1)).float().mean().item(),
            'centered_logit_rms_change': (centered-ref_centered).square().mean().sqrt().item()}
    return rows

def main():
    torch.set_num_threads(4)
    corpus = O2Corpus()
    output = ROOT/'results/o1_t01_t08_20261028'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in (2,5):
        fold = build_fold(corpus,subject)
        for arm in ('local_control','message_correction'):
            checkpoint = ROOT/f'results/local_message_pilot_20261027/jobs/{arm}/subject_{subject:02d}/best.pt'
            state = torch.load(checkpoint,map_location='cpu',weights_only=False)
            model = SummerMessageCorrection(arm).cuda()
            model.load_state_dict(state['model'])
            for split in ('val_ideal','val_self'):
                result = {'subject':subject,'arm':arm,'split':split,
                    'interventions':audit(model,fold['data'][split])}
                rows.append(result)
                print(json.dumps(result),flush=True)
            del model
        del fold
    write_json(output/'operator_audit.json', {'source_only':True,'rows':rows,
        'shuffle':'Roll whole windows within deterministic batches; retains each donor temporal sequence. Last singleton batch unchanged.'})
    print('OPERATOR AUDIT COMPLETE',flush=True)

if __name__=='__main__': main()
