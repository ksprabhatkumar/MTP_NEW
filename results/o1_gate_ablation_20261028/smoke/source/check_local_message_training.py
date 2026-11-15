"""GPU preflight and deterministic checkpoint recovery check for the local pilot."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models.summer_dynamic import SummerDynamicWHAR
from models.summer_message_correction import ARMS, SummerMessageCorrection
from run_local_message_pilot import SOURCES, ROOT, precision_context, rng_state, restore_rng
from summer_o1_experiments import scheduler_for
from workstation_experiments import save_checkpoint, seed_all, write_json
from o2_features import features


def step(model, optimizer, scheduler, x, a, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # FP32 recovery check isolates checkpoint restoration from precision drift.
    logits, _ = model(x, a)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    output = Path(parser.parse_args().output)
    output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(7)
    x = torch.randn(8, 50, 9, 5, device='cuda')
    a = torch.ones(8, 9, dtype=torch.bool, device='cuda')
    y = torch.arange(8, device='cuda')
    seed_all(13)
    original = SummerDynamicWHAR(in_features=5, graph_variant='none').cuda().eval()
    reference = original(x, a)[0].detach()
    del original
    errors = {}
    for arm in ARMS:
        seed_all(13)
        model = SummerMessageCorrection(arm).cuda().eval()
        error = float((model(x, a)[0]-reference).abs().max())
        assert error <= 1e-6, (arm, error)
        errors[arm] = error
        del model
    model = SummerMessageCorrection('message_correction').cuda().eval()
    # Test availability with a nonzero correction, not only the zero-init path.
    torch.nn.init.normal_(model.correction[-1].weight, std=.05)
    a[:, 2] = False
    changed = x.clone(); changed[:, :, 2] = 1e6
    first, _, diag = model(x, a, return_diagnostics=True)
    assert torch.allclose(first, model(changed, a)[0], atol=1e-6, rtol=1e-6)
    assert (diag['adjacency'][..., 2] == 0).all()
    assert (diag['correction'].reshape(8, 50, 9, 16)[:, :, 2] == 0).all()
    try:
        model(x, torch.zeros_like(a))
        raise AssertionError('All-missing input accepted')
    except ValueError:
        pass
    del model
    a[:] = True
    seed_all(29)
    model = SummerMessageCorrection('message_correction').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4, fused=True)
    scheduler = scheduler_for(optimizer, 8)
    step(model, optimizer, scheduler, x, a, y)
    step(model, optimizer, scheduler, x, a, y)
    assert model.message_value.weight.grad is not None
    assert float(model.message_value.weight.grad.abs().sum()) > 0
    checkpoint = output/'preflight_resume.pt'
    save_checkpoint(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(), 'epoch': 2, **rng_state()})
    reference_loss = step(model, optimizer, scheduler, x, a, y)
    expected_rng = rng_state()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    restored = SummerMessageCorrection('message_correction').cuda()
    restored_opt = torch.optim.Adam(restored.parameters(), lr=1e-4, weight_decay=5e-4, fused=True)
    restored_sched = scheduler_for(restored_opt, 8)
    restored.load_state_dict(state['model'])
    restored_opt.load_state_dict(state['optimizer'])
    restored_sched.load_state_dict(state['scheduler'])
    restore_rng(state)
    restored_loss = step(restored, restored_opt, restored_sched, x, a, y)
    max_error = max(float((value-restored.state_dict()[key]).abs().max())
                    for key, value in model.state_dict().items())
    assert max_error <= 1e-6 and abs(reference_loss-restored_loss) <= 1e-6, (max_error, reference_loss, restored_loss)
    assert scheduler.state_dict() == restored_sched.state_dict()
    assert torch.equal(expected_rng['cpu_rng'], torch.get_rng_state())
    assert all(torch.equal(left, right) for left, right in zip(expected_rng['cuda_rng'], torch.cuda.get_rng_state_all()))
    del model, optimizer, restored, restored_opt
    # Real source data GPU forward/backward with the actual precision setting.
    cache = ROOT/'dataset/workstation_o1/subject1_ideal.npz'
    with np.load(cache) as data:
        raw = torch.from_numpy(data['x'][:8]).cuda()
        labels = torch.from_numpy(data['y'][:8]).cuda()
        available = torch.from_numpy(data['availability'][:8]).cuda()
    encoded = features(raw, 'mag_dot')
    encoded = (encoded-encoded.mean((0,1),keepdim=True))/encoded.std((0,1),keepdim=True).clamp_min(1e-6)
    smoke = {}
    for arm in ARMS:
        model = SummerMessageCorrection(arm).cuda().train()
        with precision_context():
            logits, _ = model(encoded, available)
            loss = F.cross_entropy(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        smoke[arm] = float(loss.detach())
        del model
    report = {'passed': True, 'cuda': torch.cuda.get_device_name(),
              'initial_no_graph_max_errors': errors, 'missing_sensor_checks': 'passed',
              'checkpoint_resume_max_parameter_error': max_error,
              'checkpoint_resume_loss_error': abs(reference_loss-restored_loss),
              'rng_and_scheduler_recovery': 'passed', 'real_source_gpu_smoke_losses': smoke,
              'source_hashes': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}}
    write_json(output/'preflight.json', report)
    print('PREFLIGHT PASSED: no-graph equivalence, masks, gradients, real-data GPU smoke and checkpoint recovery.', flush=True)


if __name__ == '__main__':
    main()
