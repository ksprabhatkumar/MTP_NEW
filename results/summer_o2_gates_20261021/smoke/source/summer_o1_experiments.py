"""Matched summer-backbone O1 training; test scores never select checkpoints."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from models.summer_dynamic import SummerDynamicWHAR
from o2_features import random_rotations, rotate_raw
from o2_experiments import batches, evaluate
from workstation_experiments import seed_all, save_checkpoint, write_json, metric_values, profile, Condition


@dataclass(frozen=True)
class Experiment:
    name: str
    graph: str
    augmentation: str = 'none'
    gae: bool = False


EXPERIMENTS = [Experiment(name, name) for name in
    ('static_sum','physical','global','dynamic','residual','none','residual_balanced')]
EXPERIMENTS += [Experiment(g+'_'+aug, g, aug) for aug in ('rotate','drop') for g in ('physical','residual')]
EXPERIMENTS += [Experiment(g+'_gae', g, 'drop', True) for g in ('physical','residual')]
EXPERIMENTS += [Experiment('dynamicwhar','dynamicwhar')]


def model_for(condition):
    if condition.graph == 'dynamicwhar':
        from baselines.dynamicwhar_adapter import Objective1DynamicWHAR
        return Objective1DynamicWHAR().cuda()
    return SummerDynamicWHAR(graph_variant=condition.graph, use_reconstruction=condition.gae).cuda()


def optimizer_for(model):
    # Original recovered REALDISP code: Adam, lr 1e-4, weight decay 5e-4.
    return torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4, fused=True)


def scheduler_for(optimizer, epochs):
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
        steps_per_epoch=1, epochs=epochs, pct_start=.3, div_factor=1000., final_div_factor=100.)


def train_epoch(model, optimizer, fold, condition, batch_size, limit=None):
    model.train()
    started = time.perf_counter()
    total, count, cls_total, rec_total = 0., 0, 0., 0.
    grad_norms = {}
    mean, std = fold['mean'].cuda(), fold['std'].cuda()
    for batch in batches(fold['data']['train'], batch_size, shuffle=True, limit=limit):
        clean, y, availability = batch['x'], batch['y'], batch['a']
        x, mask = clean, availability.clone()
        if condition.augmentation == 'rotate':
            raw = rotate_raw(batch['raw'], random_rotations(len(clean),9,clean.device))
            x = ((raw-mean)/std) * availability[:,None,:,None]
        elif condition.augmentation == 'drop':
            selected = torch.rand(len(clean), device='cuda') < .2
            node = torch.rand(len(clean),9,device='cuda').masked_fill(~availability,-1).argmax(1)
            rows = torch.arange(len(clean),device='cuda')
            mask[rows[selected],node[selected]] = False
            x = clean * mask[:,None,:,None]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, reconstruction = model(x, mask)
            classification = F.cross_entropy(logits, y)
            reconstruction_loss = torch.zeros((),device='cuda')
            if condition.gae:
                hidden = (availability & ~mask)[:,None,:,None].expand_as(clean)
                reconstruction_loss = ((reconstruction.float()-clean).square()*hidden).sum()/hidden.sum().clamp_min(1)
            loss = classification + .1 * reconstruction_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite summer O1 loss')
        loss.backward()
        for name in ('q.weight','k.weight','branch_logits','affinity_temporal.weight'):
            parameter = dict(model.named_parameters()).get(name)
            if parameter is not None and parameter.grad is not None:
                grad_norms[name] = float(parameter.grad.norm())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        total += float(loss)*len(y)
        cls_total += float(classification)*len(y)
        rec_total += float(reconstruction_loss)*len(y)
        count += len(y)
    torch.cuda.synchronize()
    return {'loss':total/count,'classification_loss':cls_total/count,
        'reconstruction_loss':rec_total/count,'windows':count,
        'last_batch_gradient_norms':grad_norms,'seconds':time.perf_counter()-started}


@torch.inference_mode()
def routing_audit(model, fold, condition):
    if condition.graph == 'dynamicwhar':
        return {'note':'External comparator; summer routing interventions do not apply.'}
    output = {}
    model.eval()
    for scenario in ('ideal','self','induced'):
        if scenario not in fold['data']:
            continue
        part = fold['data'][scenario]
        # Evenly spread windows, not just the first activity.
        ix = torch.linspace(0,len(part['y'])-1,min(512,len(part['y'])),device='cuda').long()
        x, a = part['x'][ix], part['a'][ix]
        logits, _, d = model(x,a,return_diagnostics=True)
        adjacency = d['adjacency'].float()
        report = {'sample_windows':len(ix),'sampling':'evenly spaced across scenario',
            'mean_adjacency':adjacency.mean((0,1)).cpu().tolist(),
            'adjacency_element_std':float(adjacency.flatten(0,1).std(0).mean()),
            'branch_weights':d['branch_weights'].cpu().tolist(),
            'effective_branch_mass':d['effective_branch_mass'].cpu().tolist()}
        if condition.graph in ('residual','dynamic','residual_balanced'):
            for intervention in ('uniform','shuffled'):
                model.intervention = intervention
                modified, _ = model(x,a)
                report[intervention+'_changed_prediction_fraction'] = float((modified.argmax(-1)!=logits.argmax(-1)).float().mean())
            model.intervention = None
        output[scenario] = report
    return output


def run_job(fold, condition, subject, seed, epochs, batch_size, output, limit=None, validation_only=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result_file = output/'result.json'
    if result_file.exists():
        return json.loads(result_file.read_text())
    seed_all(seed)
    model = model_for(condition)
    optimizer = optimizer_for(model)
    # Smoke has two epochs; use constant LR to avoid short OneCycle phase singularities.
    scheduler = scheduler_for(optimizer, epochs) if limit is None else None
    config = {'condition':asdict(condition),'subject':subject,'seed':seed,'epochs':epochs,
        'batch_size':batch_size,'smoke_limit':limit,'validation_only':validation_only,
        'hidden_dim':16,'dropout':.5,'optimizer':'Adam','lr':1e-4,'weight_decay':5e-4,
        'scheduler':'OneCycleLR once/epoch' if scheduler else 'constant for smoke',
        'loss':'cross_entropy + 0.1 masked-target MSE only for GAE ablation',
        'selection':'distinct source subject ideal 33-class macro-F1',
        'provenance':'baselines/summer_original/PROVENANCE.json'}
    write_json(output/'config.json',config)
    first, best, best_epoch, history = 1, -1., 0, []
    latest = output/'latest.pt'
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        if state['config'] != config:
            raise ValueError('Summer checkpoint configuration changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        if scheduler: scheduler.load_state_dict(state['scheduler'])
        first, best, best_epoch, history = state['epoch']+1,state['best'],state['best_epoch'],state['history']
        torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        del state
    started = time.perf_counter()
    with (output/'epochs.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        for epoch in range(first,epochs+1):
            train = train_epoch(model,optimizer,fold,condition,batch_size,limit)
            val,_,_ = evaluate(model,fold,'val_ideal',batch_size,limit=limit)
            row = {'epoch':epoch,'train':train,'validation_ideal_f1':val['macro_f1'],
                'lr':optimizer.param_groups[0]['lr']}
            history.append(row); log.write(json.dumps(row)+'\n')
            if val['macro_f1'] > best:
                best, best_epoch = val['macro_f1'], epoch
                save_checkpoint(output/'best.pt',{'model':model.state_dict(),'config':config,
                    'epoch':epoch,'validation_f1':best,'mean':fold['mean'],'std':fold['std']})
            if scheduler: scheduler.step()
            save_checkpoint(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict() if scheduler else None,'config':config,
                'epoch':epoch,'best':best,'best_epoch':best_epoch,'history':history,
                'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
            print(f'{condition.name} S{subject:02d} seed={seed} epoch={epoch}/{epochs} '
                  f'loss={train["loss"]:.4f} source_val_F1={val["macro_f1"]:.4f}',flush=True)
    model.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=False)['model'])
    result = {'condition':condition.name,'subject':subject,'seed':seed,'best_epoch':best_epoch,
        'best_validation_f1':best,'epochs':epochs,'history':history,'config':config,
        'split':fold['summary'],'metrics':{}}
    scenarios = ('val_ideal','val_self') if validation_only else ('ideal','self','induced')
    for scenario in scenarios:
        if scenario not in fold['data']: continue
        metrics,y,pred = evaluate(model,fold,scenario,batch_size,limit=limit)
        result['metrics'][scenario] = metrics
        np.savez_compressed(output/f'predictions_{scenario}.npz',labels=y,predictions=pred,
            corpus_indices=fold['indices'][scenario][:len(y)])
        if 'sources' in fold:
            sources = fold['sources'][scenario][:len(y)]
            result.setdefault('per_recording',{})[scenario] = {
                source: {'fixed33':metric_values(y[sources==source],pred[sources==source]),
                    'supported':metric_values(y[sources==source],pred[sources==source],np.unique(y[sources==source]).tolist())}
                for source in np.unique(sources)}
    if not validation_only:
        result['routing'] = routing_audit(model,fold,condition)
        result['efficiency'] = profile(model,(fold['data']['ideal']['x'],fold['data']['ideal']['y'],fold['data']['ideal']['a']),Condition(condition.name,condition.graph))
    result['seconds_this_invocation'] = time.perf_counter()-started
    write_json(result_file,result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def summarize(output):
    rows = [json.loads(p.read_text()) for p in (Path(output)/'jobs').glob('*/*/*/result.json')]
    summary = {}
    for c in EXPERIMENTS:
        selected = [r for r in rows if r['condition']==c.name]
        entry = {'completed_jobs':len(selected),'expected_jobs':9}
        for scenario in ('ideal','self','induced'):
            means = [np.mean([r['metrics'][scenario]['macro_f1'] for r in selected if r['subject']==s])
                for s in (2,5,15) if any(r['subject']==s for r in selected)]
            if means:
                entry[scenario] = {'subject_mean':float(np.mean(means)),
                    'subject_std':float(np.std(means,ddof=1)) if len(means)>1 else None}
        summary[c.name]=entry
    write_json(Path(output)/'summary.json',summary)
