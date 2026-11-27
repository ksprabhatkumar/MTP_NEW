"""Compact plotting bundle from saved histories; no training."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import atomic_json

def report(out,stage):
    jobs=[json.loads(p.read_text()) for p in sorted((out/'jobs').glob('*/result.json'))]
    times=[sum(h['train']['seconds'] for h in r['history']) for r in jobs]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for r in jobs:
        c=r['config'];x=[h['epoch'] for h in r['history']]
        for col,split in enumerate(('val_ideal','val_self')):
            ax=axes[int(c['temporal']=='ssm'),col]
            ax.plot(x,[100*h[split]['macro_f1'] for h in r['history']],label=f"{c['arm']} s{c['subject']} r{c['seed']}",alpha=.65)
            ax.set(xlabel='Epoch',ylabel='Fixed33 macro-F1 (%)',title=f"{c['temporal']} {split}")
    axes[0,0].legend(fontsize=5,ncol=2);fig.tight_layout()
    fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
    estimate=(sum(times)/len(times))*31/4*90 if stage=='smoke' else None
    atomic_json(out/'timing.json',dict(jobs=len(jobs),cumulative_epoch_seconds=sum(times),projected_90_job_epoch_seconds=estimate,
        estimate_note='Smoke has only two robust epochs and a different schedule; projection is provisional, excludes final diagnostics; allow 25-40% reserve'))
