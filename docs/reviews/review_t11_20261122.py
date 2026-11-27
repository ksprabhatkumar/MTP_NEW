"""Read saved T11 artifacts; write a separate scientific review, never results."""
import collections,json,statistics,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from workstation_experiments import metric_values
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=ROOT/'results/t11_workstation_r2/full'
OUT=ROOT/'docs/reviews/t11_complete_20261122';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
mean=statistics.mean
receipt=read(B/'receipt.json');fp=read(B/'fingerprint.json')
assert receipt['passed'] and receipt['jobs_passed']==90
assert receipt['fingerprint']==fp==read(B.parent/'smoke/fingerprint.json')==read(B.parent/'preflight/preflight.json')['fingerprint']
assert read(B/'status.json')['state']==read(B.parent/'full_supervisor.json')['state']=='complete'
assert read(B/'audit.json')['passed'] and read(B/'target_evaluation/audit.json')['passed']
assert json.loads((B.parent/'child_exits.jsonl').read_text().splitlines()[-1])['returncode']==0
expected={j['id'] for j in read(ROOT/'docs/experiments/t11_core_confirmation_r1_matrix.json')['jobs']}
assert set(receipt['job_ids'])==expected
files=list((B/'jobs').glob('*/result.json'));assert {p.parent.name for p in files}==expected
rows=[];durations=collections.defaultdict(list);orders={};stresses={};checked=0;support={};paired_metadata={};best_epochs=[]
def verify(path,metrics,key):
    global checked
    with np.load(path) as d:
        y=d['labels'];logits=d['logits'];pred=d['predictions']
        assert logits.shape==(len(y),33) and np.isfinite(logits).all() and np.array_equal(pred,logits.argmax(-1))
        m=metric_values(y,pred)
        for k in ('accuracy','macro_f1','weighted_f1'):assert abs(m[k]-metrics[k])<1e-10
        metadata={k:d[k].copy() for k in ('labels','corpus_indices','subjects','recordings','scenarios','window_starts')}
        if key in paired_metadata:
            for k in metadata:assert np.array_equal(paired_metadata[key][k],metadata[k]),(path,k)
        else:paired_metadata[key]=metadata
        checked+=1
        return pred.copy(),logits.copy()
for p in files:
    v=read(p);c=v['config'];hist=v['history'];assert len(hist)==31 and c['epochs']==31
    assert v['best_epoch']==max(hist,key=lambda x:x['val_ideal']['macro_f1'])['epoch']
    best_epochs.append(v['best_epoch']);key=(c['subject'],c['seed'])
    oh=[x['train']['order_sha256'] for x in hist];assert key not in orders or orders[key]==oh;orders[key]=oh
    if c['arm']!='none_clean':
        sh=[x['train']['stress_sha256'] for x in hist];assert key not in stresses or stresses[key]==sh;stresses[key]=sh
    durations[c['temporal'],c['arm']].append(sum(x['train']['seconds'] for x in hist))
    for split in ('val_ideal','val_self'):verify(p.parent/f'predictions_{split}.npz',v['metrics'][split],('source',c['subject'],split))
rotation_disagree=0;rotation_logit_max=0.;drop=collections.defaultdict(list)
for folder in sorted((B/'target_evaluation').iterdir()):
    if not folder.is_dir():continue
    summary=read(folder/'summary.json');assert len(summary)==3
    t=summary[0]['temporal'];arm=summary[0]['arm'];subject=summary[0]['subject']
    for row in summary:
        rows.append(row);sc=row['scenario'];saved=read(folder/f'{sc}_metrics.json')
        pred,logits=verify(folder/f'predictions_{sc}.npz',row,('target',subject,sc))
        confusion=np.array(saved['confusion']);assert confusion.sum()==row['windows']
        with np.load(folder/f'predictions_{sc}.npz') as d:
            actual=np.bincount(d['labels']*33+d['predictions'],minlength=33*33).reshape(33,33)
            assert np.array_equal(actual,confusion)
        support[subject,sc]=np.flatnonzero(confusion.sum(1)==0).tolist()
        if sc=='ideal':basepred,baselogits=pred,logits
    records=read(folder/'stress_diagnostics.json')['records'];assert len(records)==14
    for d in records:
        rotation=d['kind']=='independent_fixed_rotation';suffix=d['seed'] if rotation else d['sensor']
        pred,logits=verify(folder/f"{'rotation' if rotation else 'dropout'}_{suffix}.npz",d['metrics'],('target',subject,'ideal'))
        if rotation:
            rotation_disagree+=int((pred!=basepred).sum());rotation_logit_max=max(rotation_logit_max,float(np.max(abs(logits-baselogits))))
        else:drop[t,arm].append(100*d['metrics']['macro_f1'])
assert len(rows)==270 and checked==1710
by={(x['temporal'],x['arm'],x['scenario'],x['subject'],x['seed']):x for x in rows}
groups=collections.defaultdict(list)
for x in rows:groups[x['temporal'],x['arm'],x['scenario']].append(x)
aggregate=[dict(temporal=t,arm=a,scenario=s,accuracy=100*mean(x['accuracy'] for x in rs),macro_f1=100*mean(x['macro_f1'] for x in rs)) for (t,a,s),rs in sorted(groups.items())]
contrasts=[]
for t in ('old','ssm'):
    for control in ('none_robust','local_repair'):
        for sc in ('ideal','self','induced'):
            subject_f1=[100*mean(by[t,'graph_repair',sc,s,k]['macro_f1']-by[t,control,sc,s,k]['macro_f1'] for k in (13,29,47)) for s in (2,5,15)]
            subject_acc=[100*mean(by[t,'graph_repair',sc,s,k]['accuracy']-by[t,control,sc,s,k]['accuracy'] for k in (13,29,47)) for s in (2,5,15)]
            contrasts.append(dict(temporal=t,control=control,scenario=sc,subject_f1_pp=subject_f1,mean_f1_pp=mean(subject_f1),subject_accuracy_pp=subject_acc,mean_accuracy_pp=mean(subject_acc)))
subject_rows=[dict(temporal=t,arm=a,scenario=sc,subject=s,macro_f1=100*mean(by[t,a,sc,s,k]['macro_f1'] for k in (13,29,47))) for t in ('old','ssm') for a in ('none_robust','local_repair','graph_repair') for sc in ('ideal','self','induced') for s in (2,5,15)]
summary=dict(verified_prediction_files=checked,job_count=90,epoch_count=2790,source_metric_rows=180,target_metric_rows=270,aggregate=aggregate,graph_contrasts=contrasts,subject_means=subject_rows,dropout_f1=[dict(temporal=k[0],arm=k[1],macro_f1=mean(v)) for k,v in sorted(drop.items())],missing_class_ids=[dict(subject=k[0],scenario=k[1],absent=v) for k,v in sorted(support.items())],rotation_prediction_disagreements=rotation_disagree,rotation_max_absolute_logit_difference=rotation_logit_max,mean_minutes_per_job=[dict(temporal=k[0],arm=k[1],minutes=mean(v)/60) for k,v in sorted(durations.items())],epoch_hours=sum(sum(x) for x in durations.values())/3600,receipt_hours=receipt['seconds']/3600,best_epoch31_count=best_epochs.count(31),local_checkpoint_files=len(list(B.glob('jobs/*/*.pt'))),T08='OPEN; no graph promotion',aggregation='Mean seeds within each subject, then equal subjects; balanced 3x3 design, no pooled-window weighting')
(OUT/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
arms=['none_clean','none_robust','physical_robust','local_repair','graph_repair'];labels=['Clean no graph','Robust no graph','Physical graph','Local repair','Graph repair']
for ti,t in enumerate(('old','ssm')):
    ax=axes[ti]
    for sc,offset in [('ideal',-.24),('self',0),('induced',.24)]:
        values=[next(x['macro_f1'] for x in aggregate if x['temporal']==t and x['arm']==a and x['scenario']==sc) for a in arms]
        ax.bar(np.arange(5)+offset,values,.24,label=sc)
    ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.set_ylim(0,100);ax.set_ylabel('Fixed-33 macro-F1 (%)');ax.set_title(t+' backbone: equal-subject means');ax.legend()
fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,t in zip(axes,('old','ssm')):
    for control in ('none_robust','local_repair'):
        v=next(x for x in contrasts if x['temporal']==t and x['control']==control and x['scenario']=='induced')
        ax.plot([2,5,15],v['subject_f1_pp'],'o-',label='graph minus '+control)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks([2,5,15]);ax.set(xlabel='Held-out subject (seeds averaged)',ylabel='Macro-F1 difference (percentage points)',title=t+' induced placement');ax.legend(fontsize=8)
fig.savefig(OUT/'paired_induced.png',dpi=180);fig.savefig(OUT/'paired_induced.pdf');plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k not in ('aggregate','graph_contrasts','subject_means','dropout_f1','missing_class_ids','mean_minutes_per_job')},indent=2))
print('INDUCED SUBJECT MEANS', [x for x in subject_rows if x['temporal']=='ssm' and x['scenario']=='induced'])
