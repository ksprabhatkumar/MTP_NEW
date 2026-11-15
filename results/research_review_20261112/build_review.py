"""Read-only experiment review; regenerate evidence and figures from saved predictions."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1];RESULTS=ROOT/'results'
SUITES={'raw':'summer_o1_session_20261017_232945','features':'summer_o2_session_20261021_173341','alternatives':'summer_o1_alternatives_session_20261022_001702'}
def csvsave(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    rows=[];manifest={};lookup={};counts={};recordings=[]
    for suite,folder in SUITES.items():
        paths=sorted((RESULTS/folder/'training/jobs').glob('**/result.json'));counts[suite]=len(paths)
        assert len(paths)=={'raw':126,'features':63,'alternatives':81}[suite]
        for p in paths:
            r=json.loads(p.read_text());assert r['epochs']==31 and len(r['history'])==31
            condition=r['condition'];lookup[suite,condition,r['subject'],r['seed']]=p.parent
            manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            for scenario in ('ideal','self','induced'):
                with np.load(p.parent/f'predictions_{scenario}.npz') as d:
                    y=d['labels'];pred=d['predictions'];acc=accuracy_score(y,pred);f1=f1_score(y,pred,labels=list(range(33)),average='macro',zero_division=0)
                    assert abs(acc-r['metrics'][scenario]['accuracy'])<1e-10 and abs(f1-r['metrics'][scenario]['macro_f1'])<1e-10
                rows.append(dict(suite=suite,condition=condition,subject=r['subject'],seed=r['seed'],scenario=scenario,accuracy=acc*100,macro_f1=f1*100,windows=len(y),best_epoch=r['best_epoch'],source=str(p.relative_to(ROOT))))
                for name,met in r.get('per_recording',{}).get(scenario,{}).items():
                    met=met.get('fixed33',met)
                    recordings.append(dict(suite=suite,condition=condition,subject=r['subject'],seed=r['seed'],scenario=scenario,recording=name,accuracy=met['accuracy']*100,macro_f1=met['macro_f1']*100))
    csvsave('all_job_metrics.csv',rows);csvsave('per_recording_metrics.csv',recordings)
    aggregate=[]
    for suite,condition,scenario in sorted({(r['suite'],r['condition'],r['scenario']) for r in rows}):
        selected=[r for r in rows if (r['suite'],r['condition'],r['scenario'])==(suite,condition,scenario)]
        entry=dict(suite=suite,condition=condition,scenario=scenario,jobs=len(selected))
        for metric in ('accuracy','macro_f1'):
            means=[np.mean([r[metric] for r in selected if r['subject']==s]) for s in (2,5,15)]
            entry[metric]=float(np.mean(means));entry[metric+'_subject_sd']=float(np.std(means,ddof=1))
        aggregate.append(entry)
    csvsave('scenario_comparison.csv',aggregate)
    def value(suite,c,scenario,metric='macro_f1'):return next(r[metric] for r in aggregate if (r['suite'],r['condition'],r['scenario'])==(suite,c,scenario))
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.8),sharey=True)
    variants=[('raw','none','Raw / no graph'),('raw','residual','Raw / residual graph'),('features','mag_dot_none','Mag+dot / no graph'),('features','mag_dot_residual','Mag+dot / residual graph'),('alternatives','mag_dot_gated_pairwise','Mag+dot / gated pairwise')]
    colors=['#9aa4b0','#476b91','#d8932d','#46a69b','#276958']
    for ax,scenario in zip(axes,('ideal','self','induced')):
        vals=[value(s,c,scenario) for s,c,label in variants];bars=ax.bar(range(len(vals)),vals,color=colors)
        ax.set_title(scenario.capitalize());ax.set_xticks(range(len(vals)),[str(i+1) for i in range(len(vals))]);ax.set_ylim(0,107);ax.grid(axis='y',alpha=.2)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+1,f'{v:.2f}',ha='center',fontsize=9)
    axes[0].set_ylabel('Fixed-33 macro-F1 (%)');fig.suptitle('Recorded placement tests: three subjects, three seeds (exploratory)')
    fig.legend([plt.Rectangle((0,0),1,1,color=c) for c in colors],[f'{i+1}. {v[2]}' for i,v in enumerate(variants)],loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,.14,1,.94));fig.savefig(OUT/'placement_comparison.png',dpi=180);fig.savefig(OUT/'placement_comparison.svg');plt.close(fig)
    latest=json.loads((RESULTS/'o1_graph_repair_20261029/full/summary.json').read_text())
    arms=list(latest['means']);fig,ax=plt.subplots(figsize=(9,4.6));xx=np.arange(4);width=.34
    for off,metric,label,color in [(-width/2,'accuracy','Accuracy','#476b91'),(width/2,'macro_f1','Macro-F1','#46a69b')]:
        vals=[latest['means'][a]['val_self'][metric]*100 for a in arms];bars=ax.bar(xx+off,vals,width,label=label,color=color)
        ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
    ax.set_xticks(xx,['Clean no graph','Robust no graph','Local repair','Graph repair']);ax.set_ylim(90,96);ax.set_ylabel('Self-placement validation (%)');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_title('29 October pilot: two validation subjects, one seed\nTruncated y-axis; not held-out induced tests');fig.tight_layout();fig.savefig(OUT/'repair_validation.png',dpi=180);fig.savefig(OUT/'repair_validation.svg');plt.close(fig)
    complement=[]
    for subject in (2,5,15):
        for seed in (13,29,47):
            for scenario in ('ideal','self','induced'):
                p=lookup['raw','none',subject,seed];q=lookup['features','mag_dot_none',subject,seed]
                with np.load(p/f'predictions_{scenario}.npz') as raw,np.load(q/f'predictions_{scenario}.npz') as inv:
                    assert np.array_equal(raw['labels'],inv['labels']) and np.array_equal(raw['corpus_indices'],inv['corpus_indices'])
                    a=raw['predictions']==raw['labels'];b=inv['predictions']==inv['labels']
                    complement.append(dict(subject=subject,seed=seed,scenario=scenario,raw_accuracy=100*a.mean(),invariant_accuracy=100*b.mean(),oracle_accuracy_upper_bound=100*(a|b).mean(),raw_rescue_pp=100*(a&~b).mean(),invariant_rescue_pp=100*(b&~a).mean(),both_wrong_pp=100*(~a&~b).mean()))
    csvsave('expert_complementarity.csv',complement)
    cs={scenario:{key:float(np.mean([r[key] for r in complement if r['scenario']==scenario])) for key in ('raw_accuracy','invariant_accuracy','oracle_accuracy_upper_bound','raw_rescue_pp','invariant_rescue_pp','both_wrong_pp')} for scenario in ('ideal','self','induced')}
    (OUT/'summary.json').write_text(json.dumps(dict(aggregate=aggregate,expert_complementarity=cs,latest_repair=latest),indent=2))
    (OUT/'evidence_audit.json').write_text(json.dumps(dict(passed=True,completed_job_counts=counts,prediction_files_recomputed=len(rows),result_hashes=manifest,notes=['270 historical job histories checked; 810 prediction sets recomputed','three subjects averaged equally after seed averaging','historical induced subjects already inspected; exploratory evidence','oracle uses true labels and is an upper bound only; never a deployable result']),indent=2))
    print(json.dumps(cs,indent=2))
    print('Historical audit passed: 270 jobs, 810 saved prediction sets. Figures and tables written.')
if __name__=='__main__':main()
