"""Read-only checkpoint forward diagnostics. No training or tolerance changes."""
import argparse,json,traceback
from pathlib import Path
import torch
from experiments.t08_utility_gate_20261114.run import ROOT,SOURCE,Original,PredictiveRepair,load_backbone,initial_checks
from run_o1_selection import build_fold
from o2_features import O2Corpus

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    corpus=O2Corpus();rows=[]
    for subject in (2,5):
        fold=build_fold(corpus,subject)
        for seed in (13,29):
            state=torch.load(SOURCE/f'jobs/none_robust/subject_{subject:02d}_seed_{seed}/best.pt',map_location='cpu',weights_only=False)
            for branch in ('local_repair','graph_repair'):
                job=out/f'{branch}_{subject}_{seed}';job.mkdir()
                ref=Original('none_robust').cuda().eval();ref.load_state_dict(state['model'])
                candidate=PredictiveRepair(branch).cuda().eval();load_backbone(candidate,state['model'])
                row=dict(subject=subject,seed=seed,branch=branch)
                try:initial_checks(candidate,ref,fold,job);row['initial_passed']=True
                except Exception:row['initial_passed']=False;row['error']=traceback.format_exc()
                # Fresh zero correction; inspect actual input layouts at the first divergent layer.
                candidate.repair_output.weight.data.zero_();candidate.force_gate=None;candidate.output_mode='ungated'
                captured={};handles=[]
                def hook(label):
                    def capture(module,inputs,output):
                        captured.setdefault(label,[]).append((inputs[0].detach().clone(),output.detach().clone(),list(inputs[0].stride())))
                    return capture
                for label,net in [('reference',ref),('candidate',candidate)]:
                    handles.append(net.mamba.register_forward_hook(hook(label)))
                x=fold['data']['val_ideal']['x'][:8].cuda();a=fold['data']['val_ideal']['a'][:8].cuda()
                with torch.inference_mode():
                    r1=ref(x,a)[0];r2=ref(x,a)[0];c1=candidate(x,a)[0];c2=candidate(x,a)[0]
                    ri,ro,rs=captured['reference'][0];ci,co,cs=captured['candidate'][0]
                    row.update(input_equal=torch.equal(ri,ci),reference_stride=rs,candidate_stride=cs,input_max=float((ri-ci).abs().max()),mamba_max=float((ro-co).abs().max()),logit_max=float((r1-c1).abs().max()),reference_repeat_max=float((r1-r2).abs().max()),candidate_repeat_max=float((c1-c2).abs().max()))
                    # Same layout and exact tensor supplied independently to both Mamba copies.
                    canonical=ri.contiguous();rr=ref.mamba(canonical);cc=candidate.mamba(canonical)
                    row['identical_input_mamba_max']=float((rr-cc).abs().max())
                    row['canonical_stride']=list(canonical.stride())
                for h in handles:h.remove()
                (job/'diagnostics.json').write_text(json.dumps(row,indent=2));rows.append(row)
                print(json.dumps(row),flush=True)
                del ref,candidate,captured,state
                state=torch.load(SOURCE/f'jobs/none_robust/subject_{subject:02d}_seed_{seed}/best.pt',map_location='cpu',weights_only=False)
                torch.cuda.empty_cache()
    (out/'summary.json').write_text(json.dumps(dict(diagnostics_complete=True,training_authorized=False,T08_complete=False,rows=rows),indent=2))
if __name__=='__main__':main()
