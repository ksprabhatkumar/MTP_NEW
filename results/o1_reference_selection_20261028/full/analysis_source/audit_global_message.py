"""Post-training inference audit using distant source-window donor values.

Does not train or select checkpoints. Source indices/labels are never altered.
"""
from pathlib import Path
import json
import numpy as np
import torch
from models.summer_gated_neighbor import SummerGatedNeighbor
from models.summer_gated_neighbor_open import SummerOpenGatedNeighbor
from models.summer_reference_correction import SummerReferenceCorrection
from run_o1_selection import build_fold,precision_context
from o2_features import O2Corpus
from workstation_experiments import metric_values,write_json

@torch.inference_mode()
def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/o1_t01_t08_20261028/full')
    parser.add_argument('--graph-arm',default='gated_neighbor')
    args=parser.parse_args()
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parent
    out=root/args.output
    assert (out/'completed.json').exists(), 'Wait for the frozen queue to finish'
    corpus=O2Corpus(); rows=[]
    for subject in (2,5):
        fold=build_fold(corpus,subject)
        for seed in (13,29):
            job=out/f'jobs/{args.graph_arm}/subject_{subject:02d}_seed_{seed}'
            state=torch.load(job/'best.pt',map_location='cpu',weights_only=False)
            model_type=SummerOpenGatedNeighbor if state['config'].get('initial_gate_bias')==2. else SummerGatedNeighbor
            if args.graph_arm=='message_correction': model_type=SummerReferenceCorrection
            model=model_type(args.graph_arm).cuda().eval()
            model.load_state_dict(state['model'])
            for split in ('val_ideal','val_self'):
                part=fold['data'][split]; count=len(part['y'])
                donor=(torch.arange(count)+count//2)%count
                labels=part['y'].numpy();outputs=[]
                for start in range(0,count,64):
                    ix=torch.arange(start,min(start+64,count)); di=donor[ix]
                    x=part['x'][ix].cuda();a=part['a'][ix].cuda()
                    dx=part['x'][di].cuda(); da=part['a'][di].cuda()
                    b,t,n,_=dx.shape
                    with precision_context():
                        mask=da[:,None,:].expand(b,t,n).reshape(b*t,n,1)
                        dl=model.encoder((dx*da[:,None,:,None]).reshape(b*t,n,-1),dx.new_zeros(b*t,n,n))*mask
                        dv=model.message_value(dl)*mask
                        handle=model.message_value.register_forward_hook(lambda module,args,value: dv)
                        try: logits,_=model(x,a)
                        finally: handle.remove()
                    outputs.append(logits.float().cpu().numpy())
                logits=np.concatenate(outputs);metrics=metric_values(labels,logits.argmax(-1))
                normal=json.loads((job/'result.json').read_text())['metrics'][split]
                np.savez_compressed(job/f'distant_donor_{split}.npz',logits=logits,labels=labels,
                    corpus_indices=fold['indices'][split],donor_indices=fold['indices'][split][donor.numpy()])
                rows.append(dict(subject=subject,seed=seed,split=split,metrics=metrics,
                    normal_f1=normal['macro_f1'],difference_pp=100*(metrics['macro_f1']-normal['macro_f1']),
                    donor_different_label_fraction=float((part['y'][donor]!=part['y']).float().mean())))
            del model
        del fold
    write_json(out/'distant_donor_audit.json',dict(source_only=True,
        intervention='Rotate donor values by half the complete validation set; preserve receiver local features and receiver adjacency; recompute gate using donor messages.',rows=rows))
    print(json.dumps([{k:v for k,v in r.items() if k!='metrics'} for r in rows],indent=2))

if __name__=='__main__':main()
