"""CPU-staged folds and explicit FP32 evaluation; no target fitting."""
import numpy as np
import torch
from torch.nn import functional as F
from o2_features import features
from .common import representation
from workstation_experiments import metric_values

VALIDATION={2:3,5:8,15:16}

def build_fold(corpus,subject):
    validation=VALIDATION[subject]
    assert validation in corpus.eligible_validation
    ideal=corpus.scenarios=='ideal'
    masks={'train':ideal & corpus.valid & ~np.isin(corpus.subjects,[subject,validation]),
           'val_ideal':ideal & corpus.valid & (corpus.subjects==validation),
           'val_self':(corpus.scenarios=='self') & corpus.valid & (corpus.subjects==validation)}
    indices={k:np.flatnonzero(v) for k,v in masks.items()};data={}
    for k,ix in indices.items():
        assert len(ix)>0
        encoded=torch.empty((len(ix),50,9,6 if representation()=='raw' else 2),dtype=torch.float32)
        for start in range(0,len(ix),256):encoded[start:start+256]=features(corpus.x[ix[start:start+256]],representation())
        data[k]=dict(x=encoded,y=corpus.y[ix],a=corpus.availability[ix])
    mean=data['train']['x'].mean((0,1),keepdim=True)
    std=data['train']['x'].std((0,1),keepdim=True).clamp_min(1e-6)
    for part in data.values():
        part['x'].sub_(mean).div_(std).mul_(part['a'][:,None,:,None])
        assert torch.isfinite(part['x']).all()
    raw=corpus.x[indices['train']]
    scale=raw.std((0,1)).reshape(9,2,3).square().mean(-1).sqrt().clamp_min(1e-4).repeat_interleave(3,-1)
    metadata={k:dict(corpus_indices=ix,subjects=corpus.subjects[ix],recordings=corpus.sources[ix],scenarios=corpus.scenarios[ix],window_starts=corpus.starts[ix].numpy()) for k,ix in indices.items()}
    train_subjects=sorted(set(corpus.subjects[indices['train']].tolist()))
    assert subject not in train_subjects and validation not in train_subjects
    summary=dict(test_subject_excluded=subject,validation_subject=validation,train_subjects=train_subjects,
        counts={k:len(v) for k,v in indices.items()},recordings={k:sorted(set(corpus.sources[v].tolist())) for k,v in indices.items()},
        classes={k:sorted(set(corpus.y[ix].tolist())) for k,ix in indices.items()},
        class_support={k:torch.bincount(corpus.y[ix],minlength=33).tolist() for k,ix in indices.items()},
        available_sensor_windows={k:corpus.availability[ix].sum(0).tolist() for k,ix in indices.items()},
        invalid_recordings=corpus.valid_manifest['excluded'],
        normalization='CPU float32 train-only mean/std, correction=1',representation=representation(),test_data_evaluated=False)
    assert not set(summary['recordings']['train']).intersection(summary['recordings']['val_ideal']+summary['recordings']['val_self'])
    if subject==15:assert all(len(summary['classes'][k])==33 for k in ('val_ideal','val_self'))
    return dict(data=data,indices=indices,metadata=metadata,mean=mean,std=std,raw_train=raw,raw_scale=scale,summary=summary)

@torch.inference_mode()
def evaluate(model,part,batch_size,artifact=None,metadata=None):
    model.eval();outputs=[];labels=[]
    for start in range(0,len(part['y']),batch_size):
        logits,_=model(part['x'][start:start+batch_size].cuda(),part['a'][start:start+batch_size].cuda())
        assert logits.dtype==torch.float32 and torch.isfinite(logits).all()
        outputs.append(logits.cpu().numpy());labels.append(part['y'][start:start+batch_size].numpy())
    y=np.concatenate(labels);logits=np.concatenate(outputs);pred=logits.argmax(-1)
    metrics=metric_values(y,pred);metrics['loss']=float(F.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y)))
    if artifact is not None:np.savez_compressed(artifact,labels=y,predictions=pred,logits=logits,**metadata)
    return metrics,y,pred
