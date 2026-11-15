"""Revision 8: author-reference parity and bounded laptop calibration; no full training."""
import argparse,ast,copy,hashlib,json,math,platform,sys,time,traceback,urllib.request
from pathlib import Path
import torch
from torch.nn import functional as F
from models.selective_mamba_temporal import SelectiveMambaTemporal
from models.summer_dynamic import SummerDynamicWHAR
ROOT=Path(__file__).resolve().parents[2]
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'T09-reference-audit'}),timeout=30) as r:return r.read()
def reference(out):
    commit=json.loads(fetch('https://api.github.com/repos/state-spaces/mamba/commits/main'))['sha']
    paths=['mamba_ssm/modules/mamba_simple.py','mamba_ssm/ops/selective_scan_interface.py']
    refs={}
    for path in paths:
        data=fetch(f'https://raw.githubusercontent.com/state-spaces/mamba/{commit}/{path}')
        dest=out/Path(path).name;dest.write_bytes(data);refs[path]=sha(dest)
    write(out/'reference.json',dict(commit=commit,hashes=refs))
    tree=ast.parse((out/'selective_scan_interface.py').read_text(encoding='utf8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='selective_scan_ref')
    from einops import rearrange,repeat
    env=dict(torch=torch,F=F,rearrange=rearrange,repeat=repeat)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'author_selective_scan_ref','exec'),env)
    return env['selective_scan_ref']
def independent(m,x,scan):
    u,z=m.in_proj(m.norm(x)).chunk(2,-1)
    u=F.silu(m.conv1d(u.transpose(1,2))[...,:x.shape[1]])
    dt,b,c=torch.split(m.x_proj(u.transpose(1,2)),[m.dt_rank,m.d_state,m.d_state],-1)
    delta=F.linear(dt,m.dt_proj.weight,None).transpose(1,2)
    y=scan(u,delta,-m.A_log.exp(),b.transpose(1,2),c.transpose(1,2),m.D,z.transpose(1,2),delta_bias=m.dt_proj.bias,delta_softplus=True)
    return x+m.dropout(m.out_proj(y.transpose(1,2)))
def validation(scan,out):
    torch.manual_seed(13);m=SelectiveMambaTemporal(8,3,2,4,2,0.).cuda().eval();r=copy.deepcopy(m)
    x=torch.randn(2,7,8,device='cuda',requires_grad=True);xx=x.detach().clone().requires_grad_()
    y=m(x);yy=independent(r,xx,scan);torch.testing.assert_close(y,yy,atol=1e-5,rtol=1e-4)
    probe=torch.randn_like(y);(y*probe).sum().backward();(yy*probe).sum().backward()
    errors={}
    for (name,p),(rn,q) in zip(m.named_parameters(),r.named_parameters()):
        assert name==rn and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,atol=1e-4,rtol=1e-3);errors[name]=float((p.grad-q.grad).abs().max())
    torch.testing.assert_close(x.grad,xx.grad,atol=1e-4,rtol=1e-3)
    for name in ('A_log','x_proj.weight','dt_proj.weight'):
        g=dict(m.named_parameters())[name].grad;assert torch.isfinite(g).all() and g.abs().sum()>0
    # Independently verify selective B/C projection slices receive gradients.
    assert m.x_proj.weight.grad[m.dt_rank:m.dt_rank+m.d_state].abs().sum()>0
    assert m.x_proj.weight.grad[m.dt_rank+m.d_state:].abs().sum()>0
    with torch.no_grad():
        baseline=m(x);changed=x.detach().clone();changed[:,4:]+=5
        torch.testing.assert_close(m(changed)[:,:4],baseline[:,:4],atol=1e-6,rtol=0)
        torch.testing.assert_close(m(x),baseline,atol=0,rtol=0)
        for t in range(1,8):torch.testing.assert_close(m(x[:,:t])[:,-1],baseline[:,t-1],atol=1e-5,rtol=1e-4)
    torch.save(m.state_dict(),out/'roundtrip.pt');r.load_state_dict(torch.load(out/'roundtrip.pt',weights_only=True))
    torch.testing.assert_close(m(x),r(x),atol=0,rtol=0)
    return dict(passed=True,output_max=float((y-yy).abs().max()),gradient_max=errors,input_gradient_max=float((x.grad-xx.grad).abs().max()),checks=['author scan and full projection wiring','suffix causality','prefix/step endpoint equality','reset determinism','finite nonzero selective gradients','checkpoint roundtrip'])
def groups(model):
    return [dict(params=[p for p in model.parameters() if not getattr(p,'_no_weight_decay',False)],weight_decay=5e-4),dict(params=[p for p in model.parameters() if getattr(p,'_no_weight_decay',False)],weight_decay=0.)]
def rng():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(s):torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])
def calibration(out):
    from o2_features import O2Corpus
    from run_o1_selection import build_fold
    started=time.perf_counter();fold=build_fold(O2Corpus(),2);preprocess=time.perf_counter()-started
    data=fold['data']['train'];n=len(data['y']);gen=torch.Generator().manual_seed(13);indices=torch.randperm(n,generator=gen)[:64]
    x=data['x'][indices].cuda();a=data['a'][indices].cuda();y=data['y'][indices].cuda();rows=[]
    for graph in ('none','physical'):
        for temporal in ('old','ssm'):
            torch.manual_seed(13);torch.cuda.manual_seed_all(13)
            m=SummerDynamicWHAR(in_features=5,hidden_dim=16,graph_variant=graph,use_reconstruction=False).cuda()
            if temporal=='ssm':m.mamba=SelectiveMambaTemporal().cuda()
            opt=torch.optim.Adam(groups(m),lr=1e-4);m.train()
            def step():
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x,a)[0],y);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();return float(loss.detach())
            # Serialize optimizer and all RNG, then compare the next stochastic step.
            step();saved=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),order=gen.get_state(),rng=rng())
            path=out/f'{graph}_{temporal}_recovery.pt';torch.save(saved,path)
            order_expected=torch.randperm(n,generator=gen);expected_loss=step();expected=copy.deepcopy(m.state_dict())
            saved=torch.load(path,weights_only=False);m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);gen.set_state(saved['order']);restore(saved['rng'])
            assert torch.equal(order_expected,torch.randperm(n,generator=gen));actual=step();assert abs(actual-expected_loss)<=1e-6
            for k,v in m.state_dict().items():torch.testing.assert_close(v,expected[k],atol=1e-6,rtol=0)
            for _ in range(2):step()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(5):
                torch.cuda.synchronize();t=time.perf_counter();step();torch.cuda.synchronize();times.append(time.perf_counter()-t)
            peak=torch.cuda.max_memory_allocated();m.eval();lat=[]
            with torch.no_grad():
                for i in range(40):
                    torch.cuda.synchronize();t=time.perf_counter();m(x[:1],a[:1]);torch.cuda.synchronize()
                    if i>=10:lat.append(time.perf_counter()-t)
            row=dict(graph=graph,temporal=temporal,recovery_passed=True,step_seconds=times,batch_one_seconds=lat,peak_allocated_bytes=peak,parameters=sum(p.numel() for p in m.parameters()),train_windows=n,estimated_epoch_seconds=sum(times)/5*math.ceil(n/64),estimate_excludes='full validation/checkpoint overhead',order_indices=indices.tolist())
            rows.append(row);write(out/'calibration.json',dict(preprocess_seconds=preprocess,rows=rows,complete=len(rows)==4));print(row,flush=True)
            del m,opt,saved,expected;torch.cuda.empty_cache()
    return rows
class Tee:
    def __init__(self,a,b):self.a=a;self.b=b
    def write(self,s):self.a.write(s);self.b.write(s);self.b.flush()
    def flush(self):self.a.flush();self.b.flush()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    log=(out/'console.log').open('w',encoding='utf8');sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    start=time.perf_counter();write(out/'status.json',dict(state='running',full_training=False))
    try:
        assert torch.cuda.is_available();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),precision='FP32',batch=64,backend='unoptimized reference'))
        write(out/'sources.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'models/selective_mamba_temporal.py',ROOT/'models/summer_dynamic.py',ROOT/'run_o1_selection.py']})
        scan=reference(out);result=validation(scan,out);write(out/'validation.json',result);rows=calibration(out)
        write(out/'audit.json',dict(passed=True,combinations=len(rows),full_training=False,T08_complete=False))
        write(out/'status.json',dict(state='complete',seconds=time.perf_counter()-start))
    except BaseException:
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf8');write(out/'status.json',dict(state='failed',seconds=time.perf_counter()-start));raise
if __name__=='__main__':main()
