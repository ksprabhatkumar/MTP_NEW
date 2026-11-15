# Colab notebook cells — T09 and T10 (revision 12)
Run in order in the synced project. Commands stay in the foreground. Keep the runtime connected during training. T10 output and checkpoints are written to Drive.

## Cell 1 — mount Drive; verify current source, cache and Mamba reference

~~~python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import hashlib, json, os, subprocess, sys
ROOT=Path('/content/MTP_NEW'); assert ROOT.is_dir(), 'Sync current project here first.'; os.chdir(ROOT)
REF=ROOT/'results/t09_reference_r9'
assert all((REF/n).is_file() for n in ('mamba_simple.py','selective_scan_interface.py','LICENSE','reference.json')), 'Copy pinned results/t09_reference_r9 from laptop.'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta=json.loads((REF/'reference.json').read_text()); assert meta['commit']=='e9594ce1c732d97440f0332fdc43170a2294dbfa'
for name,digest in meta['hashes'].items(): assert sha(REF/Path(name).name)==digest, name
assert sha(ROOT/'experiments/t09_selective_ssm/check.py')=='32f3b5f9233087b0444773aedcd3833eb6762717b506454757fc7c433dc89085', 'Sync the reviewed current T09 checker.'
CACHE=ROOT/'dataset/workstation_o1'; assert (CACHE/'audit.json').is_file() and len(list(CACHE.glob('*.npz')))==46
OUT=Path('/content/drive/MyDrive/mtp/results/t09_selective_ssm_colab_r12'); assert not OUT.exists(), 'Preserve old attempt; choose a fresh suffixed folder.'; OUT.parent.mkdir(parents=True,exist_ok=True)
import torch
print('Prechecks passed; GPU:',torch.cuda.get_device_name())
~~~

If a source/cache/reference check fails, sync the exact file first. Do not bypass hash checks.

## Cell 2 — T09 offline validation, no scientific training

~~~python
subprocess.run([sys.executable,'-u','-m','experiments.t09_selective_ssm.check','--reference-dir',str(REF),'--output',str(OUT)],cwd=ROOT,check=True)
status=json.loads((OUT/'status.json').read_text()); audit=json.loads((OUT/'audit.json').read_text())
assert status['state']=='complete' and audit['passed'] is True and audit['combinations']==4
sources=json.loads((OUT/'sources.json').read_text()); sources={k.replace('\\','/'):v for k,v in sources.items()}
expected={'experiments/t09_selective_ssm/check.py':sha(ROOT/'experiments/t09_selective_ssm/check.py'),'models/selective_mamba_temporal.py':sha(ROOT/'models/selective_mamba_temporal.py'),'models/summer_dynamic.py':sha(ROOT/'models/summer_dynamic.py'),'run_o1_selection.py':sha(ROOT/'run_o1_selection.py')}
assert sources==expected, 'T09 source provenance mismatch'
print('T09 passed:',audit,'reference:',meta['commit'])
~~~

## Cell 3 — current-data preflight and T10 ten-job smoke

~~~python
BASE=Path('/content/drive/MyDrive/mtp/results/t10_temporal_r1_colab'); PREFLIGHT=BASE/'smoke/preflight'; SMOKE=BASE/'smoke'
BASE.mkdir(parents=True,exist_ok=True); assert not PREFLIGHT.exists() and not SMOKE.exists(), 'Preserve old attempt; use a fresh _fixN folder.'; PREFLIGHT.mkdir(parents=True)
subprocess.run([sys.executable,'-u','-m','experiments.t10_temporal.checks','--owner','colab','--output',str(PREFLIGHT)],cwd=ROOT,check=True)
pre=json.loads((PREFLIGHT/'preflight.json').read_text()); assert pre['passed'] and pre['jobs']==10 and pre['owner']=='colab'
subprocess.run([sys.executable,'-u','-m','experiments.t10_temporal.run','--owner','colab','--stage','smoke','--output',str(SMOKE),'--preflight',str(PREFLIGHT/'preflight.json'),'--t09-output',str(OUT)],cwd=ROOT,check=True)
receipt=json.loads((SMOKE/'receipt.json').read_text()); assert receipt['passed'] and receipt['stage']=='smoke' and receipt['jobs_passed']==10 and receipt['audit']['passed'] and receipt['audit']['audited_jobs']==10
print('All ten T10 smoke jobs and audits passed.')
~~~

Smoke is ten full-data jobs at four epochs each; it is infrastructure validation and is not among the scientific 40 jobs.

## Cell 4 — Colab’s assigned 30-job full queue (conditional)

Run only after Cell 3 completes successfully. The runner rechecks the smoke receipt, preflight, T09 validation, and source/data fingerprints before training.

~~~python
FULL=BASE/'full'; receipt=json.loads((SMOKE/'receipt.json').read_text())
assert receipt.get('passed') is True and receipt.get('owner')=='colab' and receipt.get('jobs_passed')==10 and receipt.get('audit',{}).get('passed') is True
assert json.loads((OUT/'audit.json').read_text()).get('passed') is True
subprocess.run([sys.executable,'-u','-m','experiments.t10_temporal.run','--owner','colab','--stage','full','--output',str(FULL),'--preflight',str(PREFLIGHT/'preflight.json'),'--smoke-receipt',str(SMOKE/'receipt.json'),'--t09-output',str(OUT)],cwd=ROOT,check=True)
status=json.loads((FULL/'status.json').read_text()); assert status['state']=='complete' and status['jobs_passed']==30 and status['audit']['passed'] and status['audit']['audited_jobs']==30
print('Colab queue complete:',status['audit'])
~~~

Colab owns pairs (excluded subject, seed) 2/29, 5/13 and 5/29, all five arms and both temporal backbones, 16 epochs. Do not run laptop pair 2/13 on Colab. If disconnected, reconnect, keep the same code/output, and rerun Cell 4; resume requires matching fingerprints and checkpoints.
