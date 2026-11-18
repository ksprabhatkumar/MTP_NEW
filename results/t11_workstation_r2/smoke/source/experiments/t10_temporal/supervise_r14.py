"""Durable revision-14 continuation; no scientific source changes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/t10_laptop_remaining_20261117_r1'

def record(state, **fields):
    payload = dict(state=state, supervisor_pid=os.getpid(), updated=time.time(), **fields)
    with (BASE / 'supervisor_events.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps(payload) + '\n')
    tmp = BASE / 'supervisor_status.tmp'
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf8')
    os.replace(tmp, BASE / 'supervisor_status.json')

def main():
    os.chdir(ROOT)
    from experiments.t10_temporal.run import queue_lock, validate_receipt, validate_t09, inventory, data_hashes
    with queue_lock(BASE / '.supervisor.lock'):
        try:
            # Acquire/release the existing smoke lock before resuming; failure
            # means another writer may exist. Never bypass or remove this lock.
            with queue_lock(BASE / 'smoke/.queue.lock'):
                pass
            record('smoke_resuming')
            common = [sys.executable, '-u', '-m', 'experiments.t10_temporal.run', '--owner', 'laptop_remaining', '--preflight', str(BASE/'preflight/preflight.json'), '--t09-output', str(ROOT/'results/t09_selective_ssm_r10')]
            smoke = common + ['--stage', 'smoke', '--output', str(BASE/'smoke')]
            with (BASE/'supervisor_children.log').open('a', encoding='utf8') as log:
                subprocess.run(smoke, check=True, stdout=log, stderr=subprocess.STDOUT)
                validate_receipt(BASE/'smoke/receipt.json', 'laptop_remaining')
                audit = json.loads((BASE/'smoke/audit.json').read_text())
                assert audit['passed'] and audit['audited_jobs'] == 10
                receipt = json.loads((BASE/'smoke/receipt.json').read_text())
                from experiments.t10_temporal.model import ARMS, TEMPORALS
                expected = {f'{t}__{a}__subject_2__seed_13' for t in TEMPORALS for a in ARMS}
                assert set(receipt['job_ids']) == expected
                pre = json.loads((BASE/'preflight/preflight.json').read_text())
                assert pre['passed'] and pre['sources'] == inventory() and pre['data'] == data_hashes()
                validate_t09(ROOT/'results/t09_selective_ssm_r10')
                assert not (BASE/'full').exists(), 'Full output already exists; inspect before any continuation'
                record('smoke_passed')
                full = common + ['--stage', 'full', '--output', str(BASE/'full'), '--smoke-receipt', str(BASE/'smoke/receipt.json')]
                proc = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT)
                launch = dict(command=full, pid=proc.pid, supervisor_pid=os.getpid(), started=time.time(), smoke_receipt=str(BASE/'smoke/receipt.json'), expected_jobs=30)
                (BASE/'full_launch_receipt.json').write_text(json.dumps(launch, indent=2), encoding='utf8')
                record('full_running', child_pid=proc.pid)
                code = proc.wait()
                assert code == 0, f'full runner exit {code}'
                result = json.loads((BASE/'full/receipt.json').read_text())
                assert result['passed'] and result['jobs_passed'] == 30
                record('full_complete', jobs=30)
        except BaseException:
            record('stopped', error=traceback.format_exc())
            raise

if __name__ == '__main__':
    main()
