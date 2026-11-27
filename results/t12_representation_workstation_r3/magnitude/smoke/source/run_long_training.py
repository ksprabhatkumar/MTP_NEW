"""Sequential process runner for the approved summer-based O1 experiment.

Each task launches an explicit command without a shell, streams its log, checks
its declared artifacts, and finishes before the next starts. The model backend
owns epoch/optimizer checkpoints; this runner owns stage/job-level recovery.
An unresolved scientific plan is deliberately rejected before execution.

  prabhat\Scripts\python.exe run_long_training.py --plan summer_o1_session.json --check
  prabhat\Scripts\python.exe -u run_long_training.py --plan summer_o1_session.json --output results/NEW_RUN
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now().astimezone().isoformat()


def save_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate(plan):
    problems = list(plan.get('unresolved', []))
    if plan.get('ready') is not True:
        problems.append('The scientific plan is not marked ready.')
    tasks = plan.get('tasks', [])
    if not tasks:
        problems.append('No verified training backend/tasks have been registered.')
    ids = []
    for task in tasks:
        ids.append(task.get('id'))
        if not task.get('id') or not task.get('plan_step'):
            problems.append('Every task needs an id and an entire_plan.md step.')
        command = task.get('command')
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            problems.append(f'Invalid argument-list command: {task.get("id")}')
        if not task.get('artifacts'):
            problems.append(f'No success artifacts declared: {task.get("id")}')
    if len(ids) != len(set(ids)):
        problems.append('Task ids must be unique.')
    if any(not isinstance(x, str) or not x.replace('_', '').replace('-', '').isalnum() for x in ids):
        problems.append('Task ids must contain only letters, digits, underscores or hyphens.')
    return problems


def expand(value, output):
    for key, replacement in {'{root}': str(ROOT), '{run_dir}': str(output),
                              '{python}': sys.executable}.items():
        value = value.replace(key, replacement)
    return value


def artifacts_exist(task, output):
    for name in task['artifacts']:
        path = Path(expand(name, output))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def publish_status(output, status, plan_file):
    status['updated'] = now()
    save_json(output / 'status.json', status)
    text = (
        'SUMMER-BASED O1 LONG SESSION HANDOFF\n'
        f'Updated: {status["updated"]}\nWorkspace: {ROOT}\n'
        f'Plan: {plan_file}\nRun: {output}\n'
        f'State: {status["state"]}\nWorker PID: {os.getpid()}\n'
        f'Current entire_plan.md step: {status.get("plan_step", "preflight")}\n'
        f'Current task: {status.get("task", "none")}\n'
        f'Completed tasks: {status.get("completed", 0)}/{status.get("total", 0)}\n'
        f'Error: {status.get("error", "none")}\n\n'
        'Logs: task_logs/<task_id>.log; completed.json; status.json.\n'
        'The task backend saves model/optimizer/scheduler/RNG epoch checkpoints.\n'
        'Resume only after checking that the prior worker and child exited.\n'
        f'Resume: "{sys.executable}" -u "{ROOT / "run_long_training.py"}" '
        f'--plan "{plan_file}" --output "{output}"\n'
        'Same plan/source hashes are required. Completed tasks are skipped only\n'
        'when their declared artifacts remain present. A failed task stops the queue.\n'
        'Keep static, residual dynamic and no-graph results; do not select test\n'
        'settings to obtain a desired winner. See entire_plan.md for next phases.\n'
    )
    (output / 'TRAINING_HANDOFF.txt').write_text(text, encoding='utf-8')
    (ROOT / 'current_step_update.txt').write_text(text, encoding='utf-8')


def run(plan_file, output):
    plan_file, output = Path(plan_file).resolve(), Path(output).resolve()
    plan_bytes = plan_file.read_bytes()
    plan = json.loads(plan_bytes)
    problems = validate(plan)
    if problems:
        raise ValueError('Plan unresolved:\n- ' + '\n- '.join(problems))
    output.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (output / '.session.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError('Another session owns this output directory.')
    status = {'state': 'preflight', 'pid': os.getpid(), 'total': len(plan['tasks']), 'completed': 0}
    child = None
    try:
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in plan.get('source_files', [])}
        signature = {'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(), 'sources': hashes,
                     'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        signature_file = output / 'signature.json'
        if signature_file.exists() and json.loads(signature_file.read_text()) != signature:
            raise ValueError('Plan or source changed; inspect provenance before resuming.')
        save_json(signature_file, signature)
        (output / 'plan.snapshot.json').write_bytes(plan_bytes)
        (output / 'task_logs').mkdir(exist_ok=True)
        complete_file = output / 'completed.json'
        completed = json.loads(complete_file.read_text()) if complete_file.exists() else {}
        status['completed'] = len(completed)
        publish_status(output, status, plan_file)
        for task in plan['tasks']:
            task_id = task['id']
            if task_id in completed:
                if not artifacts_exist(task, output):
                    raise ValueError(f'Completed artifact missing for {task_id}; inspect before rerun.')
                continue
            status.update(state='running', task=task_id, plan_step=task['plan_step'])
            command = [expand(token, output) for token in task['command']]
            status['command'] = command
            publish_status(output, status, plan_file)
            started = now()
            with (output / 'task_logs' / (task_id + '.log')).open('a', encoding='utf-8') as log:
                log.write(f'\nSTART {started}\nCOMMAND {json.dumps(command)}\n')
                log.flush()
                child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                status['child_pid'] = child.pid
                publish_status(output, status, plan_file)
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                child.stdout.close()
                code = child.wait()
                child = None
            if code != 0:
                raise RuntimeError(f'Task {task_id} exited {code}; inspect its log and epoch checkpoint.')
            if not artifacts_exist(task, output):
                raise RuntimeError(f'Task {task_id} exited successfully but expected artifacts are absent.')
            completed[task_id] = {'started': started, 'finished': now(), 'plan_step': task['plan_step']}
            save_json(complete_file, completed)
            status['completed'] = len(completed)
            status.pop('child_pid', None)
            publish_status(output, status, plan_file)
        status['state'] = 'complete'
        publish_status(output, status, plan_file)
    except BaseException as error:
        # On interruption the child is stopped before the lock is released, so
        # a resume cannot create two active model writers through this runner.
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if child is not None and child.stdout is not None:
            child.stdout.close()
        status.update(state='failed', error=repr(error), traceback=traceback.format_exc())
        publish_status(output, status, plan_file)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(ROOT / 'summer_o1_session.json'))
    parser.add_argument('--output')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        problems = validate(json.loads(Path(args.plan).read_text(encoding='utf-8')))
        print(json.dumps({'ready': not problems, 'blockers': problems}, indent=2))
        sys.exit(2 if problems else 0)
    if not args.output:
        parser.error('--output is required for execution; reuse it for resume')
    run(args.plan, args.output)
