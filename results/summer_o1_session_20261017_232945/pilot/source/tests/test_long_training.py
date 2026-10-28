import json
from pathlib import Path
import sys
import unittest
import uuid
from unittest.mock import patch

import run_long_training as runner


class LongSessionTests(unittest.TestCase):
    def test_pending_scientific_decisions_prevent_launch(self):
        errors = runner.validate({'ready': False, 'unresolved': ['source missing'], 'tasks': []})
        self.assertIn('source missing', errors)
        self.assertGreaterEqual(len(errors), 2)

    def test_sequential_failure_resume_and_artifact_verification(self):
        root = runner.ROOT / 'tmp' / ('runner_test_' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        output = root / 'run'
        helper = root / 'task.py'
        helper.write_text(
            'from pathlib import Path\nimport sys\n'
            'p=Path(sys.argv[1]); n=sys.argv[2]\n'
            'with (p/"events.txt").open("a") as f: f.write(n+"\\n")\n'
            'if n=="2" and not (p/"retry_allowed").exists(): sys.exit(7)\n'
            '(p/(n+".done")).write_text("complete")\n', encoding='utf-8')
        plan = {'ready': True, 'unresolved': [], 'tasks': [
            {'id': f'task{n}', 'plan_step': 'P4.4',
             'command': [sys.executable, str(helper), '{run_dir}', str(n)],
             'artifacts': ['{run_dir}/' + str(n) + '.done']}
            for n in (1, 2, 3)]}
        path = root / 'plan.json'
        path.write_text(json.dumps(plan), encoding='utf-8')
        with patch.object(runner, 'ROOT', root):
            with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                runner.run(path, output)
            self.assertEqual((output / 'events.txt').read_text().splitlines(), ['1', '2'])
            (output / 'retry_allowed').write_text('yes')
            runner.run(path, output)
            runner.run(path, output)
            self.assertEqual((output / 'events.txt').read_text().splitlines(), ['1', '2', '2', '3'])
            self.assertEqual(json.loads((output / 'status.json').read_text())['state'], 'complete')
            (output / '1.done').unlink()
            with self.assertRaisesRegex(ValueError, 'Completed artifact missing'):
                runner.run(path, output)


if __name__ == '__main__':
    unittest.main()
