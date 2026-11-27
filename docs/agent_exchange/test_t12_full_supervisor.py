"""CPU-only supervisor regressions: no GPU gates, trainer or subprocess launched."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock

spec=importlib.util.spec_from_file_location('supervisor',Path(__file__).with_name('t12_full_sequential_r24.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SupervisorTests(unittest.TestCase):
    def test_reserved_folder_accepted_once(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'run';token=m.reserve_output(out)
            (out/'sequence_supervisor.log').write_text('launcher opened log')
            m.claim_output(out,token)
            with self.assertRaises(FileExistsError):m.claim_output(out,token)
            with self.assertRaises(FileExistsError):m.reserve_output(out)
    def test_wrong_reservation_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'run';m.reserve_output(out)
            with self.assertRaises(AssertionError):m.claim_output(out,'wrong')
            self.assertFalse((out/'started.json').exists())
    def sequence(self,code=0,audit_error=False):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'run';token=m.reserve_output(out)
            gates=dict(raw_receipt_sha256='raw',magnitude_receipt_sha256='mag',raw_fingerprint={},magnitude_fingerprint={})
            child=Mock(pid=123);child.wait.return_value=code
            with patch.object(m,'ROOT',root),patch.object(m,'validate_gates',return_value=gates),patch.object(m.subprocess,'Popen',return_value=child) as launch,patch.object(m,'full_result',side_effect=RuntimeError('audit failure') if audit_error else None,return_value={'passed':True}):
                if code or audit_error:
                    with self.assertRaises(RuntimeError):m.run_sequence(out,token)
                else:m.run_sequence(out,token)
                state=json.loads((out/'status.json').read_text())
                calls=launch.call_args_list
            return state,calls
    def test_raw_then_magnitude_after_audit(self):
        state,calls=self.sequence()
        self.assertEqual(state['state'],'complete');self.assertEqual(state['completed'],['raw_full','magnitude_full'])
        self.assertEqual(len(calls),2)
        self.assertIn('raw',calls[0].args[0]);self.assertIn('magnitude',calls[1].args[0])
        self.assertNotIn('--detach',calls[0].args[0])
    def test_nonzero_exit_blocks_magnitude(self):
        state,calls=self.sequence(code=1)
        self.assertEqual(state['state'],'failed');self.assertEqual(len(calls),1)
        self.assertFalse(state['magnitude_started'])
    def test_audit_failure_blocks_magnitude(self):
        state,calls=self.sequence(audit_error=True)
        self.assertEqual(state['state'],'failed');self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
