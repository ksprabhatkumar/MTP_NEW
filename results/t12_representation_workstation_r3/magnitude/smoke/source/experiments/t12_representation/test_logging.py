"""CPU-only regression tests for process shutdown and exception logging."""
import subprocess
import sys
import unittest
from .common import ROOT

class LoggingTests(unittest.TestCase):
    def child(self,body):
        code='''import sys,tempfile
from experiments.t12_representation.common import console_log
before=(sys.stdout,sys.stderr)
with tempfile.TemporaryFile(mode='w+',encoding='utf8') as log:
    with console_log(log):
        print('stdout saved',flush=True)
        print('stderr saved',file=sys.stderr,flush=True)
        BODY
    assert (sys.stdout,sys.stderr)==before
    log.seek(0)
    assert 'stdout saved' in log.read()
print('after close',flush=True)
'''.replace('BODY',body)
        return subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)

    def test_success_exits_zero(self):
        result=self.child('pass')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('after close',result.stdout)
        self.assertNotIn('unraisablehook',result.stderr)

    def test_failure_preserves_traceback(self):
        result=self.child("raise RuntimeError('intentional training failure')")
        self.assertEqual(result.returncode,1,result.stderr)
        self.assertIn('RuntimeError: intentional training failure',result.stderr)
        self.assertNotIn('unraisablehook',result.stderr)

if __name__=='__main__':unittest.main()
