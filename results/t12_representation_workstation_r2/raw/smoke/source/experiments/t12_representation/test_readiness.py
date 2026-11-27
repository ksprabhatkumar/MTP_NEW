"""Readiness regressions without CUDA/model training or external commands."""
import subprocess,unittest
from unittest.mock import patch
from . import readiness

class ReadinessTests(unittest.TestCase):
    def test_actual_native_resources(self):
        values=readiness.system_resources()
        self.assertGreater(values['total_physical_memory_bytes'],0)
        self.assertGreater(values['logical_processors'],0)
    def test_no_powershell_required(self):
        record={}
        with patch.object(readiness.subprocess,'run',return_value=subprocess.CompletedProcess(['nvidia-smi'],0,'GPU inventory','')) as run:
            readiness.collect_hardware(record)
        self.assertEqual(run.call_count,1)
        self.assertEqual(run.call_args.args[0],['nvidia-smi'])
        self.assertGreater(record['system_resources']['total_physical_memory_bytes'],0)
    def test_driver_failure_stops_and_records_stderr(self):
        record={}
        with patch.object(readiness.subprocess,'run',return_value=subprocess.CompletedProcess(['nvidia-smi'],1,'partial','driver failed')):
            with self.assertRaises(RuntimeError):readiness.collect_hardware(record)
        self.assertEqual(record['hardware_inventory'][0]['stderr'],'driver failed')
        self.assertEqual(record['hardware_inventory'][0]['returncode'],1)
    def test_native_failure_is_not_ignored(self):
        with patch.object(readiness.subprocess,'run',return_value=subprocess.CompletedProcess(['nvidia-smi'],0,'GPU','')),patch.object(readiness,'system_resources',side_effect=OSError('native inventory failed')):
            with self.assertRaises(OSError):readiness.collect_hardware({})

if __name__=='__main__':unittest.main()
