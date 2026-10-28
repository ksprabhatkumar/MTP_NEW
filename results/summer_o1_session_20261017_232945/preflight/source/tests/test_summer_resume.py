import json
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch
import numpy as np
import torch
import summer_o1_experiments as experiments


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA integration test')
class SummerResumeTests(unittest.TestCase):
    def test_epoch_resume_restores_scheduler_optimizer_and_rng(self):
        root=Path(__file__).resolve().parents[1]/'tmp'/('summer_resume_'+uuid.uuid4().hex)
        torch.manual_seed(12)
        x=torch.randn(8,10,9,6,device='cuda')
        part={'x':x,'raw':x,'y':torch.arange(8,device='cuda')%3,
              'a':torch.ones(8,9,device='cuda',dtype=torch.bool)}
        fold={'data':{'train':part,'val_ideal':part,'val_self':part},
              'indices':{'train':np.arange(8),'val_ideal':np.arange(8),'val_self':np.arange(8)},
              'mean':torch.zeros(1,1,9,6),'std':torch.ones(1,1,9,6),
              'summary':{'feature_mode':'raw'}}
        c=experiments.Experiment('residual','residual')
        original=experiments.save_checkpoint
        def interrupt(path,state):
            original(path,state)
            if path.name=='latest.pt' and state['epoch']==2:
                raise RuntimeError('simulated power interruption')
        with patch.object(experiments,'save_checkpoint',interrupt):
            with self.assertRaisesRegex(RuntimeError,'simulated power'):
                experiments.run_job(fold,c,2,13,4,4,root/'resumed',validation_only=True)
        experiments.run_job(fold,c,2,13,4,4,root/'resumed',validation_only=True)
        experiments.run_job(fold,c,2,13,4,4,root/'control',validation_only=True)
        resumed=torch.load(root/'resumed/latest.pt',weights_only=False,map_location='cpu')
        control=torch.load(root/'control/latest.pt',weights_only=False,map_location='cpu')
        self.assertEqual(resumed['scheduler'],control['scheduler'])
        for name in resumed['model']:
            torch.testing.assert_close(resumed['model'][name],control['model'][name],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
