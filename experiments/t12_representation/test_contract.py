"""CPU-only contract checks. GPU recovery/full-data smoke remain mandatory."""
import os,unittest
import torch
from o2_features import features,random_rotations,rotate_raw
from experiments.t10_temporal.model import TemporalFactorial as Original
from .model import TemporalFactorial
from .common import selected,inventory
from .training import make_teacher,loss_for_batch,streams

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.old=os.environ.get('MTP_T12_REPRESENTATION');torch.set_num_threads(4)
    def tearDown(self):
        if self.old is None:os.environ.pop('MTP_T12_REPRESENTATION',None)
        else:os.environ['MTP_T12_REPRESENTATION']=self.old
    def test_matrix_and_shared_weights(self):
        excluded=('encoder.gin_agg.nn.0.','encoder.gin_att.nn.0.','sender_encoder.0.')
        all_ids=set()
        for rep in ('raw','magnitude'):
            os.environ['MTP_T12_REPRESENTATION']=rep
            self.assertEqual(len(selected('smoke')),20);self.assertEqual(len(selected('full')),90)
            all_ids.update(j['id'] for j in selected('full'))
            for temporal in ('old','ssm'):
                torch.manual_seed(13);old=Original('graph_repair',temporal)
                torch.manual_seed(13);new=TemporalFactorial('graph_repair',temporal)
                for k,v in old.state_dict().items():
                    if not k.startswith(excluded):torch.testing.assert_close(v,new.state_dict()[k],atol=0,rtol=0)
        self.assertEqual(len(all_ids),180)
        self.assertIn('experiments/t12_representation/model.py',inventory())
    def test_real_widths_backward_and_invariance(self):
        raw=torch.randn(2,8,9,6);a=torch.ones(2,9,dtype=torch.bool)
        rot=rotate_raw(raw,random_rotations(2,9,'cpu',torch.Generator().manual_seed(7)))
        torch.testing.assert_close(features(raw,'magnitude'),features(rot,'magnitude'),atol=2e-6,rtol=1e-6)
        self.assertFalse(torch.allclose(raw,rot))
        for rep in ('raw','magnitude'):
            os.environ['MTP_T12_REPRESENTATION']=rep
            for temporal in ('old','ssm'):
                for arm in ('none_clean','none_robust','physical_robust','local_repair','graph_repair'):
                    model=TemporalFactorial(arm,temporal)
                    y,_=model(features(raw,rep),a)
                    self.assertEqual(y.shape,(2,33));self.assertTrue(torch.isfinite(y).all())
                    y.square().mean().backward()
                    self.assertIsNotNone(model.encoder.gin_agg.nn[0].weight.grad)
                    if arm=='graph_repair':
                        model.zero_grad(set_to_none=True)
                        encoded=features(raw,rep);m=encoded.mean((0,1),keepdim=True);s=encoded.std((0,1),keepdim=True).clamp_min(1e-6)
                        batch=dict(x=(encoded-m)/s,raw=raw,a=a,y=torch.tensor([0,1]))
                        loss,parts,_=loss_for_batch(model,make_teacher(model),batch,dict(mean=m,std=s,raw_scale=torch.ones(9,6)),streams(13)['corruption'],3)
                        self.assertTrue(torch.isfinite(loss));loss.backward()
                        self.assertIsNotNone(model.sender_encoder[0].weight.grad)

if __name__=='__main__':unittest.main()
