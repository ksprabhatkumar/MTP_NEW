import unittest
from pathlib import Path
import torch
from models.summer_dynamic import SummerDynamicWHAR


class SummerGraphTests(unittest.TestCase):
    def test_original_pyg_model_forward_and_gradient_parity(self):
        path = Path(__file__).resolve().parents[1]/'baselines/summer_original/parity_reference.pt'
        self.assertTrue(path.exists(), 'Generate original PyG fixture before running this test')
        saved = torch.load(path, weights_only=False, map_location='cpu')
        model = SummerDynamicWHAR(in_features=9, graph_variant='static_sum', use_reconstruction=True).eval()
        report = model.load_state_dict(saved['state'], strict=False)
        self.assertFalse(report.unexpected_keys)
        x = saved['x'].clone().requires_grad_()
        logits, reconstruction = model(x)
        torch.testing.assert_close(logits, saved['logits'], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(reconstruction, saved['reconstruction'], atol=2e-5, rtol=2e-5)
        (logits.square().mean() + reconstruction.square().mean()).backward()
        torch.testing.assert_close(x.grad, saved['input_gradient'], atol=2e-5, rtol=2e-5)
        for name, gradient in saved['parameter_gradients'].items():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad, gradient, atol=2e-5, rtol=2e-5)

    def test_residual_formula_and_missing_sensor_contract(self):
        torch.manual_seed(4)
        model = SummerDynamicWHAR().eval()
        x = torch.randn(2,8,9,6)
        mask = torch.ones(2,9,dtype=torch.bool)
        mask[0,3] = False
        _, _, d = model(x, mask, return_diagnostics=True)
        a = d['adjacency']
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue((a >= 0).all())
        self.assertEqual(a.diagonal(dim1=-2,dim2=-1).abs().max(), 0)
        self.assertEqual(a[0,:,:,3].abs().max(), 0)
        self.assertEqual(d['latent'][0,:,3].abs().max(), 0)
        torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)))
        h = d['encoded']
        c = torch.softmax(model.q(h) @ model.k(h).transpose(1,2) / (16**.5), -1)
        s = (d['branch_weights'][0] * model.physical + d['branch_weights'][1] *
             torch.nn.functional.softplus(model.global_logits) + d['branch_weights'][2]*c)
        s = s * (1-torch.eye(9)) * mask[:,None,:].expand(2,8,9).reshape(16,1,9)
        torch.testing.assert_close(a.reshape(16,9,9), s/s.sum(-1,keepdim=True))
        changed = x.clone(); changed[0,:,3] = 10000
        torch.testing.assert_close(model(x,mask)[0], model(changed,mask)[0])
        one = torch.zeros_like(mask); one[:,0]=True
        _, _, d = model(x,one,return_diagnostics=True)
        self.assertEqual(d['adjacency'][:,:,0,:].sum(), 0)
        with self.assertRaisesRegex(ValueError, 'abstain'):
            model(x, torch.zeros_like(mask))

    def test_dynamic_graph_receives_gradients_and_changes_with_input(self):
        model = SummerDynamicWHAR(graph_variant='dynamic').eval()
        x = torch.randn(3,8,9,6)
        logits, _, d = model(x,return_diagnostics=True)
        logits.square().mean().backward()
        self.assertGreater(model.q.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.affinity_temporal.weight.grad.abs().sum().item(), 0)
        self.assertGreater((d['adjacency'][0]-d['adjacency'][1]).abs().max().item(), 1e-7)


if __name__ == '__main__':
    unittest.main()
