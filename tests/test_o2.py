import unittest
import numpy as np
import torch

from models.mamba_whar import GAE_Mamba_WHAR
from o2_features import features, random_rotations, rotate_raw
from o2_experiments import validation_selection


class OrientationFeatureTests(unittest.TestCase):
    def test_invariants_survive_independent_sensor_rotations(self):
        torch.manual_seed(32)
        raw = torch.randn(4, 12, 9, 6)
        rotation = random_rotations(4, 9, 'cpu')
        self.assertTrue(torch.allclose(rotation.transpose(-1,-2) @ rotation,
                         torch.eye(3).expand(4,9,3,3), atol=1e-5))
        self.assertTrue(torch.allclose(torch.linalg.det(rotation), torch.ones(4,9), atol=1e-5))
        rotated = rotate_raw(raw, rotation)
        for kind in ('magnitude', 'mag_dot'):
            before, after = features(raw, kind), features(rotated, kind)
            self.assertTrue(torch.allclose(before, after, atol=2e-5, rtol=2e-5))
            mean = before.mean((0,1), keepdim=True)
            std = before.std((0,1), keepdim=True).clamp_min(1e-6)
            self.assertTrue(torch.allclose((before-mean)/std, (after-mean)/std,
                                            atol=2e-4, rtol=2e-4))
            model = GAE_Mamba_WHAR(in_features=before.shape[-1], hidden_dim=8, dropout=0).eval()
            with torch.inference_mode():
                prediction_before, _ = model((before-mean)/std)
                prediction_after, _ = model((after-mean)/std)
            self.assertTrue(torch.allclose(prediction_before, prediction_after, atol=2e-4, rtol=2e-4))

    def test_lagged_dot_first_step_uses_current_vector(self):
        raw = torch.zeros(1, 2, 1, 6)
        raw[0, 0, 0, :3] = torch.tensor([3., 4., 0.])
        raw[0, 1, 0, :3] = torch.tensor([1., 2., 0.])
        f = features(raw, 'mag_dot')
        self.assertEqual(float(f[0, 0, 0, 3]), 25.)
        self.assertEqual(float(f[0, 1, 0, 3]), 11.)

    def test_validation_rule_uses_common_classes_and_ideal_floor(self):
        ideal = ({}, np.array([0,1,1]), np.array([0,1,0]))
        self_place = ({}, np.array([0,1,2]), np.array([0,1,2]))
        result = validation_selection(ideal, self_place, [0,1])
        self.assertAlmostEqual(result['ideal_common_f1'], (2/3+2/3)/2)
        self.assertAlmostEqual(result['self_common_f1'], 1.)
        self.assertLess(result['score'], (result['ideal_common_f1']+1)/2)


if __name__ == '__main__':
    unittest.main()
