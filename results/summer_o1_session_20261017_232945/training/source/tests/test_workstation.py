import unittest
import numpy as np
import torch
from pathlib import Path
from contextlib import nullcontext
import uuid
from unittest.mock import patch
from baselines.dynamicwhar_adapter import Objective1DynamicWHAR
from models.mamba_whar import GAE_Mamba_WHAR, ResidualDynamicGraph
from workstation_experiments import metric_values
import workstation_experiments as experiments
from prepare_workstation_data import COLS, STARTS


class WorkstationTests(unittest.TestCase):
    def test_dynamicwhar_correct_layout_and_backward(self):
        model = Objective1DynamicWHAR()
        x = torch.randn(2, 50, 9, 6, requires_grad=True)
        logits, _ = model(x)
        self.assertEqual(logits.shape, (2, 33))
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        model.eval()
        self.assertEqual(model(x[:1])[0].shape, (1, 33))

    def test_realdisp_channels_exclude_magnetometer_quaternion(self):
        self.assertEqual(len(COLS), 54)
        self.assertEqual(len(set(COLS)), 54)
        for node, start in enumerate(STARTS):
            self.assertEqual(COLS[node*6:node*6+6], list(range(start, start+6)))

    def test_graph_masks_every_sender_and_has_useful_gradients(self):
        graph = ResidualDynamicGraph(3, 6, 8)
        x = torch.randn(2, 3, 6, requires_grad=True)
        prior = torch.ones(3, 3) - torch.eye(3)
        availability = torch.tensor([[True, False, True], [False, False, False]])
        for variant in graph.VALID_VARIANTS:
            adjacency, _ = graph(x, prior, availability, variant)
            self.assertTrue(torch.equal(adjacency[:, :, 1], torch.zeros(2, 3)))
            self.assertTrue(torch.equal(adjacency[1], torch.zeros(3, 3)))
            self.assertTrue(torch.isfinite(adjacency).all())
        adjacency, _ = graph(x, prior)
        (adjacency * torch.arange(9).reshape(1, 3, 3)).sum().backward()
        self.assertGreater(graph.q_proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(graph.q_proj.weight.grad).all())

    def test_gae_masked_reconstruction_backward(self):
        model = GAE_Mamba_WHAR(hidden_dim=8, use_reconstruction=True)
        clean = torch.randn(2, 5, 9, 6)
        a = torch.ones(2, 9, dtype=torch.bool)
        a[:, 3] = False
        logits, reconstruction = model(clean * a[:, None, :, None], a)
        self.assertEqual(reconstruction.shape, clean.shape)
        loss = logits.square().mean() + (reconstruction[:, :, 3] - clean[:, :, 3]).square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_macro_f1_uses_declared_class_set(self):
        metrics = metric_values(np.array([0, 0]), np.array([0, 1]), [0, 1])
        self.assertAlmostEqual(metrics['macro_f1'], 1/3)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires workstation GPU')
    def test_train_only_normalization_and_disjoint_records(self):
        corpus = experiments.Corpus.__new__(experiments.Corpus)
        corpus.subjects = np.repeat(np.arange(1, 18), 3)
        corpus.scenarios = np.array(['ideal'] * 51)
        corpus.sources = np.array([f'subject{s}' for s in corpus.subjects])
        corpus.x = torch.randn(51, 50, 9, 6)
        corpus.y = torch.arange(51) % 33
        corpus.availability = torch.ones(51, 9, dtype=torch.bool)
        first = corpus.fold(2)
        corpus.x[np.isin(corpus.subjects, [2, 3])] += 1000
        second = corpus.fold(2)
        self.assertTrue(torch.equal(first['mean'], second['mean']))
        self.assertTrue(torch.equal(first['std'], second['std']))
        self.assertEqual(first['mean'].shape, (1, 1, 9, 6))
        recordings = first['summary']['recordings']
        self.assertFalse(set(recordings['train']) & set(recordings['validation']))
        self.assertFalse(set(recordings['train']) & set(recordings['ideal']))
        self.assertFalse(set(recordings['validation']) & set(recordings['ideal']))

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires workstation GPU')
    def test_resume_after_interruption_preserves_completed_epoch(self):
        Path('tmp').mkdir(exist_ok=True)
        x = torch.randn(8, 50, 9, 6, device='cuda')
        y = torch.arange(8, device='cuda')
        a = torch.ones(8, 9, dtype=torch.bool, device='cuda')
        fold = {'data': {'train': (x, y, a), 'validation': (x, y, a), 'ideal': (x, y, a)},
                'indices': {'ideal': np.arange(8)}, 'mean': torch.zeros(1, 1, 9, 6),
                'std': torch.ones(1, 1, 9, 6), 'summary': {'synthetic_test_only': True}}
        condition = experiments.CONDITIONS[0]
        original = experiments.train_epoch
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('simulated interruption')
            return original(*args, **kwargs)
        # mkdtemp's restrictive Windows ACL conflicts with the managed sandbox.
        # Keep tiny diagnostic test artifacts under the workspace instead.
        with nullcontext(Path('tmp') / ('resume_test_' + uuid.uuid4().hex)) as directory:
            output = Path(directory)
            with patch.object(experiments, 'train_epoch', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    experiments.train_job(fold, condition, 2, 13, 2, 8, output, limit=1)
            self.assertTrue((output / 'latest.pt').exists())
            result = experiments.train_job(fold, condition, 2, 13, 2, 8, output, limit=1)
            self.assertEqual([r['epoch'] for r in result['history']], [1, 2])
            self.assertTrue((output / 'predictions_ideal.npz').exists())


if __name__ == '__main__':
    unittest.main()
