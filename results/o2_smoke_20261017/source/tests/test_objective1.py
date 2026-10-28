import unittest

import torch

from models.mamba_whar import GAE_Mamba_WHAR, ResidualDynamicGraph
from process_all_data import _realdisp_file_metadata


class ResidualDynamicGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = ResidualDynamicGraph(num_nodes=3, in_features=6, attention_dim=4)
        self.x = torch.randn(2, 3, 6, requires_grad=True)
        self.prior = torch.tensor([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]])

    def test_residual_graph_is_finite_nonnegative_and_row_normalized(self):
        adjacency, diagnostics = self.graph(self.x, self.prior)
        self.assertEqual(tuple(adjacency.shape), (2, 3, 3))
        self.assertTrue(torch.isfinite(adjacency).all())
        self.assertTrue((adjacency >= 0).all())
        self.assertTrue(torch.allclose(torch.diagonal(adjacency, dim1=1, dim2=2), torch.zeros(2, 3)))
        self.assertTrue(torch.allclose(adjacency.sum(dim=-1), torch.ones(2, 3), atol=1e-6))
        adjacency.sum().backward()
        self.assertTrue(torch.isfinite(self.x.grad).all())
        self.assertIn("branch_weights", diagnostics)

    def test_missing_senders_and_empty_rows_are_safe(self):
        available = torch.tensor([[True, False, False], [False, False, False]])
        adjacency, diagnostics = self.graph(self.x.detach(), self.prior, available)
        self.assertTrue(torch.isfinite(adjacency).all())
        self.assertTrue(torch.equal(adjacency[1], torch.zeros(3, 3)))
        self.assertGreater(int(diagnostics["masked_rows"]), 0)

    def test_diagnostics_are_deterministic_in_eval_mode(self):
        model = GAE_Mamba_WHAR(hidden_dim=8, dropout=0.0)
        model.eval()
        x = torch.randn(2, 5, 9, 6)
        _, _, first = model(x, return_diagnostics=True)
        _, _, second = model(x, return_diagnostics=True)
        self.assertTrue(torch.equal(first["adjacency"], second["adjacency"]))


class DataContractTests(unittest.TestCase):
    def test_realdisp_scenario_names_are_normalized(self):
        self.assertEqual(_realdisp_file_metadata("subject2_mutual4.log"), (2, "induced", "4"))
        self.assertEqual(_realdisp_file_metadata("subject15_ideal.log"), (15, "ideal", "0"))
        self.assertIsNone(_realdisp_file_metadata("unexpected.log"))


if __name__ == "__main__":
    unittest.main()
