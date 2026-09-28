import unittest

import torch
import torch.nn.functional as F
from l2rwplus.losses import PrototypeLoss
from l2rwplus.engine.federation import fedprox_term, moon_loss


class LossTests(unittest.TestCase):
    def test_rectification_has_finite_gradient_and_fixed_targets(self):
        x = torch.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=True)
        targets = {0: torch.tensor([2.0, 1.0], requires_grad=True), 1: torch.zeros(2)}
        actual = PrototypeLoss()(x, torch.tensor([0, 1]), targets, "cpu")
        q = torch.stack([targets[0], targets[1]]).detach()
        expected = 0.5 * F.mse_loss(x, q) + 0.5 * (1 - F.cosine_similarity(x, q)).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNone(targets[0].grad)


    def test_fedprox_does_not_penalize_private_classifier(self):
        encoder = torch.nn.Parameter(torch.tensor([2.0]))
        head = torch.nn.Parameter(torch.tensor([100.0]))
        value = fedprox_term(
            [("encoder", encoder), ("classifier.weight", head)],
            {"encoder": torch.tensor([1.0]), "classifier.weight": torch.tensor([0.0])},
        )
        value.backward()
        torch.testing.assert_close(value, torch.tensor(1.0))
        self.assertIsNone(head.grad)


    def test_moon_keeps_reference_features_fixed(self):
        x = torch.randn(4, 8, requires_grad=True)
        g = torch.randn(4, 8, requires_grad=True)
        p = torch.randn(4, 8, requires_grad=True)
        moon_loss(x, g, p).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNone(g.grad)
        self.assertIsNone(p.grad)



if __name__ == "__main__":
    unittest.main()
