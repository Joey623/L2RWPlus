import unittest

import numpy as np
import torch
from l2rwplus.engine.federation import (
    private_relabel, mutual_nn_groups, umrb_broadcast, ufed_head_rows,
    fed_avg_sd, strip_classifier,
)


class FederationTests(unittest.TestCase):
    def test_private_labels_are_fixed_local_bijections(self):
        labels = np.repeat(np.arange(50), 2)
        a, mapping = private_relabel(labels, 1, 1)
        b, _ = private_relabel(labels, 2, 1)
        again, same_mapping = private_relabel(labels, 1, 1)
        np.testing.assert_array_equal(a, again)
        self.assertEqual(mapping, same_mapping)
        self.assertEqual(set(mapping.values()), set(range(50)))
        self.assertTrue(np.all(a[::2] == a[1::2]))
        self.assertFalse(np.array_equal(a, b))


    def test_encoder_aggregation_excludes_different_local_heads(self):
        a = {"encoder": torch.tensor([1.0, 3.0]), "classifier.weight": torch.ones(2, 2)}
        b = {"encoder": torch.tensor([5.0, 7.0]), "classifier.weight": torch.ones(3, 2)}
        out = fed_avg_sd([a, b], [1, 3], skip_prefixes=("classifier.",))
        torch.testing.assert_close(out["encoder"], torch.tensor([4.0, 6.0]))
        self.assertNotIn("classifier.weight", out)
        self.assertEqual(list(strip_classifier(a)), ["encoder"])


    def test_correspondence_ignores_label_numbers_and_zero_prototypes(self):
        p = [
            {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0]), 2: torch.zeros(2)},
            {0: torch.tensor([0.0, 1.0]), 1: torch.tensor([1.0, 0.0])},
        ]
        groups, _ = mutual_nn_groups(p, tau=0.9)
        self.assertEqual(groups, [[(0, 0), (1, 1)], [(0, 1), (1, 0)]])
        q = umrb_broadcast(p, groups)
        torch.testing.assert_close(q[0][2], p[0][2])
        for group in groups:
            self.assertEqual(len(group), len({client for client, _ in group}))


    def test_transitive_groups_keep_at_most_one_identity_per_client(self):
        gen = torch.Generator().manual_seed(3)
        p = [{i: torch.randn(4, generator=gen) for i in range(8)} for _ in range(4)]
        groups, _ = mutual_nn_groups(p, tau=-1)
        members = []
        for group in groups:
            self.assertEqual(len(group), len({c for c, _ in group}))
            members.extend(group)
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(len(members), 32)


    def test_classifier_updates_only_matched_rows(self):
        a = torch.tensor([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
        b = torch.tensor([[3.0, 4.0], [30.0, 40.0]])
        original_a = a.clone()
        changes = ufed_head_rows([a, b], [[(0, 1), (1, 0)], [(0, 0)]])
        self.assertEqual(set(changes[0]), {1})
        self.assertEqual(set(changes[1]), {0})
        torch.testing.assert_close(changes[0][1], torch.tensor([6.5, 12.0]))
        torch.testing.assert_close(a, original_a)



if __name__ == "__main__":
    unittest.main()
