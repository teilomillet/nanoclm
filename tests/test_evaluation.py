"""Small, independent checks of the claims our experiment relies on."""

import copy
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

import evaluate
import nanoclm
from prepare_banking77 import download_sources, read_unique_rows


class EvaluationTests(unittest.TestCase):
    def test_metrics_against_hand_computed_confusion(self):
        # One mistake: class 0 -> 1. F1 per class is 2/3, 2/3, 1.
        result = evaluate.classification_metrics(torch.tensor([0, 1, 1, 2]),
                                                 torch.tensor([0, 0, 1, 2]), 3)
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], 7 / 9, places=6)

    def test_missing_predictions_receive_zero_f1(self):
        result = evaluate.classification_metrics(torch.tensor([0, 0]), torch.tensor([0, 1]), 2)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3, places=6)

    def test_paired_interval_zero_and_constant_improvement(self):
        config = evaluate.Config(bootstrap_samples=100)
        targets = torch.tensor([0, 0, 1, 1])
        for delta in (0.0, 1.0, -1.0):
            result = evaluate.paired_interval(torch.full((4,), delta), targets, config)
            self.assertEqual(result["ci95"], [delta, delta])
            self.assertEqual(result["accuracy_delta"], delta)

    def test_centroid_uses_training_labels(self):
        train = nanoclm.EncodedPairs(torch.tensor([[1., 0.], [1., 0.], [0., 1.]]),
                                    torch.eye(2), [[0, 1], [2]], torch.tensor([0, 0, 1]))
        self.assertTrue(torch.equal(evaluate.nearest_centroid(train), torch.eye(2)))

    def example_data(self):
        return {"candidates": ["a", "b"], **{
            split: [{"id": f"{split}-{action}", "state": f"query {split} {action}", "action": action}
                    for action in ("a", "b")]
            for split in ("train", "val", "test")}}

    def test_split_overlap_is_rejected(self):
        data = self.example_data()
        evaluate.validate_data(data)
        data["test"][0]["state"] = "  QUERY   TRAIN a  "
        with self.assertRaisesRegex(ValueError, "repeated"):
            evaluate.validate_data(data)

    def test_duplicate_ids_and_unknown_candidates_are_rejected(self):
        data = self.example_data()
        duplicate = copy.deepcopy(data)
        duplicate["test"][0]["id"] = data["train"][0]["id"]
        with self.assertRaisesRegex(ValueError, "repeated"):
            evaluate.validate_data(duplicate)
        data["test"][0]["action"] = "unknown"
        with self.assertRaisesRegex(ValueError, "candidate"):
            evaluate.validate_data(data)

    def test_source_deduplication_preserves_ids_and_train_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_text("text,category\nAlready in train,a\nNew query,b\n NEW  QUERY ,b\n")
            groups, skipped = read_unique_rows(path, "test", {"already in train"})
            self.assertEqual(skipped, 2)
            self.assertEqual(groups["b"][0]["id"], "test:1")

    def test_corrupted_source_is_rejected_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "train.csv").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                download_sources(path)

    def test_checkpoint_dataset_mismatch_is_rejected_before_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(self.example_data()))
            with patch("evaluate.torch.load", return_value={"data_sha256": "wrong", "training_config": {}}):
                with self.assertRaisesRegex(ValueError, "dataset mismatch"):
                    evaluate.evaluate(evaluate.Config(data_path=path))

    def test_batches_have_one_state_per_action(self):
        # State values encode their true action IDs, providing an external check.
        data = nanoclm.EncodedPairs(torch.tensor([[0], [0], [1], [2]]),
                                   torch.tensor([[0], [1], [2]]), [[0, 1], [2], [3]],
                                   torch.tensor([0, 0, 1, 2]))
        rng = random.Random(42)
        for _ in range(20):
            states, actions = nanoclm.sample_batch(data, rng, nanoclm.Config(batch_size=3))
            self.assertTrue(torch.equal(states, actions))
            self.assertEqual(len(actions.unique()), 3)

    def test_validation_checks_all_candidates_with_repeated_targets(self):
        class FixedScores(torch.nn.Module):
            def forward(self, states, actions):
                return torch.tensor([[4., 0., 0.], [0., 0., 4.], [0., 4., 0.]])
        data = nanoclm.EncodedPairs(torch.zeros(3, 2), torch.zeros(3, 2),
                                   [[0, 1], [2], []], torch.tensor([0, 0, 1]))
        model = FixedScores().train()
        loss, accuracy = nanoclm.evaluate(model, data)
        self.assertAlmostEqual(accuracy, 2 / 3, places=6)
        self.assertGreater(loss, 1)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
