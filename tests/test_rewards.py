import sys, unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from common.protocol import (
    extract_answer,
    correctness_reward,
    format_reward,
    answer_matches,
)
from corver.rewards.sentence import (
    sentence_corver_penalty,
    compute_sentence_corver_penalties,
)
from corver.rewards.batch import CachedLocalCounts, fast_batch_returns
from corver.training.trainer import token_advantages
from corver.rewards.alignment import decoded_token_spans, map_generated_sentences


class CharTokenizer:
    all_special_ids = [0]
    is_fast = False

    def decode(self, ids, **kw):
        return "".join(chr(int(x)) for x in ids if x)


class Rewards(unittest.TestCase):
    def test_count_boundaries(self):
        for count, expected in [
            (None, 0),
            (0, -0.2),
            (1, -0.1),
            (4, -0.1),
            (5, 0),
            (19, 0),
            (20, 0.1),
            (999, 0.1),
        ]:
            with self.subTest(count=count):
                self.assertEqual(sentence_corver_penalty(count), expected)

    def test_first_valid_and_no_query_failure_fallback(self):
        extractor = SimpleNamespace(
            extract_triplets_batch=lambda _: [
                [
                    ["he", "r", "Paris"],
                    ["Paris", "r", "Paris"],
                    ["Paris", "r", "France"],
                    ["Rome", "r", "Italy"],
                ]
            ]
        )

        class Client:
            def count_batch(self, qs):
                self.qs = qs
                return [(None, None)]

        client = Client()
        p, d = compute_sentence_corver_penalties(["text"], extractor, client)
        self.assertEqual(p, [0])
        self.assertEqual(client.qs, ["Paris AND France"])
        self.assertEqual(d[0]["triplet"], ["Paris", "r", "France"])

    def test_dropped_rows_fail(self):
        with self.assertRaises(ValueError):
            compute_sentence_corver_penalties(
                ["x"], SimpleNamespace(extract_triplets_batch=lambda _: []), None
            )
        with self.assertRaises(ValueError):
            compute_sentence_corver_penalties(
                ["x"],
                SimpleNamespace(
                    extract_triplets_batch=lambda _: [[["Paris", "r", "France"]]]
                ),
                SimpleNamespace(count_batch=lambda _: []),
            )

    def test_answer_boundaries(self):
        self.assertEqual(
            extract_answer(
                "<think><answer>wrong</answer><think>x</think></think><answer>Paris</answer>"
            ),
            ("Paris", "bounded"),
        )
        self.assertEqual(
            extract_answer("</think><answer>Paris\nmore"), ("Paris", "fallback")
        )
        self.assertEqual(extract_answer("Paris"), ("", "missing"))
        self.assertEqual(
            correctness_reward("<answer>The Paris!</answer>", ["Paris"]), 2
        )
        for answer in ["", "I don't know", "I do not know", "I have no comment"]:
            self.assertEqual(
                correctness_reward(f"<answer>{answer}</answer>", [answer, "Paris"]), -1
            )
        self.assertTrue(
            answer_matches("York", ["New York"])
        )  # Bidirectional substring match.

    def test_format_length(self):
        self.assertEqual(format_reward("<answer>Paris</answer>", 512), 1)
        self.assertEqual(format_reward("<answer>Paris</answer>", 1024), 0.5)
        self.assertEqual(format_reward("<answer>Paris", 1), -1)

    def test_small_variance_is_not_token_normalization(self):
        returns = torch.tensor([[0.1, -0.1, 999], [0.2, -0.2, 999]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
        adv, means, std = token_advantages(returns, mask, 2)
        torch.testing.assert_close(
            adv, torch.tensor([[1000.0, -1000.0, 0], [2000.0, -2000.0, 0]])
        )
        self.assertEqual(std.sum(), 0)
        self.assertEqual(means.sum(), 0)

    def test_sample_std(self):
        ret = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
        adv, means, std = token_advantages(ret, torch.ones_like(ret), 2)
        torch.testing.assert_close(means, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(std, torch.tensor([2**0.5, 2**0.5]))
        torch.testing.assert_close(adv, (ret - 3) / (2**0.5 + 1e-4))

    def test_cache_zero_failure_and_limits(self):
        class Client:
            max_clause_freq = 500000
            max_diff_tokens = 1000
            calls = 0

            def count(self, q):
                self.calls += 1
                return (None if q == "failure" else 0), 0.1

        c = Client()
        cache = CachedLocalCounts(c)
        cache.count("zero")
        cache.count("zero")
        self.assertEqual(c.calls, 1)
        cache.count("failure")
        cache.count("failure")
        self.assertEqual(c.calls, 3)
        c.max_diff_tokens = 10
        cache.count("zero")
        self.assertEqual(c.calls, 4)

    def test_sentence_rewards_and_padding(self):
        text = "<think>Paris is in France.</think><answer>Paris</answer>"
        ids = torch.tensor([[ord(x) for x in text] + [0, 0]])
        mask = (ids != 0).long()
        ext = SimpleNamespace(
            extract_triplets_batch=lambda ss: [[["Paris", "in", "France"]] for _ in ss]
        )
        client = SimpleNamespace(count_batch=lambda qs: [(0, 0.0) for _ in qs])
        values, info = fast_batch_returns(
            ids, mask, CharTokenizer(), ext, client, [2], [1], 2
        )
        self.assertAlmostEqual(values[0, text.index("Paris")].item(), 4.8, places=5)
        self.assertEqual(values[0, 0].item(), 5)
        self.assertEqual(values[0, -1].item(), 0)
        self.assertEqual(info[0]["num_sentences"], 2)

    def test_noncanonical_token_decode(self):
        class FragmentTokenizer:
            all_special_ids = []
            is_fast = False

            def decode(self, ids, **kw):
                return {0: "", 1: "�", 2: "é", 3: "é."}[len(ids)]

        text, spans, mode = decoded_token_spans(FragmentTokenizer(), [1, 2, 3])
        self.assertEqual(text, "é.")
        self.assertEqual(spans, [(0, 1), (0, 1), (1, 2)])


if __name__ == "__main__":
    unittest.main()
