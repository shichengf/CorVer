import sys, unittest
from pathlib import Path

import torch
from corver.training.loss import grpo_logprob_loss, align_generation_metadata
from corver.training.loss import selected_completion_logps, pack_padding
from types import SimpleNamespace


class TestStepwiseLoss(unittest.TestCase):
    def test_prompt_padding_and_microbatch_logps(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Parameter(
                    torch.arange(49).reshape(7, 7).float() / 17
                )

            def forward(self, input_ids, attention_mask, logits_to_keep, **kwargs):
                return SimpleNamespace(
                    logits=self.table[input_ids][:, -logits_to_keep:]
                )

        model = TinyModel()
        ids = torch.tensor([[0, 0, 1, 2, 3, 4, 0], [0, 5, 6, 2, 1, 3, 4]])
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1]])
        packed = pack_padding(ids, 0)
        actual = selected_completion_logps(model, packed, mask, 3, 2)
        split = selected_completion_logps(model, packed, mask, 3, 1)
        direct = (
            model.table[ids[:, 3:6]]
            .log_softmax(-1)
            .gather(-1, ids[:, 4:, None])
            .squeeze(-1)
        )
        direct = direct * mask[:, 4:]
        torch.testing.assert_close(actual, direct)
        torch.testing.assert_close(split, direct)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(model.table.grad).all())

    def test_repeated_sampler_alignment(self):
        a = {"question": "a", "best_answer": "A"}
        b = {"question": "b", "best_answer": "B"}
        self.assertEqual(
            align_generation_metadata([a, a, b, b], 4, 2),
            (["a", "a", "b", "b"], ["A", "A", "B", "B"]),
        )
        self.assertEqual(
            align_generation_metadata([a, b], 4, 2),
            (["a", "a", "b", "b"], ["A", "A", "B", "B"]),
        )
        with self.assertRaises(ValueError):
            align_generation_metadata([a], 3, 2)

    def test_on_policy_gradient_and_mask(self):
        new = torch.tensor([[-2.0, -3.0, -999.0]], requires_grad=True)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        adv = torch.tensor([[2.0, -1.0, 100.0]])
        loss, length, kl, *_ = grpo_logprob_loss(
            new.detach(),
            new,
            new.detach(),
            None,
            None,
            mask,
            0.01,
            adv,
            current_gradient_accumulation_steps=2,
        )
        self.assertAlmostEqual(loss.item(), -0.25)
        self.assertEqual(length.item(), 2)
        self.assertEqual(kl.item(), 0)
        loss.backward()
        torch.testing.assert_close(new.grad, torch.tensor([[-0.5, 0.25, 0.0]]))

    def test_clip_stops_wrong_direction_update(self):
        new = torch.tensor([[0.5, -0.5]], requires_grad=True)
        loss, *_ = grpo_logprob_loss(
            None,
            new,
            torch.zeros_like(new),
            None,
            None,
            torch.ones_like(new),
            0,
            torch.tensor([[1.0, -1.0]]),
        )
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        loss.backward()
        torch.testing.assert_close(new.grad, torch.zeros_like(new))

    def test_per_token_advantages_match_dense_reference(self):
        torch.manual_seed(4)
        logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
        ref_logits = torch.randn(2, 3, 5, dtype=torch.float64)
        ids = torch.tensor([[0, 2, 3], [1, 4, 0]])
        new = logits.log_softmax(-1).gather(-1, ids[..., None]).squeeze(-1)
        ref = ref_logits.log_softmax(-1).gather(-1, ids[..., None]).squeeze(-1)
        adv = torch.tensor([[0.2, -0.5, 0.7], [-0.4, 0.3, 0.2]])
        mask = torch.ones_like(new)
        loss, *_ = grpo_logprob_loss(ref, new, new.detach(), None, ids, mask, 0.01, adv)
        expected = (
            -torch.exp(new - new.detach()) * adv
            + 0.01 * (torch.exp(ref - new) - (ref - new) - 1)
        ).mean()
        torch.testing.assert_close(loss.double(), expected, atol=1e-7, rtol=1e-6)
        (actual_grad,) = torch.autograd.grad(loss, logits, retain_graph=True)
        (expected_grad,) = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
