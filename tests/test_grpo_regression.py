import pytest
import torch

from verifiers.rl.trainer.config import RLConfig
from verifiers.rl.trainer.trainer import RLTrainer


requires_cuda = pytest.mark.skipif(False, reason="(unused)")


class _DummyTrainer(RLTrainer):
    def __init__(self, args: RLConfig):
        self.args = args
        # mirror fields set in RLTrainer.__init__ used by compute_loss
        self.mask_ratio_low = args.mask_ratio_low
        self.mask_ratio_high = args.mask_ratio_high


@pytest.mark.parametrize("device_str", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_grpo_compute_loss_consistency(device_str):
    device = torch.device(device_str)
    args = RLConfig(output_dir="outputs/test", run_name="grpo-loss", rl_algo="grpo")
    trainer = _DummyTrainer(args)

    B, L = 2, 5
    loss_mask = torch.tensor(
        [[1, 1, 0, 1, 1], [1, 0, 0, 1, 1]], dtype=torch.bool, device=device
    )
    trainer_logprobs = torch.randn(B, L, device=device)
    inference_logprobs = torch.randn(B, L, device=device)
    advantages = torch.randn(B, L, device=device)
    entropies = torch.rand(B, L, device=device)

    inputs = {
        "loss_mask": loss_mask,
        "trainer_logprobs": trainer_logprobs,
        "inference_logprobs": inference_logprobs,
        "advantages": advantages,
        "entropies": entropies,
    }

    loss_ref, _ = trainer.compute_loss(None, inputs, return_outputs=True)

    with torch.no_grad():
        log_ir = trainer_logprobs - inference_logprobs
        ir = torch.exp(log_ir)
        is_low = ir < args.mask_ratio_low
        is_high = ir > args.mask_ratio_high
        keep = (~is_low & ~is_high & loss_mask)
        manual = (-(ir * advantages))[keep].sum()

    assert torch.allclose(loss_ref, manual, atol=1e-6)
