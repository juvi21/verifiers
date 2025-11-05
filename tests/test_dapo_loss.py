import pytest
import torch

from verifiers.rl.trainer.config import RLConfig
from verifiers.rl.trainer.trainer import RLTrainer


def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


class _DummyDapoTrainer(RLTrainer):
    def __init__(self, args: RLConfig):
        # Minimal init: only fields used by _compute_dapo_loss
        self.args = args


@pytest.mark.parametrize("device_str", available_devices())
def test_dapo_compute_loss_runs_on_cpu_and_gpu(device_str):
    device = torch.device(device_str)
    cfg = RLConfig(
        output_dir="outputs/test",
        run_name="dapo-loss",
        rl_algo="dapo",
        dapo={
            "eps_low": 0.2,
            "eps_high": 0.3,
        },
    )
    trainer = _DummyDapoTrainer(cfg)

    B, L = 2, 7
    inputs = {
        "loss_mask": torch.tensor(
            [[1, 1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1, 1]], dtype=torch.bool, device=device
        ),
        "entropies": torch.rand(B, L, device=device),
        "trainer_logprobs": torch.randn(B, L, device=device),
        "inference_logprobs": torch.randn(B, L, device=device),
        "advantages": torch.randn(B, L, device=device),
    }

    loss, summaries = trainer._compute_dapo_loss(inputs, inputs["loss_mask"])  # type: ignore[attr-defined]
    # Basic sanity: finite loss and presence of metrics
    assert torch.isfinite(loss)
    assert "importance_sampling" in summaries
    assert "clip_fraction" in summaries

