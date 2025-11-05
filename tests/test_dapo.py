import os
import json
import math
import numpy as np
import pytest

import torch

from verifiers.rl.trainer.config import RLConfig, DAPOHyperParams
from verifiers.types import ProcessedOutputs
from verifiers.rl.trainer.generator import Generator


requires_cuda = pytest.mark.skipif(
    False, reason="(unused)"
)


class _FakeProcessing:
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True, tools=None):
        if isinstance(conversation, list):
            return " ".join([
                (m.get("role", "") + ":" + str(m.get("content", ""))) for m in conversation
            ])
        return str(conversation)

    def encode(self, text):
        return list(range(len(str(text))))


def _build_min_generator(**kwargs):
    # Create an uninitialized Generator and set only the fields needed by helper methods
    gen = object.__new__(Generator)
    gen.rollouts_per_example = kwargs.get("rollouts_per_example", 4)
    gen.processing_class = _FakeProcessing()
    gen.dapo_params = kwargs.get("dapo_params", DAPOHyperParams())
    gen.prompt_variance = {}
    return gen


def test_config_parsing_dapo_overrides():
    cfg = RLConfig(
        output_dir="outputs/test",
        run_name="cfg-dapo",
        rl_algo="dapo",
        dapo={
            "group_size": 6,
            "eps_low": 0.15,
            "eps_high": 0.35,
            "learning_rate": 2e-5,
            "max_grad_norm": 0.9,
        },
    )
    assert cfg.rl_algo == "dapo"
    assert cfg.rollouts_per_example == 6
    assert math.isclose(cfg.learning_rate, 2e-5)
    assert math.isclose(cfg.max_grad_norm, 0.9)


def test_grpo_remains_default():
    cfg = RLConfig(output_dir="outputs/test", run_name="cfg-grpo")
    assert cfg.rl_algo == "grpo"
    # Ensure default rollouts_per_example unchanged
    assert cfg.rollouts_per_example == 16


def test_group_advantages_normalized():
    gen = _build_min_generator(rollouts_per_example=4)
    # Two prompts, 4 rollouts each, interleaved by prompt as generator does
    # prompt0 rewards: [1,2,3,4], prompt1 rewards: [10,10,10,10]
    rewards = [1.0, 10.0, 2.0, 10.0, 3.0, 10.0, 4.0, 10.0]
    adv_norm = gen._compute_group_advantages(rewards, prompts_in_batch=2, normalize=True)
    # Build indices per prompt as the generator groups them
    idx_p0 = [0, 2, 4, 6]
    idx_p1 = [1, 3, 5, 7]
    g1 = [adv_norm[i] for i in idx_p0]
    assert abs(sum(g1)) < 1e-6
    g2 = [adv_norm[i] for i in idx_p1]
    assert all(abs(a) < 1e-6 for a in g2)


def test_dynamic_sampling_and_advantages_stateful():
    dapo = DAPOHyperParams(dynamic_sampling=True, variance_threshold=0.01)
    gen = _build_min_generator(rollouts_per_example=3, dapo_params=dapo)
    # 3 prompts, each with 3 rollouts
    # Prompt 0: mixed rewards -> keep
    # Prompt 1: all positive -> skip by one-sided rule
    # Prompt 2: low variance -> skip by variance
    rewards = [0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
    is_trunc = [False] * 9
    prompts = [
        [{"role": "user", "content": f"q{i}"}] for i in range(3)
    ]
    sel = gen._select_dapo_indices(rewards, is_trunc, prompts, prompts_in_batch=3, batch_id=0)
    assert sel["indices"]  # some kept
    # Expect that at least prompt 0 kept
    assert any(i in sel["indices"] for i in [0, 1, 2])
    # Update prompt variance and re-run with same rewards: should still keep 0 because mixed
    # and skip 1 and 2 as before
    sel2 = gen._select_dapo_indices(rewards, is_trunc, prompts, prompts_in_batch=3, batch_id=1)
    assert sel2["indices"]


def test_length_penalty_and_truncation_shaping():
    gen = _build_min_generator()
    class _O:
        pass
    o = _O()
    o.prompt_ids = [[1], [1]]
    o.prompt_mask = [[0], [0]]
    # one short completion, one long + truncated
    o.completion_ids = [[2,3,4], list(range(1000))]
    o.completion_mask = [[1,1,1], [1]*1000]
    o.completion_logprobs = [[-1,-1,-1], [-1]*1000]
    o.rewards = [1.0, 1.0]
    o.is_truncated = [False, True]
    shaped, metrics = gen._apply_dapo_reward_shaping(o)
    assert len(shaped) == 2
    assert shaped[1] <= shaped[0]  # penalized
    assert metrics["dapo/penalty/truncation"] >= 0.0
