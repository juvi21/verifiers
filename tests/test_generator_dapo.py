import asyncio
from typing import Any

import numpy as np
import pytest
from datasets import Dataset

from verifiers.rl.trainer.generator import Generator
from verifiers.types import ProcessedOutputs


class _FakeProcessing:
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True, tools=None):
        if isinstance(conversation, list):
            return " ".join(str(x) for x in conversation)
        return str(conversation)

    def encode(self, text):
        return list(range(len(str(text))))


class _DummyEnv:
    def __init__(self, n_prompts: int):
        self.dataset = Dataset.from_dict({"prompt": [f"q{i}" for i in range(n_prompts)]})

    async def a_generate(self, *args, **kwargs):
        class R:
            pass
        r = R()
        # Values are ignored because we monkeypatch process_env_results_vllm
        r.prompt = []
        r.completion = []
        r.state = []
        r.reward = []
        r.metrics = {}
        return r


def _po(N: int, prompt_len: int, completion_len: int, rewards: list[float]):
    prompts = [[1] * prompt_len for _ in range(N)]
    p_mask = [[0] * prompt_len for _ in range(N)]
    comps = [[2] * completion_len for _ in range(N)]
    c_mask = [[1] * completion_len for _ in range(N)]
    c_logp = [[-1.0] * completion_len for _ in range(N)]
    return ProcessedOutputs(
        prompt_ids=prompts,
        prompt_mask=p_mask,
        completion_ids=comps,
        completion_mask=c_mask,
        completion_logprobs=c_logp,
        rewards=rewards,
        is_truncated=[False] * N,
    )


@pytest.mark.asyncio
async def test_generate_batch_grpo_microbatches(monkeypatch):
    # Two prompts, 2 rollouts each => N=4
    env = _DummyEnv(n_prompts=2)
    gen = Generator(
        env=env,
        client_base_url="http://localhost:8000/v1",
        client_api_key="local",
        client_limit=8,
        client_timeout=30.0,
        model_name="dummy",
        sampling_args={"n": 1},
        rollouts_per_example=2,
        batch_size=4,
        micro_batch_size=2,
        num_processes=1,
        generation_timeout=10.0,
        processing_class=_FakeProcessing(),
        mask_env_responses=True,
        max_seq_len=512,
        max_prompt_len=128,
        mask_truncated_completions=False,
        zero_truncated_completions=False,
        max_concurrent=64,
        rl_algo="grpo",
    )

    rewards = [1.0, 10.0, 3.0, 10.0]  # interleaved groups: [1,3], [10,10]
    po = _po(N=4, prompt_len=3, completion_len=5, rewards=rewards)
    monkeypatch.setattr(env, "process_env_results_vllm", lambda *a, **k: po)

    batch = await gen.generate_batch(0)
    assert batch.global_item_count == 4 * 5  # all completion tokens counted
    # advantages in GRPO = r - mean(group)
    # group0: [1,3] mean=2 -> [-1, +1]; group1: [10,10] mean=10 -> [0,0]
    # advantages are broadcast across tokens; we check first microbatch's advantages are constant per sample
    mb = batch.microbatches[0][0]
    # first two samples belong to group 0 (indices 0 and 2)
    a0 = mb.advantages[0]
    a1 = mb.advantages[1]
    assert np.isclose(a0[0], -1.0)
    assert np.isclose(a1[0], +1.0)


@pytest.mark.asyncio
async def test_generate_batch_dapo_dynamic_sampling(monkeypatch):
    # Two prompts, 2 rollouts each => N=4, with one one-sided group
    env = _DummyEnv(n_prompts=2)
    gen = Generator(
        env=env,
        client_base_url="http://localhost:8000/v1",
        client_api_key="local",
        client_limit=8,
        client_timeout=30.0,
        model_name="dummy",
        sampling_args={"n": 1},
        rollouts_per_example=2,
        batch_size=4,
        micro_batch_size=2,
        num_processes=1,
        generation_timeout=10.0,
        processing_class=_FakeProcessing(),
        mask_env_responses=True,
        max_seq_len=512,
        max_prompt_len=128,
        mask_truncated_completions=False,
        zero_truncated_completions=False,
        max_concurrent=64,
        rl_algo="dapo",
    )

    # group0 rewards [0,1] (mixed), group1 rewards [10,10] (one-sided -> likely skipped)
    rewards = [0.0, 10.0, 1.0, 10.0]
    po = _po(N=4, prompt_len=2, completion_len=4, rewards=rewards)
    monkeypatch.setattr(env, "process_env_results_vllm", lambda *a, **k: po)

    batch = await gen.generate_batch(0)
    # Expect some samples skipped due to one-sided group; global_item_count < 4*4
    assert batch.global_item_count < 16
    # Metrics include DAPO selection keys
    assert "dapo/skip_one_sided" in batch.metrics_dict
    assert batch.metrics_dict["dapo/prompts_total"] == 2.0

