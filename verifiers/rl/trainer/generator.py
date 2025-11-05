import asyncio
import logging
import queue
import threading
import time
from typing import Any, Dict, List

import httpx
import numpy as np
from datasets import Dataset
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from verifiers import Environment
from verifiers.rl.trainer.config import DAPOHyperParams


class Microbatch(BaseModel):
    """Microbatch for batch generation"""

    input_ids: list[list[int]]
    loss_mask: list[list[int]]
    sampling_logprobs: list[list[float]]
    advantages: list[list[float]]
    items: int


class Batch(BaseModel):
    """Result from batch generation"""

    batch_id: int
    microbatches: list[list[Microbatch]]
    items_per_process: list[int]
    global_item_count: int
    # logging
    generation_time: float = 0.0
    prompts: list[Any] = Field(default_factory=list)
    completions: list[Any] = Field(default_factory=list)
    metrics_dict: dict[str, float] = Field(default_factory=dict)
    rewards_dict: dict[str, list[float]] = Field(default_factory=dict)


class Generator:
    """
    Manages asynchronous batch generation in parallel with RL training.
    """

    def __init__(
        self,
        env: Environment,
        client_base_url: str,
        client_api_key: str,
        client_limit: int,
        client_timeout: float,
        model_name: str,
        sampling_args: dict[str, Any],
        rollouts_per_example: int,
        batch_size: int,
        micro_batch_size: int,
        num_processes: int,
        generation_timeout: float,
        processing_class: PreTrainedTokenizerBase,
        mask_env_responses: bool,
        max_seq_len: int,
        max_prompt_len: int,
        mask_truncated_completions: bool,
        zero_truncated_completions: bool,
        max_concurrent: int,
        rl_algo: str = "grpo",
        dapo_params: DAPOHyperParams | None = None,
    ):
        self.env = env
        self.client_base_url = client_base_url
        self.client_api_key = client_api_key
        self.client_limit = client_limit
        self.client_timeout = client_timeout
        self.client = None  # created in worker thread
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.rollouts_per_example = rollouts_per_example
        self.prompts_per_batch = batch_size // rollouts_per_example
        self.micro_batch_size = micro_batch_size
        self.num_processes = num_processes
        self.generation_timeout = generation_timeout
        self.processing_class = processing_class
        self.mask_env_responses = mask_env_responses
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.mask_truncated_completions = mask_truncated_completions
        self.zero_truncated_completions = zero_truncated_completions
        self.max_concurrent = max_concurrent
        self.rl_algo = rl_algo.lower()
        self.dapo_params = dapo_params or DAPOHyperParams()
        self.prompt_variance: Dict[str, float] = {}

        # queues for communication
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_generating = False
        self.completed_batches = {}

        self.worker_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        self.is_generating = False
        self.worker_loop = None

        max_length = self.max_prompt_len
        assert env.dataset is not None

        def filter_by_prompt_length(example, processing_class):
            prompt = example["prompt"]
            if isinstance(prompt, list):
                prompt_text = processing_class.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = prompt
            prompt_ids = processing_class.encode(prompt_text)
            return len(prompt_ids) <= max_length

        env.dataset = env.dataset.filter(
            filter_by_prompt_length,
            fn_kwargs={"processing_class": processing_class},
        )

    def get_dataset_slice(self, batch_id: int) -> Dataset:
        """Get dataset slice for a given batch id"""
        num_rows = self.prompts_per_batch
        dataset = self.env.get_dataset()
        total_rows = len(dataset)
        if total_rows == 0:
            raise ValueError("Environment dataset is empty")
        offset = (batch_id * num_rows) % total_rows
        indices = [(offset + i) % total_rows for i in range(num_rows)]
        return dataset.select(indices)

    def start(self):
        """Start the async generation worker thread"""
        self.worker_thread = threading.Thread(
            target=self.generation_worker, daemon=True, name="BatchGenerator"
        )
        self.worker_thread.start()

    def stop(self):
        """Stop the async generation worker thread"""
        self.stop_event.set()
        self.request_queue.put(None)  # poison pill
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

    def submit_batch(self, batch_id: int):
        self.request_queue.put(batch_id)

    def get_batch(self, batch_id: int) -> Batch:
        """
        Get a completed batch result. Blocks until the batch is ready.

        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum time to wait

        Returns:
            BatchResult: The completed batch result

        Raises:
            TimeoutError: batch doesn't complete within timeout
            RuntimeError: generation failed
        """
        timeout = self.generation_timeout
        start_time = time.time()
        while True:
            if batch_id in self.completed_batches:
                return self.completed_batches.pop(batch_id)
            try:
                result = self.result_queue.get(timeout=0.1)
                self.completed_batches[result.batch_id] = result
                if result.batch_id == batch_id:
                    return self.completed_batches.pop(batch_id)
            except queue.Empty:
                pass

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch {batch_id} timed out after {timeout}s")

    def generation_worker(self):
        """Worker thread that processes generation requests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop
        self.client = AsyncOpenAI(
            base_url=self.client_base_url,
            api_key=self.client_api_key,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.client_limit),
                timeout=self.client_timeout,
            ),
        )
        try:
            while not self.stop_event.is_set():
                try:
                    batch_id = self.request_queue.get(timeout=0.1)
                    if batch_id is None:  # poison pill
                        break
                    result = loop.run_until_complete(self.generate_batch(batch_id))
                    self.result_queue.put(result)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in generation worker: {e}")
                    raise e
        finally:
            loop.run_until_complete(self.client.close())
            loop.close()
            asyncio.set_event_loop(None)

    async def generate_batch(self, batch_id: int) -> Batch:
        """
        Generate a single batch asynchronously.
        """
        self.is_generating = True
        assert self.client is not None
        start_time = time.time()
        batch_ds = self.get_dataset_slice(batch_id)
        repeated_ds = batch_ds.repeat(self.rollouts_per_example)
        env_results = await self.env.a_generate(
            repeated_ds,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
            score_rollouts=True,
            max_concurrent=self.max_concurrent,
        )
        self.is_generating = False
        wall_clock_s = time.time() - start_time

        processed_results = self.env.process_env_results_vllm(
            prompts=env_results.prompt,
            completions=env_results.completion,
            states=env_results.state,
            rewards=env_results.reward,
            processing_class=self.processing_class,
            max_seq_len=self.max_seq_len,
            mask_env_responses=self.mask_env_responses,
            mask_truncated_completions=self.mask_truncated_completions,
            zero_truncated_completions=self.zero_truncated_completions,
        )

        dapo_metrics: dict[str, float] = {}
        if self.rl_algo == "dapo":
            rewards, shaping_metrics = self._apply_dapo_reward_shaping(
                processed_results
            )
            processed_results.rewards = rewards
            dapo_metrics.update(shaping_metrics)
            selection = self._select_dapo_indices(
                rewards,
                processed_results.is_truncated,
                env_results.prompt,
                len(batch_ds),
                batch_id,
            )
            if selection["indices"]:
                filtered_rewards = self._filter_processed_outputs(
                    processed_results,
                    env_results,
                    rewards,
                    selection["indices"],
                )
                processed_results.rewards = filtered_rewards
                rewards = filtered_rewards
                advantages = selection["advantages"]
            else:
                advantages = self._compute_group_advantages(
                    processed_results.rewards,
                    len(batch_ds),
                    normalize=self.dapo_params.normalize_advantages,
                )
            dapo_metrics.update(selection["metrics"])
        else:
            rewards = processed_results.rewards
            advantages = self._compute_group_advantages(
                rewards, len(batch_ds), normalize=False
            )

        rewards_dict = {"reward": processed_results.rewards}
        for k in env_results.metrics:
            rewards_dict[k] = env_results.metrics[k]

        metrics_dict = {}
        if rewards:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            metrics_dict["reward"] = float(rewards_arr.mean())
            metrics_dict["reward/std"] = float(rewards_arr.std())

        if advantages:
            adv_arr = np.asarray(advantages, dtype=np.float32)
            metrics_dict["advantage/absmean"] = float(np.abs(adv_arr).mean())

        for reward_name, values in env_results.metrics.items():
            if len(values) == 0:
                continue
            reward_values = np.asarray(values, dtype=np.float32)
            metrics_dict[f"reward/{reward_name}"] = float(reward_values.mean())

        completion_lengths = [len(ids) for ids in processed_results.completion_ids]
        if completion_lengths:
            completion_lengths_arr = np.asarray(completion_lengths, dtype=np.float32)
            metrics_dict["tokens/completion"] = float(completion_lengths_arr.mean())

            completion_mask_lengths = np.asarray(
                [sum(mask) for mask in processed_results.completion_mask],
                dtype=np.float32,
            )
            valid_tokens = completion_mask_lengths.sum()
            total_tokens = completion_lengths_arr.sum()
            if total_tokens > 0:
                masked_fraction = 1.0 - (valid_tokens / total_tokens)
                metrics_dict["tokens/masked_fraction"] = float(masked_fraction)

        generation_ms: list[float] = []
        scoring_ms: list[float] = []
        total_ms: list[float] = []
        for state in env_results.state:
            timing = state.get("timing", {})
            if "generation_ms" in timing:
                generation_ms.append(float(timing["generation_ms"]))
            if "scoring_ms" in timing:
                scoring_ms.append(float(timing["scoring_ms"]))
            if "total_ms" in timing:
                total_ms.append(float(timing["total_ms"]))

        if generation_ms:
            metrics_dict["timing/generation_ms"] = float(np.mean(generation_ms))
        if scoring_ms:
            metrics_dict["timing/scoring_ms"] = float(np.mean(scoring_ms))
        if total_ms:
            metrics_dict["timing/total_ms"] = float(np.mean(total_ms))

        metrics_dict["wall_clock/generate_s"] = float(wall_clock_s)
        if dapo_metrics:
            metrics_dict.update(dapo_metrics)

        # build per-process microbatches
        N = len(processed_results.rewards)
        per_proc_indices = np.array_split(np.arange(N), self.num_processes)
        microbatches: list[list[Microbatch]] = []
        items_per_process: list[int] = []
        for proc_indices in per_proc_indices:
            proc_list = proc_indices.tolist()
            proc_mbs: list[Microbatch] = []
            proc_item_total = 0
            if not proc_list:
                microbatches.append(proc_mbs)
                items_per_process.append(0)
                continue
            for cursor in range(0, len(proc_list), self.micro_batch_size):
                batch_indices = proc_list[cursor : cursor + self.micro_batch_size]
                ids_chunk = [
                    processed_results.prompt_ids[i]
                    + processed_results.completion_ids[i]
                    for i in batch_indices
                ]
                mask_chunk = [
                    processed_results.prompt_mask[i]
                    + processed_results.completion_mask[i]
                    for i in batch_indices
                ]
                slogp_chunk = [
                    [0.0] * len(processed_results.prompt_mask[i])
                    + processed_results.completion_logprobs[i]
                    for i in batch_indices
                ]
                lengths = [len(mask) for mask in mask_chunk]
                adv_chunk = [
                    [advantages[i]] * lengths[idx]
                    for idx, i in enumerate(batch_indices)
                ]
                mb_items = sum(sum(mask) for mask in mask_chunk)
                microbatch = Microbatch(
                    input_ids=ids_chunk,
                    loss_mask=mask_chunk,
                    sampling_logprobs=slogp_chunk,
                    advantages=adv_chunk,
                    items=mb_items,
                )
                proc_item_total += mb_items
                proc_mbs.append(microbatch)
            microbatches.append(proc_mbs)
            items_per_process.append(proc_item_total)

        global_item_count = sum(items_per_process)

        return Batch(
            batch_id=batch_id,
            microbatches=microbatches,
            items_per_process=items_per_process,
            global_item_count=global_item_count,
            generation_time=wall_clock_s,
            rewards_dict=rewards_dict,
            completions=env_results.completion,
            prompts=env_results.prompt,
            metrics_dict=metrics_dict,
        )

    def _apply_dapo_reward_shaping(self, outputs) -> tuple[list[float], dict[str, float]]:
        rewards = []
        length_penalties = []
        trunc_penalties = []
        target = max(1, self.dapo_params.length_penalty_target)
        cache = self.dapo_params.length_cache
        max_penalty = self.dapo_params.length_penalty_max
        trunc_penalty = self.dapo_params.truncation_penalty
        for reward, completion_ids, is_truncated in zip(
            outputs.rewards, outputs.completion_ids, outputs.is_truncated
        ):
            adj_reward = reward
            penalty = 0.0
            if (
                self.dapo_params.length_penalty_enabled
                and len(completion_ids) > target + cache
            ):
                over = len(completion_ids) - (target + cache)
                denom = max(target * 0.5, 1)
                frac = min(1.0, over / denom)
                penalty = max_penalty * frac
                adj_reward = adj_reward * (1.0 - penalty)
            length_penalties.append(penalty)
            if is_truncated:
                adj_reward = adj_reward * (1.0 - trunc_penalty)
                trunc_penalties.append(trunc_penalty)
            else:
                trunc_penalties.append(0.0)
            rewards.append(adj_reward)
        metrics = {
            "dapo/penalty/length": float(np.mean(length_penalties))
            if length_penalties
            else 0.0,
            "dapo/penalty/truncation": float(np.mean(trunc_penalties))
            if trunc_penalties
            else 0.0,
        }
        return rewards, metrics

    def _filter_processed_outputs(
        self,
        processed_results,
        env_results,
        rewards,
        indices: list[int],
    ) -> list[float]:
        def gather(seq):
            return [seq[i] for i in indices]

        processed_results.prompt_ids = gather(processed_results.prompt_ids)
        processed_results.prompt_mask = gather(processed_results.prompt_mask)
        processed_results.completion_ids = gather(processed_results.completion_ids)
        processed_results.completion_mask = gather(processed_results.completion_mask)
        processed_results.completion_logprobs = gather(
            processed_results.completion_logprobs
        )
        processed_results.is_truncated = gather(processed_results.is_truncated)
        filtered_rewards = gather(rewards)
        processed_results.rewards = filtered_rewards

        env_results.prompt = gather(env_results.prompt)
        env_results.completion = gather(env_results.completion)
        env_results.state = gather(env_results.state)
        env_results.reward = gather(env_results.reward)
        if env_results.metrics:
            for key, values in env_results.metrics.items():
                env_results.metrics[key] = gather(values)
        return filtered_rewards

    def _compute_group_advantages(
        self,
        rewards: list[float],
        prompts_in_batch: int,
        normalize: bool,
    ) -> list[float]:
        advantages = [0.0] * len(rewards)
        total = len(rewards)
        if prompts_in_batch == 0 or total == 0:
            return advantages
        for prompt_idx in range(prompts_in_batch):
            group_indices = [
                prompt_idx + k * prompts_in_batch
                for k in range(self.rollouts_per_example)
                if (prompt_idx + k * prompts_in_batch) < total
            ]
            if not group_indices:
                continue
            group_rewards = [rewards[i] for i in group_indices]
            mean = float(np.mean(group_rewards))
            if normalize:
                std = float(np.std(group_rewards)) + 1e-8
                adv_values = [(r - mean) / std for r in group_rewards]
            else:
                adv_values = [r - mean for r in group_rewards]
            for idx, adv in zip(group_indices, adv_values):
                advantages[idx] = adv
        return advantages

    def _select_dapo_indices(
        self,
        rewards: list[float],
        is_truncated: list[bool],
        prompts: list[Any],
        prompts_in_batch: int,
        batch_id: int,
    ) -> dict[str, Any]:
        total_samples = len(rewards)
        if total_samples == 0 or prompts_in_batch == 0:
            return {"indices": [], "advantages": [], "metrics": {}}
        kept_indices: set[int] = set()
        adv_map: Dict[int, float] = {}
        group_variances: list[float] = []
        skipped_groups: list[dict[str, Any]] = []
        selected_prompts: set[int] = set()
        trunc_skips = 0
        lowvar_skips = 0
        sign_skips = 0
        rng = np.random.default_rng(batch_id)
        # build canonical prompt keys for the first sample in each group
        def key_of(prompt_any: Any) -> str:
            try:
                if isinstance(prompt_any, list):
                    return self.processing_class.apply_chat_template(
                        prompt_any, tokenize=False, add_generation_prompt=True
                    )
                return str(prompt_any)
            except Exception:
                return str(prompt_any)

        for prompt_idx in range(prompts_in_batch):
            prompt_key = key_of(prompts[prompt_idx]) if prompt_idx < len(prompts) else str(prompt_idx)
            group_indices = [
                prompt_idx + k * prompts_in_batch
                for k in range(self.rollouts_per_example)
                if (prompt_idx + k * prompts_in_batch) < total_samples
            ]
            if not group_indices:
                continue
            valid = [idx for idx in group_indices if not is_truncated[idx]]
            if not valid:
                trunc_skips += 1
                skipped_groups.append({"prompt": prompt_idx, "indices": valid})
                continue
            group_rewards = [rewards[idx] for idx in valid]
            variance = float(np.var(group_rewards)) if len(group_rewards) > 1 else 0.0
            keep = True
            if self.dapo_params.dynamic_sampling:
                historical_var = self.prompt_variance.get(prompt_key)
                if historical_var is not None and historical_var < self.dapo_params.variance_threshold:
                    keep = False
                    lowvar_skips += 1
                else:
                    has_pos = any(r > 0 for r in group_rewards)
                    has_non_pos = any(r <= 0 for r in group_rewards)
                    if variance < self.dapo_params.variance_threshold:
                        keep = False
                        lowvar_skips += 1
                    elif not (has_pos and has_non_pos):
                        keep = False
                        sign_skips += 1
            if keep:
                advantages = self._compute_dapo_advantages(
                    group_rewards, self.dapo_params.normalize_advantages
                )
                for idx, adv in zip(valid, advantages):
                    kept_indices.add(idx)
                    adv_map[idx] = adv
                group_variances.append(variance)
                selected_prompts.add(prompt_idx)
                # update historical variance
                self.prompt_variance[prompt_key] = variance
            else:
                skipped_groups.append({"prompt": prompt_idx, "indices": valid, "key": prompt_key})
                # record observed variance for future decisions
                self.prompt_variance[prompt_key] = variance

        total_prompts = max(prompts_in_batch, 1)
        skip_ratio = len(skipped_groups) / total_prompts
        reincluded = 0
        if skip_ratio > self.dapo_params.skip_ratio_threshold and skipped_groups:
            num_to_reinclude = max(
                1, int(len(skipped_groups) * self.dapo_params.reinclude_fraction)
            )
            choices = (
                rng.choice(len(skipped_groups), size=num_to_reinclude, replace=False)
                if len(skipped_groups) > 1
                else [0]
            )
            for choice in np.atleast_1d(choices):
                group = skipped_groups[int(choice)]
                valid = group["indices"]
                if not valid:
                    continue
                group_rewards = [rewards[idx] for idx in valid]
                advantages = self._compute_dapo_advantages(
                    group_rewards, self.dapo_params.normalize_advantages
                )
                for idx, adv in zip(valid, advantages):
                    kept_indices.add(idx)
                    adv_map[idx] = adv
                selected_prompts.add(group["prompt"])
                reincluded += 1

        ordered_indices = sorted(list(kept_indices))
        ordered_advantages = [adv_map[idx] for idx in ordered_indices]
        metrics = {
            "dapo/prompts_total": float(total_prompts),
            "dapo/prompts_selected": float(len(selected_prompts)),
            "dapo/prompts_skipped": float(total_prompts - len(selected_prompts)),
            "dapo/skip_ratio": float(skip_ratio),
            "dapo/skip_truncated": float(trunc_skips),
            "dapo/skip_lowvar": float(lowvar_skips),
            "dapo/skip_one_sided": float(sign_skips),
            "dapo/reincluded": float(reincluded),
            "dapo/group_variance": float(np.mean(group_variances))
            if group_variances
            else 0.0,
        }

        return {
            "indices": ordered_indices,
            "advantages": ordered_advantages,
            "metrics": metrics,
        }

    def _compute_dapo_advantages(
        self,
        group_rewards: List[float],
        normalize: bool,
    ) -> List[float]:
        mean = float(np.mean(group_rewards))
        if normalize:
            std = float(np.std(group_rewards)) + 1e-8
            return [(r - mean) / std for r in group_rewards]
        return [r - mean for r in group_rewards]
