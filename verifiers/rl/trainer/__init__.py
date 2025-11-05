import logging
import importlib

import torch._dynamo

from .config import RLConfig

torch._dynamo.config.suppress_errors = True


def __getattr__(name: str):
    if name == "RLTrainer":
        return getattr(importlib.import_module("verifiers.rl.trainer.trainer"), name)
    if name == "GRPOTrainer":
        def _grpo_trainer(model, processing_class, env, args):
            logging.warning("GRPOTrainer is deprecated and renamed to RLTrainer.")
            RLTrainer = getattr(importlib.import_module("verifiers.rl.trainer.trainer"), "RLTrainer")
            return RLTrainer(model, processing_class, env, args)
        return _grpo_trainer
    if name in {"GRPOConfig", "grpo_defaults"}:
        def _cfg(**kwargs):
            logging.warning("GRPOConfig/grpo_defaults is deprecated and renamed to RLConfig.")
            return RLConfig(**kwargs)
        return _cfg
    if name == "lora_defaults":
        def _lora_defaults(**_kwargs):
            raise ValueError("lora_defaults is deprecated and replaced with RLConfig.")
        return _lora_defaults
    raise AttributeError(name)


__all__ = [
    "RLConfig",
    "RLTrainer",
    "GRPOTrainer",
    "GRPOConfig",
    "grpo_defaults",
    "lora_defaults",
]
