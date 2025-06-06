# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    RunningMoments,
    add_bos_token_if_needed,
    add_eos_token_if_needed,
    cap_exp,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


def _tokenize(
    features: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    args: DPOConfig,
    processor: Optional[Callable] = None,
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, List]:
    """
    Tokenizes and processes a batch of input features using the provided tokenizer and processor.
    """
    batch = defaultdict(list)

    if model is None:
        prompt = features["prompt"]
        images = features.get("images", [None] * len(features["prompt"]))
        negative1_images = features.get("negative1_images", [None] * len(features["prompt"]))
        negative2_images = features.get("negative2_images", [None] * len(features["prompt"]))
        negative3_images = features.get("negative3_images", [None] * len(features["prompt"]))
        negative4_images = features.get("negative4_images", [None] * len(features["prompt"]))
        prompt_tokens = _process_prompt(prompt, processor, tokenizer, images)
        negative1_prompt_tokens = _process_negative_prompt(prompt, processor, tokenizer, negative1_images, negative_idx=1)
        negative2_prompt_tokens = _process_negative_prompt(prompt, processor, tokenizer, negative2_images, negative_idx=2)
        negative3_prompt_tokens = _process_negative_prompt(prompt, processor, tokenizer, negative3_images, negative_idx=3)
        negative4_prompt_tokens = _process_negative_prompt(prompt, processor, tokenizer, negative4_images, negative_idx=4)
    
        chosen_tokens = _process_answer(prompt, features["chosen"], processor, tokenizer, images)
        rejected_tokens = _process_answer(prompt, features["rejected"], processor, tokenizer, images)
        # import ipdb; ipdb.set_trace()
        prompt_len_input_ids = _adjust_prompt_length(prompt_tokens, chosen_tokens, rejected_tokens)

        prompt_tokens, chosen_tokens, rejected_tokens = _add_special_tokens(
            tokenizer, prompt_len_input_ids, prompt_tokens, chosen_tokens, rejected_tokens
        )

        _truncate_tokens(chosen_tokens, rejected_tokens, prompt_tokens, args)

        # Makes `chosen_input_ids` and `chosen_attention_mask`
        _build_sequence_tokens(batch, chosen_tokens, args, "chosen")
        # Makes `rejected_input_ids` and `rejected_attention_mask`
        _build_sequence_tokens(batch, rejected_tokens, args, "rejected")

        # batch["chosen_input_ids"]
        # batch["chosen_attention_mask"]
        # batch["rejected_input_ids"]
        # batch["rejected_attention_mask"]

        # Try chaging L92-93 to: using negative_images
        # Then see if above tokens change or not.

        # Makes `prompt_input_ids` and `prompt_attention_mask`
        _append_prompt_tokens_to_batch(batch, prompt_tokens)
        # Makes `negative_prompt_input_ids` and `negative_prompt_attention_mask`
        _append_prompt_tokens_to_batch(batch, negative1_prompt_tokens)
        _append_prompt_tokens_to_batch(batch, negative2_prompt_tokens)
        _append_prompt_tokens_to_batch(batch, negative3_prompt_tokens)
        _append_prompt_tokens_to_batch(batch, negative4_prompt_tokens)
        # import ipdb; ipdb.set_trace()
    else:
        _tokenize_encoder_decoder(
            batch, tokenizer, features["prompt"], features["chosen"], features["rejected"], args, model
        )
    return dict(batch)


def _process_prompt(
    prompts: List[str], processor: Optional[Callable], tokenizer: PreTrainedTokenizerBase, images: List[Optional[Any]]
) -> List[Dict[str, List[int]]]:
    """
    Processes a list of prompts by tokenizing them, optionally using a processor for additional processing.
    """
    if processor:
        processor_kwargs = (
            {"add_special_tokens": False} if "add_special_tokens" in inspect.signature(processor).parameters else {}
        )
        prompt_tokens = []
        for prompt, image in zip(prompts, images):
            tokens = processor(text=prompt, images=image, **processor_kwargs)
            tokens = {k: v[0] for k, v in tokens.items()}
            if not isinstance(tokens["input_ids"], list):
                tokens["input_ids"] = tokens["input_ids"].tolist()
                tokens["attention_mask"] = tokens["attention_mask"].tolist()
            prompt_tokens.append(tokens)
    else:
        prompt_tokens = [tokenizer(prompt, add_special_tokens=False) for prompt in prompts]
    return [{f"prompt_{k}": v for k, v in tokens.items()} for tokens in prompt_tokens]


def _process_negative_prompt(
    prompts: List[str], processor: Optional[Callable], tokenizer: PreTrainedTokenizerBase, images: List[Optional[Any]], negative_idx: int
) -> List[Dict[str, List[int]]]:
    """
    Processes a list of prompts by tokenizing them, optionally using a processor for additional processing.
    """
    # import ipdb; ipdb.set_trace()
    if processor:
        processor_kwargs = (
            {"add_special_tokens": False}
            if "add_special_tokens" in inspect.signature(processor).parameters
            else {}
        )
        negative_prompt_tokens = []
        for prompt, image in zip(prompts, images):
            tokens = processor(text=prompt, images=image, **processor_kwargs)
            tokens = {k: v[0] for k, v in tokens.items()}
            if not isinstance(tokens["input_ids"], list):
                tokens["input_ids"] = tokens["input_ids"].tolist()
                tokens["attention_mask"] = tokens["attention_mask"].tolist()
            negative_prompt_tokens.append(tokens)
    else:
        negative_prompt_tokens = [tokenizer(prompt, add_special_tokens=False) for prompt in prompts]
    return [{f"negative{negative_idx}_prompt_{k}": v for k, v in tokens.items()} for tokens in negative_prompt_tokens]


def _process_answer(
    prompts: List[str],
    answers: List[str],
    processor: Optional[Callable],
    tokenizer: PreTrainedTokenizerBase,
    images: List[Optional[Any]],
) -> List[Dict[str, Any]]:
    return [
        _build_tokenized_answer(prompt, answer, image, processor=processor, tokenizer=tokenizer)
        for prompt, answer, image in zip(prompts, answers, images)
    ]


def _adjust_prompt_length(
    prompt_tokens: List[Dict[str, List[int]]],
    chosen_tokens: List[Dict[str, List[int]]],
    rejected_tokens: List[Dict[str, List[int]]],
) -> List[int]:
    prompt_len_input_ids = []
    for p_tokens, c_tokens, r_tokens in zip(prompt_tokens, chosen_tokens, rejected_tokens):
        c_len = len(c_tokens["prompt_input_ids"])
        r_len = len(r_tokens["prompt_input_ids"])
        min_len = min(c_len, r_len)

        for k, v in p_tokens.items():
            p_tokens[k] = v[:min_len]

        num_diff_tokens = sum([a != b for a, b in zip(c_tokens["prompt_input_ids"], r_tokens["prompt_input_ids"])])
        num_diff_len = abs(c_len - r_len)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the last token due to tokenizer merge ops."
            )
        prompt_len_input_ids.append(min_len)
    return prompt_len_input_ids


def _add_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    prompt_len_input_ids: List[int],
    prompt_tokens: List[Dict[str, List[int]]],
    chosen_tokens: List[Dict[str, List[int]]],
    rejected_tokens: List[Dict[str, List[int]]],
) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
    for i in range(len(prompt_tokens)):
        prompt_tokens[i], chosen_tokens[i], rejected_tokens[i] = add_bos_token_if_needed(
            tokenizer.bos_token_id,
            prompt_len_input_ids[i],
            prompt_tokens[i],
            len(chosen_tokens[i]["prompt_input_ids"]),
            chosen_tokens[i],
            len(rejected_tokens[i]["prompt_input_ids"]),
            rejected_tokens[i],
        )

        chosen_tokens[i], rejected_tokens[i] = add_eos_token_if_needed(
            tokenizer.eos_token_id, chosen_tokens[i], rejected_tokens[i]
        )
    return prompt_tokens, chosen_tokens, rejected_tokens


def _truncate_tokens(
    chosen_tokens: List[Dict[str, List[int]]],
    rejected_tokens: List[Dict[str, List[int]]],
    prompt_tokens: List[Dict[str, List[int]]],
    args: DPOConfig,
) -> None:
    """
    Truncates the tokens in chosen, rejected, and prompt sequences to ensure they fit within the maximum length constraints.
    """
    if args.truncation_mode not in ["keep_start", "keep_end"]:
        raise ValueError(f"Invalid truncation mode: {args.truncation_mode}")

    for c_tokens, r_tokens, p_tokens in zip(chosen_tokens, rejected_tokens, prompt_tokens):
        longer_response_length = max(len(c_tokens["input_ids"]), len(r_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [c_tokens, r_tokens, p_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > args.max_length:
                if args.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: args.max_prompt_length]
                elif args.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-args.max_prompt_length :]

        # if that's still too long, truncate the response from the end
        for answer_tokens in [c_tokens, r_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > args.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: args.max_length - args.max_prompt_length]


def _build_sequence_tokens(
    batch: Dict[str, List[int]], tokens: List[Dict[str, List[int]]], args: DPOConfig, prefix: str
) -> None:
    for token in tokens:
        sequence_tokens = {f"{prefix}_{k}": token[f"prompt_{k}"] + token[k] for k in ["input_ids", "attention_mask"]}
        sequence_tokens[f"{prefix}_labels"] = sequence_tokens[f"{prefix}_input_ids"][:]
        sequence_tokens[f"{prefix}_labels"][: len(token["prompt_input_ids"])] = [args.label_pad_token_id] * len(
            token["prompt_input_ids"]
        )
        for k, v in sequence_tokens.items():
            batch[k].append(v)


def _append_prompt_tokens_to_batch(batch: Dict[str, List[int]], prompt_tokens: List[Dict[str, List[int]]]) -> None:
    for p_tokens in prompt_tokens:
        for k, v in p_tokens.items():
            batch[k].append(v)


def _tokenize_encoder_decoder(
    batch: Dict[str, List[int]],
    tokenizer: PreTrainedTokenizerBase,
    prompt: List[str],
    chosen: List[str],
    rejected: List[str],
    args: DPOConfig,
    model: Optional[PreTrainedModel],
) -> None:
    chosen_tokens = tokenizer(chosen, truncation=True, max_length=args.max_target_length, add_special_tokens=True)
    rejected_tokens = tokenizer(rejected, truncation=True, max_length=args.max_target_length, add_special_tokens=True)
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=args.max_prompt_length, add_special_tokens=True)

    batch["chosen_labels"] = chosen_tokens["input_ids"]
    batch["rejected_labels"] = rejected_tokens["input_ids"]
    batch["prompt_input_ids"] = prompt_tokens["input_ids"]
    batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

    if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
        # Ensure the sequences are of the same length
        max_length = max(len(seq) for seq in batch["chosen_labels"] + batch["rejected_labels"])
        batch["chosen_labels"] = [
            seq + [tokenizer.pad_token_id] * (max_length - len(seq)) for seq in batch["chosen_labels"]
        ]
        batch["rejected_labels"] = [
            seq + [tokenizer.pad_token_id] * (max_length - len(seq)) for seq in batch["rejected_labels"]
        ]

        batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
            labels=torch.tensor(batch["rejected_labels"])
        )
        batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
            labels=torch.tensor(batch["chosen_labels"])
        )


def _build_tokenized_answer(
    prompt: str,
    answer: str,
    images: Optional[List[Any]] = None,
    processor: Optional[Callable] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Dict[str, Any]:
    """
    Build tokenized response, handling vision models and different tokenizers.
    """

    def tokenize(text, images=None):
        if processor:
            processor_kwargs = (
                {"add_special_tokens": False}
                if "add_special_tokens" in inspect.signature(processor).parameters
                else {}
            )
            tokenized = processor(text=text, images=images, **processor_kwargs)
            tokenized = {k: v[0] for k, v in tokenized.items()}
            if not isinstance(tokenized["input_ids"], list):
                tokenized["input_ids"] = tokenized["input_ids"].tolist()
                tokenized["attention_mask"] = tokenized["attention_mask"].tolist()
        else:
            tokenized = tokenizer(text, add_special_tokens=False)
        return tokenized

    full_tokenized = tokenize(prompt + answer, images)
    prompt_tokenized = tokenize(prompt, images)

    prompt_input_ids = prompt_tokenized["input_ids"]
    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    if len(full_tokenized["input_ids"]) != len(prompt_input_ids + answer_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    return_dict = {
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "input_ids": answer_input_ids,
        "attention_mask": answer_attention_mask,
    }
    if "pixel_values" in full_tokenized:
        return_dict["prompt_pixel_values"] = full_tokenized["pixel_values"]
    if "pixel_attention_mask" in full_tokenized:
        return_dict["prompt_pixel_attention_mask"] = full_tokenized["pixel_attention_mask"]

    return return_dict


class DPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`DPOConfig`):
            The DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "dpo"]

    @_deprecate_arguments(
        version="1.0.0",
        deprecated_args=[
            "beta",
            "label_smoothing",
            "loss_type",
            "label_pad_token_id",
            "padding_value",
            "truncation_mode",
            "max_length",
            "max_prompt_length",
            "max_target_length",
            "is_encoder_decoder",
            "disable_dropout",
            "generate_during_eval",
            "precompute_ref_log_probs",
            "dataset_num_proc",
            "model_init_kwargs",
            "ref_model_init_kwargs",
            "model_adapter_name",
            "ref_adapter_name",
            "reference_free",
            "force_use_ref_model",
        ],
        custom_message="Deprecated positional argument(s) used in DPOTrainer, please use the DPOConfig to set these arguments instead.",
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Optional[str] = None,
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
    ):
        
        # Initialize the add_image_dpo attribute
        self.add_image_dpo = args.add_image_dpo

        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_init_kwargs = model_init_kwargs

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the DPOTrainer/DPOConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if ref_model_init_kwargs is not None:
            warnings.warn(
                "You passed `ref_model_init_kwargs` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.ref_model_init_kwargs = ref_model_init_kwargs

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the DPOTrainer/DPOConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if force_use_ref_model:
            warnings.warn(
                "You passed `force_use_ref_model` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.force_use_ref_model = force_use_ref_model

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval:
            warnings.warn(
                "You passed `generate_during_eval` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.generate_during_eval = generate_during_eval
        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if is_encoder_decoder is not None:
            warnings.warn(
                "You passed `is_encoder_decoder` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.is_encoder_decoder = is_encoder_decoder
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder to the DPOTrainer/DPOConfig."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if model is not None:
            self.is_vision_model = model.config.model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()
        else:
            warnings.warn(
                "No model provided, cannot determine if it is a vision model. Setting is_vision_model to False."
            )
            self.is_vision_model = False

        if self.is_vision_model:
            self.processor = tokenizer
            self.tokenizer = tokenizer.tokenizer  # tokenizer is actually a processor at this point
        else:
            self.tokenizer = tokenizer

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if model_adapter_name is not None:
            warnings.warn(
                "You passed `model_adapter_name` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_adapter_name = model_adapter_name
        self.model_adapter_name = args.model_adapter_name

        if ref_adapter_name is not None:
            warnings.warn(
                "You passed `ref_adapter_name` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.ref_adapter_name = ref_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if reference_free:
            warnings.warn(
                "You passed `reference_free` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.reference_free = reference_free
        self.reference_free = args.reference_free

        if precompute_ref_log_probs:
            warnings.warn(
                "You passed `precompute_ref_log_probs` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.precompute_ref_log_probs = precompute_ref_log_probs

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")

        if max_length is not None:
            warnings.warn(
                "You passed `max_length` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_length = max_length
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_length = 512

        if max_prompt_length is not None:
            warnings.warn(
                "You passed `max_prompt_length` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_prompt_length = max_prompt_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_prompt_length = 128

        if max_target_length is not None:
            warnings.warn(
                "You passed `max_target_length` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_target_length = max_target_length
        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_target_length = 128

        if label_pad_token_id != -100:
            warnings.warn(
                "You passed `label_pad_token_id` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_pad_token_id = label_pad_token_id
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=self.tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if not disable_dropout:
            warnings.warn(
                "You passed `disable_dropout` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.disable_dropout = disable_dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        if padding_value is not None:
            warnings.warn(
                "You passed `padding_value` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.padding_value = padding_value
        self.padding_value = args.padding_value if padding_value is not None else self.tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        if truncation_mode != "keep_end":
            warnings.warn(
                "You passed `truncation_mode` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.truncation_mode = truncation_mode
        self.truncation_mode = args.truncation_mode
        self.max_target_length = args.max_target_length
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type is not None:
            warnings.warn(
                "You passed `loss_type` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.loss_type = loss_type
        if label_smoothing != 0:
            warnings.warn(
                "You passed `label_smoothing` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_smoothing = label_smoothing
        if (
            args.loss_type in ["hinge", "ipo", "bco_pair", "sppo_hard", "nca_pair", "apo_zero", "apo_down"]
            and args.label_smoothing > 0
        ):
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )
        if args.loss_type == "kto_pair":
            raise ValueError("Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.")

        if beta != 0.1:
            warnings.warn(
                "You passed `beta` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.beta = beta
        self.beta = args.beta
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed `dataset_num_proc` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset, lower writer batch size to avoid OOM (frequent in vision models)
            fn_kwargs = {
                "tokenizer": self.tokenizer,
                "args": args,
                "processor": self.processor if self.is_vision_model else None,
                "model": model if self.is_encoder_decoder else None,
            }
            train_dataset = train_dataset.map(
                _tokenize,
                fn_kwargs=fn_kwargs,
                batched=True,
                num_proc=self.dataset_num_proc,
                writer_batch_size=10,
                desc="Tokenizing train dataset",
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    _tokenize,
                    fn_kwargs=fn_kwargs,
                    batched=True,
                    num_proc=self.dataset_num_proc,
                    writer_batch_size=10,
                    desc="Tokenizing eval dataset",
                )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        if self.loss_type == "bco_pair":
            self.running = RunningMoments(self.accelerator)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

                # Unnecessary cache clearing to avoid OOM
                torch.cuda.empty_cache()
                self.accelerator.free_memory()

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="reference_chosen_logps", column=all_reference_chosen_logps)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps
    
    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_decoder_input_ids"] = torch.cat(
                [batch["chosen_decoder_input_ids"], batch["rejected_decoder_input_ids"]], dim=0
            ).to(device=device)
        """
        ######Editted #####
        # Check if prompt_pixel_attention_mask exists in the batch
        if "prompt_pixel_attention_mask" in batch:
            # Create a copy of the attention mask to avoid modifying the original
            modified_attention_mask = batch["prompt_pixel_attention_mask"].clone()

            # Example: Set a specific region of the image to be ignored
            # You can modify this part based on your specific requirements
            # For instance, to ignore the top-left quarter of the image:
            h, w = modified_attention_mask.shape[-2:]
            modified_attention_mask[..., :h//2, :w//2] = 0

            # Use the modified attention mask for concatenation
            concatenated_batch["pixel_attention_mask"] = torch.cat(
                [modified_attention_mask, modified_attention_mask], dim=0
            ).to(device=device)
        else:
            # If no pixel attention mask is provided, create a default one (all ones)
            image_pixel_values = batch["prompt_pixel_values"]
            default_attention_mask = torch.ones_like(image_pixel_values[:, :1, :, :])  # Use the first channel as a template
            concatenated_batch["pixel_attention_mask"] = torch.cat(
                [default_attention_mask, default_attention_mask], dim=0
            ).to(device=device)
        # import ipdb; ipdb.set_trace()
        ##### Editted ####
        """
        if is_vision_model:
            image_pixel_values = batch["prompt_pixel_values"]
            concatenated_batch["pixel_values"] = torch.cat(
                [image_pixel_values, image_pixel_values], dim=0
            )
            if "prompt_pixel_attention_mask" in batch:
                concatenated_batch["pixel_attention_mask"] = torch.cat(
                    [batch["prompt_pixel_attention_mask"], batch["prompt_pixel_attention_mask"]], dim=0
                )
        return concatenated_batch

    @staticmethod
    def concatenated_image_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            # max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
            max_length = batch["chosen_lables"].shape[1]
        else:
            # max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
            max_length = batch["chosen_input_ids"].shape[1]

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],  
                        concatenated_batch[concatenated_key],
                        concatenated_batch[concatenated_key],
                        concatenated_batch[concatenated_key],
                        concatenated_batch[concatenated_key],
                    ),
                    dim=0,
                ).to(device=device)

        # for k in batch:
            # if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                # if "labels" in k or is_encoder_decoder:
                    # pad_value = label_pad_token_id
                # elif k.endswith("_input_ids"):
                    # pad_value = padding_value
                # elif k.endswith("_attention_mask"):
                    # pad_value = 0
                # concatenated_key = k.replace("rejected", "concatenated")
                # concatenated_batch[concatenated_key] = torch.cat(
                    # (
                        # concatenated_batch[concatenated_key],
                        # pad_to_length(batch[k], max_length, pad_value=pad_value),
                    # ),
                    # dim=0,
                # ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_decoder_input_ids"] = torch.cat(
                [batch["chosen_decoder_input_ids"], batch["rejected_decoder_input_ids"]], dim=0
            ).to(device=device)

        if is_vision_model:
            image_pixel_values = batch["prompt_pixel_values"]
            negative1_image_pixel_values = batch["negative1_prompt_pixel_values"]
            negative2_image_pixel_values = batch["negative2_prompt_pixel_values"]
            negative3_image_pixel_values = batch["negative3_prompt_pixel_values"]
            negative4_image_pixel_values = batch["negative4_prompt_pixel_values"]
            concatenated_batch["pixel_values"] = torch.cat(
                [image_pixel_values, negative1_image_pixel_values, negative2_image_pixel_values, negative3_image_pixel_values, negative4_image_pixel_values], dim=0
            )

            if "prompt_pixel_attention_mask" in batch:
                concatenated_batch["pixel_attention_mask"] = torch.cat(
                    [batch["prompt_pixel_attention_mask"], batch["negative1_prompt_pixel_attention_mask"], batch["negative2_prompt_pixel_attention_mask"], batch["negative3_prompt_pixel_attention_mask"], batch["negative4_prompt_pixel_attention_mask"]], dim=0
                )
        return concatenated_batch

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps.to(self.accelerator.device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios = pi_logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = pi_logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        elif self.loss_type == "aot_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)

            delta = chosen_logratios_sorted - rejected_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios_sorted, _ = torch.sort(pi_logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)

            delta = pi_logratios_sorted - ref_logratios_sorted

            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output

            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood

            losses = losses_chosen + losses_rejected

        elif self.loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output

            losses_chosen = F.sigmoid(self.beta * chosen_logratios)  # Decrease chosen likelihood
            losses_rejected = 1 - F.sigmoid(
                self.beta * (chosen_logratios - rejected_logratios)
            )  # Decrease rejected likelihood more

            losses = losses_chosen + losses_rejected

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def dpo_pl5_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected1_logps: torch.FloatTensor,
        policy_rejected2_logps: torch.FloatTensor,
        policy_rejected3_logps: torch.FloatTensor,
        policy_rejected4_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected1_logps: torch.FloatTensor,
        reference_rejected2_logps: torch.FloatTensor,
        reference_rejected3_logps: torch.FloatTensor,
        reference_rejected4_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected1_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            policy_rejected2_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected1_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            reference_rejected2_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected1_logratios = policy_rejected1_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected1_logps.to(self.accelerator.device)
        rejected2_logratios = policy_rejected2_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected2_logps.to(self.accelerator.device)
        rejected3_logratios = policy_rejected3_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected3_logps.to(self.accelerator.device)
        rejected4_logratios = policy_rejected4_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected4_logps.to(self.accelerator.device)

        chosen_score = self.beta * (policy_chosen_logps - reference_chosen_logps).to(self.accelerator.device)
        rejected1_score = self.beta * (policy_rejected1_logps - reference_rejected1_logps).to(self.accelerator.device)
        rejected2_score = self.beta * (policy_rejected2_logps - reference_rejected2_logps).to(self.accelerator.device)
        rejected3_score = self.beta * (policy_rejected3_logps - reference_rejected3_logps).to(self.accelerator.device)
        rejected4_score = self.beta * (policy_rejected4_logps - reference_rejected4_logps).to(self.accelerator.device)
        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid": 
            losses1 = -(torch.log(torch.exp(chosen_score)) - torch.log(torch.exp(chosen_score) + torch.exp(rejected1_score) + torch.exp(rejected2_score) + torch.exp(rejected3_score) + torch.exp(rejected4_score)))
            losses2 = -(torch.log(torch.exp(rejected1_score)) - torch.log(torch.exp(rejected1_score) + torch.exp(rejected2_score) + torch.exp(rejected3_score) + torch.exp(rejected4_score)))
            losses3 = -(torch.log(torch.exp(rejected2_score)) - torch.log(torch.exp(rejected2_score) + torch.exp(rejected3_score) + torch.exp(rejected4_score)))
            losses4 = -(torch.log(torch.exp(rejected3_score)) - torch.log(torch.exp(rejected3_score) + torch.exp(rejected4_score)))
            losses = losses1 + losses2 + losses3 + losses4
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected1_rewards = (
            self.beta
            * (
                policy_rejected1_logps.to(self.accelerator.device)
                - reference_rejected1_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected2_rewards = (
            self.beta
            * (
                policy_rejected2_logps.to(self.accelerator.device)
                - reference_rejected2_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected3_rewards = (   
            self.beta
            * (
                policy_rejected3_logps.to(self.accelerator.device)
                - reference_rejected3_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected4_rewards = (   
            self.beta
            * (
                policy_rejected4_logps.to(self.accelerator.device)
                - reference_rejected4_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected1_rewards, rejected2_rewards, rejected3_rewards, rejected4_rewards



    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits (batch and sequence length dim) {logits.shape[:-1]} and labels must have the same shape {labels.shape}."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)
    
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], is_image_dpo: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        """
        Comments for Fatima:
        batch.keys():
        dict_keys(['rejected', 'chosen', 'prompt', 'img_path', 'images', 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'prompt_input_ids', 'prompt_attention_mask', 'prompt_pixel_values', 'prompt_pixel_attention_mask'])
        batch['chosen_input_ids'].shape: torch.Size([1, 232])
        batch['chosen_attention_mask'].shape: torch.Size([1, 232])
        batch['rejected_input_ids'].shape: torch.Size([1, 221])
        batch['rejected_attention_mask'].shape: torch.Size([1, 221])
        batch['prompt_input_ids'].shape: torch.Size([1, 89])
        batch['prompt_pixel_values'].shape: torch.Size([1, 1, 3, 427, 640])
        batch['prompt_pixel_attention_mask'].shape: torch.Size([1, 1, 427, 640])
        """ 
        if is_image_dpo:
            concatenated_batch = self.concatenated_image_inputs(
                batch,
                is_encoder_decoder=self.is_encoder_decoder,
                is_vision_model=self.is_vision_model,
                # is_vision_model = True
                label_pad_token_id=self.label_pad_token_id,
                padding_value=self.padding_value,
                device=self.accelerator.device,
            )
        else:
            concatenated_batch = self.concatenated_inputs(
                batch,
                is_encoder_decoder=self.is_encoder_decoder,
                is_vision_model=self.is_vision_model,
                # is_vision_model = True
                label_pad_token_id=self.label_pad_token_id,
                padding_value=self.padding_value,
                device=self.accelerator.device,
            )
        """
        concatenated_batch.keys():
        dict_keys(['concatenated_input_ids', 'concatenated_attention_mask', 'concatenated_labels', 'pixel_values', 'pixel_attention_mask'])
        concatenated_batch['concatenated_input_ids'].shape: torch.Size([3, 232])
        concatenated_batch['concatenated_attention_mask'].shape: torch.Size([3, 232])
        concatenated_batch['pixel_values'].shape: torch.Size([3, 1, 3, 427, 640])
        concatenated_batch['pixel_attention_mask'].shape: torch.Size([3, 1, 427, 640])
        """
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.get("concatenated_decoder_input_ids")

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            # for llava, the model returns logits for the entire sequence, including the image tokens (placed before the text tokens)
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:]

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion

        if is_image_dpo:
            chosen_logps = all_logps[:len_chosen]
            rejected1_logps = all_logps[len_chosen:len_chosen*2]
            rejected2_logps = all_logps[len_chosen*2:len_chosen*3]
            rejected3_logps = all_logps[len_chosen*3:len_chosen*4]
            rejected4_logps = all_logps[len_chosen*4:]

            chosen_logits = all_logits[:len_chosen]
            rejected1_logits = all_logits[len_chosen:len_chosen*2]
            rejected2_logits = all_logits[len_chosen*2:len_chosen*3]
            rejected3_logits = all_logits[len_chosen*3:len_chosen*4]
            rejected4_logits = all_logits[len_chosen*4:]
            if self.aux_loss_enabled:
                return (chosen_logps, rejected1_logps, rejected2_logps, rejected3_logps, rejected4_logps, chosen_logits, rejected1_logits, rejected2_logits, rejected3_logits, rejected4_logits, nll_loss, outputs.aux_loss)

            return (chosen_logps, rejected1_logps, rejected2_logps, rejected3_logps, rejected4_logps, chosen_logits, rejected1_logits, rejected2_logits, rejected3_logits, rejected4_logits, nll_loss)
        else:
            chosen_logps = all_logps[:len_chosen]
            rejected_logps = all_logps[len_chosen:]

            chosen_logits = all_logits[:len_chosen]
            rejected_logits = all_logits[len_chosen:]

            if self.aux_loss_enabled:
                return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
        add_image_dpo: bool = False,
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        if add_image_dpo:
            forward_image_dpo_output = self.concatenated_forward(model, batch, is_image_dpo=True)
            (
                policy_chosen_logps_image_dpo,
                policy_rejected1_logps_image_dpo,
                policy_rejected2_logps_image_dpo,
                policy_rejected3_logps_image_dpo,
                policy_rejected4_logps_image_dpo,
                policy_chosen_logits_image_dpo,
                policy_rejected1_logits_image_dpo,
                policy_rejected2_logits_image_dpo,
                policy_rejected3_logits_image_dpo,
                policy_rejected4_logits_image_dpo,
                policy_nll_loss_image_dpo,
            ) = forward_image_dpo_output[:11]
        
        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and (self.precompute_ref_log_probs or self.args.rpo_alpha is not None)
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                        if add_image_dpo:
                            (
                                reference_chosen_logps_image_dpo,
                                reference_rejected1_logps_image_dpo,
                                reference_rejected2_logps_image_dpo,
                                reference_rejected3_logps_image_dpo,
                                reference_rejected4_logps_image_dpo,
                                _,
                                _,
                                _,
                                _,
                                _,
                                _,
                            ) = self.concatenated_forward(self.model, batch, is_image_dpo=True)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)
                    if add_image_dpo:
                        (
                            reference_chosen_logps_image_dpo,
                            reference_rejected1_logps_image_dpo,
                            reference_rejected2_logps_image_dpo,
                            reference_rejected3_logps_image_dpo,
                            reference_rejected4_logps_image_dpo,
                            _,
                            _,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.ref_model, batch, is_image_dpo=True)

        losses_text_dpo, chosen_rewards_text_dpo, rejected_rewards_text_dpo = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies_text_dpo = (chosen_rewards_text_dpo > rejected_rewards_text_dpo).float()

        if add_image_dpo:
            losses_image_dpo, chosen_rewards_image_dpo, rejected1_rewards_image_dpo, rejected2_rewards_image_dpo, rejected3_rewards_image_dpo, rejected4_rewards_image_dpo = self.dpo_pl5_loss(
                policy_chosen_logps_image_dpo,
                policy_rejected1_logps_image_dpo,
                policy_rejected2_logps_image_dpo,
                policy_rejected3_logps_image_dpo,
                policy_rejected4_logps_image_dpo,
                reference_chosen_logps_image_dpo,
                reference_rejected1_logps_image_dpo,
                reference_rejected2_logps_image_dpo,
                reference_rejected3_logps_image_dpo,
                reference_rejected4_logps_image_dpo,
            )
            reward_accuracies_image_dpo = (chosen_rewards_image_dpo > rejected1_rewards_image_dpo).float()
        
        losses = losses_text_dpo
        chosen_rewards = chosen_rewards_text_dpo
        rejected_rewards = rejected_rewards_text_dpo
 
        if add_image_dpo:
            losses = losses_text_dpo + losses_image_dpo
            chosen_rewards = chosen_rewards_text_dpo
            rejected_rewards = rejected_rewards_text_dpo
            
            # Calculate anchor loss
            anchor_loss = -F.logsigmoid(self.beta * (policy_chosen_logps - reference_chosen_logps))
            losses = losses + anchor_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            # RPO loss from V3 of the paper:
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        if add_image_dpo:
            metrics[f"{prefix}rewards_text/chosen"] = chosen_rewards_text_dpo.mean().cpu()
            metrics[f"{prefix}rewards_text/rejected"] = rejected_rewards_text_dpo.mean().cpu()
            metrics[f"{prefix}rewards_text/accuracies"] = reward_accuracies_text_dpo.mean().cpu()
            metrics[f"{prefix}rewards_text/margins"] = (chosen_rewards_text_dpo - rejected_rewards_text_dpo).mean().cpu()
            metrics[f"{prefix}rewards_image/chosen"] = chosen_rewards_image_dpo.mean().cpu()
            metrics[f"{prefix}rewards_image/rejected1"] = rejected1_rewards_image_dpo.mean().cpu()
            metrics[f"{prefix}rewards_image/rejected2"] = rejected2_rewards_image_dpo.mean().cpu()
            metrics[f"{prefix}rewards_image/rejected3"] = rejected3_rewards_image_dpo.mean().cpu()
            metrics[f"{prefix}rewards_image/rejected4"] = rejected4_rewards_image_dpo.mean().cpu()
            metrics[f"{prefix}rewards_image/accuracies"] = reward_accuracies_image_dpo.mean().cpu()
            metrics[f"{prefix}rewards_image/margins"] = (chosen_rewards_image_dpo - rejected1_rewards_image_dpo).mean().cpu()
   
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss, metrics

        return losses.mean(), metrics
    
    def compute_listwise_dpo_loss(self, model, hard_negatives, beta):
    # Implement the Listwise DPO loss calculation
        policy_logps = []
        ref_logps = []
    
        # Get logprobs for policy model
        with torch.no_grad():
            for negative in hard_negatives:
                policy_output = self.concatenated_forward(model, negative)
                policy_logps.append(policy_output[0])  # Assuming logps are the first output
    
        # Get logprobs for reference model
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    for negative in hard_negatives:
                        ref_output = self.concatenated_forward(self.model, negative)
                        ref_logps.append(ref_output[0])
            else:
                    for negative in hard_negatives:
                        ref_output = self.concatenated_forward(self.ref_model, negative)
                        ref_logps.append(ref_output[0])
    
        # Convert lists to tensors
        policy_logps = torch.stack(policy_logps)
        ref_logps = torch.stack(ref_logps)
    
        # Compute Listwise DPO loss
        loss = -torch.mean(policy_logps - ref_logps + beta * torch.log(torch.softmax(ref_logps / beta, dim=0)))
        return loss

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train", add_image_dpo=self.add_image_dpo)

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
    
    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with generate_context_manager:
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(
        self,
        commit_message: Optional[str] = "End of training",
        blocking: bool = True,
        **kwargs,
    ) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "dpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        Unlike the parent class, we don't use the `token` argument to mitigate security risks.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
