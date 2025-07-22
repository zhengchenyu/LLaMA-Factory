# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
import time
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from deepspeed import DeepSpeedEngine
from dlrover.python.common.storage import PosixDiskStorage
from dlrover.trainer.torch.flash_checkpoint.deepspeed import AsyncCheckpointAgent
from dlrover.trainer.torch.flash_checkpoint.deepspeed_engine import DeepSpeedCheckpointEngine
from dlrover.trainer.torch.flash_checkpoint.engine import CheckpointEngine
from dlrover.trainer.torch.flash_checkpoint.full_ckpt_engine import FullCheckpointEngine
from transformers import Seq2SeqTrainer, TrainingArguments, TrainerState, TrainerControl
from transformers.integrations import INTEGRATION_TO_CALLBACK, TensorBoardCallback
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

torch_native_save = torch.save
torch_native_load = torch.load

logger = logging.get_logger(__name__)

class CustomTensorBoardCallback(TensorBoardCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.tb_writer:
            cost = kwargs.get("cost", -1)
            if cost > -1:
                self.tb_writer.add_scalar(f"Checkpoint/{self.mode}", cost, state.x)


class HfFlashCheckpointer(object):
    def __init__(self, checkpoint_dir, storage=None):
        self.checkpoint_dir = checkpoint_dir
        self.storage = PosixDiskStorage() if not storage else storage
        self.ckpt_agent = AsyncCheckpointAgent(self.storage)
        self.async_save_engine: Optional[CheckpointEngine] = None

    def save_checkpoint_to_memory(self, step):
        success = self.async_save_engine.save_to_memory(
            step,
            self.ckpt_agent.state_dict,
            self.ckpt_agent.paths,
        )
        return success

    def save_checkpoint_to_storage(self, step):
        success = self.async_save_engine.save_to_storage(
            step,
            self.ckpt_agent.state_dict,
            self.ckpt_agent.paths,
        )
        return success

    def wait_for_copying(self):
        self.async_save_engine.wait_for_copying()

class HfDeepSpeedCheckpointer(HfFlashCheckpointer):
    def __init__(
        self,
        engine: DeepSpeedEngine,
        checkpoint_dir,
        storage=None,
        comm_backend="",
        non_blocking=False,
    ):
        super().__init__(checkpoint_dir, storage)
        self.engine = engine
        global_shard_num = 1
        if self.engine.zero_optimization():
            global_shard_num = dist.get_world_size(
                self.engine.optimizer.dp_process_group
            )
        zero_stage = self.engine.zero_optimization_stage()
        logger.info_rank0(f"HfDeepSpeedCheckpointer inited with non_blocking is {non_blocking}")
        self.async_save_engine = DeepSpeedCheckpointEngine(
            checkpoint_dir,
            storage=self.storage,
            global_shard_num=global_shard_num,
            zero_stage=zero_stage,
            comm_backend=comm_backend,
            non_blocking=non_blocking,
        )


class HfDdpCheckpointer(HfFlashCheckpointer):
    def __init__(
        self,
        checkpoint_dir,
        storage=None,
        comm_backend="",
        non_blocking=False,
    ):
        super().__init__(checkpoint_dir, storage)
        logger.info_rank0(f"HfDdpCheckpointer inited with non_blocking is {non_blocking}")
        self.async_save_engine = FullCheckpointEngine(
            checkpoint_dir,
            storage=self.storage,
            comm_backend=comm_backend,
            non_blocking=non_blocking,
        )

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        # reset INTEGRATION_TO_CALLBACK
        INTEGRATION_TO_CALLBACK["tensorboard"] = CustomTensorBoardCallback
        self.save_counter_for_disk = 0

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        self.use_flash_checkpoint = self.args.use_flash_checkpoint
        self.non_blocking = self.args.non_blocking
        self.save_to_disk_count = self.args.save_to_disk_count
        logger.info_rank0(f"zcydebug: use_flash_checkpoint is {self.use_flash_checkpoint}, "
                          f"non_blocking is {self.non_blocking}, "
                          f"save_to_disk_count is {self.save_to_disk_count}")

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        if self.use_flash_checkpoint and self.non_blocking and hasattr(self, "flash_checkpointer"):
            self.flash_checkpointer.wait_for_copying()

    def _save_checkpoint(self, model, trial):
        save_start_time = time.monotonic()
        if self.use_flash_checkpoint:
            self.save_counter_for_disk = (self.save_counter_for_disk + 1) % self.save_to_disk_count
            if (self.save_counter_for_disk % self.save_to_disk_count) == 0:
                self.save_to_disk = True
                logger.info_rank0(f"zcydebug: save_to_disk is true")
            else:
                self.save_to_disk = False
                logger.info_rank0(f"zcydebug: save_to_disk is false")
            run_dir = self._get_output_dir(trial=trial)
            if not hasattr(self, "flash_checkpointer"):
                if self.is_deepspeed_enabled:
                    self.flash_checkpointer = HfDeepSpeedCheckpointer(
                        self.model_wrapped, run_dir, non_blocking=self.non_blocking
                    )
                elif not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                    self.flash_checkpointer = HfDdpCheckpointer(run_dir, non_blocking=self.non_blocking)
                else:
                    raise ValueError(
                        "Flash Checkpoint only supports DeepSpeed or DDP."
                    )
            torch.save = self.flash_checkpointer.ckpt_agent.save
        super()._save_checkpoint(model, trial)
        if self.use_flash_checkpoint:
            torch.save = torch_native_save
            if self.save_to_disk:
                logger.info_rank0(f"zcydebug: save to storage")
                self.flash_checkpointer.save_checkpoint_to_storage(self.state.global_step)
            else:
                logger.info_rank0(f"zcydebug: save to memory")
                self.flash_checkpointer.save_checkpoint_to_memory(self.state.global_step)
        self.state.mode = "save_checkpoint"
        self.state.cost = time.monotonic() - save_start_time
        self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        # do we need ???
        # torch_native_save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _get_last_checkpoint_step(self):
        tracer_file = os.path.join(self.args.output_dir, "dlrover_latest.txt")
        if not os.path.exists(tracer_file):
            return 0
        with open(tracer_file, "r") as f:
            step = int(f.read())
        return step

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        load_start_time = time.monotonic()
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        self.state.mode = "load_checkpoint"
        self.state.cost = time.monotonic() - load_start_time
        self.control = self.callback_handler.on_save(self.args, self.state, self.control)
