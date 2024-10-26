import random
from random import sample
from typing import Dict, Union, List, Literal, Tuple
from trl.commands.cli_utils import DPOScriptArguments
from utils.configs import H4ArgumentParser
from accelerate import Accelerator
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from trl import (
    DPOConfig,
    DPOTrainer,
    CPOTrainer,
    ModelConfig,
    maybe_extract_prompt,
    maybe_apply_chat_template,
    get_peft_config,
    get_quantization_config, FDivergenceType,
)
from utils.hh_dataset import extract_dialogue

random.seed(42)


class MyDPOTrainer(DPOTrainer):
    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
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
                        reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                            self.model, batch
                        )[:2]
                else:
                    reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                        self.ref_model, batch
                    )[:2]

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            batch,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            # RPO loss from V3 of the paper:
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss, metrics

        return losses.mean(), metrics

    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            batch=None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
        #     not self.reference_free
        # ) * reference_chosen_logps.to(self.accelerator.device)
        # rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
        #     not self.reference_free
        # ) * reference_rejected_logps.to(self.accelerator.device)
        assert self.f_divergence_type != FDivergenceType.ALPHA_DIVERGENCE.value and self.f_divergence_type != FDivergenceType.JS_DIVERGENCE.value

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # https://arxiv.org/pdf/2405.14734 Table 3
        if self.loss_type == "rdpo":
            # https://arxiv.org/pdf/2403.19159
            # α = 0.2 for tldr (Paper Section 4.3: For HH, we use α ∈ [0, 0.005, 0.01]; for TL;DR, we use α ∈ [0, 0.2, 0.5] (α = 0 is standard DPO))
            rdpo_term = (batch["chosen_labels"] != -100).sum(-1) - (batch["rejected_labels"] != -100).sum(-1)
            losses = -F.logsigmoid(self.beta * logits + 0.2 * rdpo_term)
        elif self.loss_type == "pdpo":
            # https://arxiv.org/pdf/2402.13228
            # Section5.2: We test λ ∈ {5, 50, 500} and find that performance on MetaMath and ARC is relatively unaffected by the choice of λ
            pdpo_term = torch.relu(reference_chosen_logps - policy_chosen_logps)
            losses = -F.logsigmoid(self.beta * (logits - 50 * pdpo_term))
        elif self.loss_type == "sppo":
            loss_w = ((policy_chosen_logps - reference_chosen_logps) - (1 / self.beta) * (1.0 - 0.5)) ** 2
            loss_l = ((policy_rejected_logps - reference_rejected_logps) + (1 / self.beta) * (1.0 - 0.5)) ** 2
            losses = (loss_w + loss_l) / 2
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = (
                self.beta
                * (
                        policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(
                    self.accelerator.device)
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


def training_prepare(args, model_config, training_args,
                     TrainerClass=DPOTrainer, ModelClass=AutoModelForCausalLM,
                     device_map=None):
    if device_map is None:
        device_map = {"": Accelerator().local_process_index}
    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map,
        quantization_config=quantization_config
    )
    model = ModelClass.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    # print(peft_config)
    if peft_config is None:
        ref_model = ModelClass.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    ################
    # Dataset
    ################
    if "senti" in args.dataset_name:
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name)
        def process(row):
            row["prompt"] = [{"role": "user", "content": row["prompt"]}]
            row["chosen"] = [{"role": "assistant", "content": row["chosen"]}]
            row["rejected"] = [{"role": "assistant", "content": row["rejected"]}]
            return row

        with PartialState().local_main_process_first():
            if "hh-rlhf" in args.dataset_name:
                dataset = dataset.map(extract_dialogue, num_proc=training_args.dataset_num_proc)
            elif "openai_summarize_comparisons" in args.dataset_name:
                dataset = dataset.map(process, num_proc=training_args.dataset_num_proc)
            elif "ultrafeedback_binarized" in args.dataset_name:
                dataset = dataset.remove_columns("messages")
            dataset = dataset.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)
            dataset = dataset.map(
                maybe_apply_chat_template, num_proc=training_args.dataset_num_proc, fn_kwargs={"tokenizer": tokenizer}
            )
    ################
    # Training
    ################
    trainer = TrainerClass(
        *[model] if (TrainerClass is CPOTrainer or issubclass(TrainerClass, CPOTrainer)) else [model, ref_model],
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split].select(
            sample(range(len(dataset[args.dataset_test_split])), 
                   k=min(2048,len(dataset[args.dataset_test_split])))), # sample 2048 examples for evaluation
        tokenizer=tokenizer,
        peft_config=peft_config
    )
    return trainer


def main():
    parser = H4ArgumentParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse()

    if training_args.loss_type.split('-')[0] in ["pdpo", "sppo"]:
        TrainerClass = MyDPOTrainer
    else:
        TrainerClass = DPOTrainer

    trainer = training_prepare(args, model_config, training_args, TrainerClass=TrainerClass)
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()