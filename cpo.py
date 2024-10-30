from typing import Dict, Union, List, Literal, Tuple
import torch
import torch.nn.functional as F
from trl.commands.cli_utils import DPOScriptArguments
from utils.configs import H4ArgumentParser
from dpo import training_prepare
from trl import (
    CPOTrainer,
    CPOConfig,
    ModelConfig,
)


class MyCPOTrainer(CPOTrainer):
    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the CPO loss and other metrics for the given batch of inputs for train or test."""
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

        losses, chosen_rewards, rejected_rewards = self.cpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            batch,
        )

        loss = losses.mean() + self.cpo_alpha * policy_nll_loss
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            loss += getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss

        return loss, metrics

    def cpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            batch=None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        len_y_chosen = (batch["chosen_labels"] != -100).sum(-1)
        len_y_rejected = (batch["rejected_labels"] != -100).sum(-1)

        # https://arxiv.org/pdf/2405.14734 Table 3
        if self.loss_type == "rrhf":
            # https://arxiv.org/pdf/2304.05302
            losses = torch.relu(- policy_chosen_logps / len_y_chosen + policy_rejected_logps / len_y_rejected)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'simpo']"
            )

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()

        return losses, chosen_rewards, rejected_rewards


def main():
    parser = H4ArgumentParser((DPOScriptArguments, CPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse()

    if training_args.loss_type.split('-')[0] in ["rrhf"]:
        TrainerClass = MyCPOTrainer
    else:
        TrainerClass = CPOTrainer

    trainer = training_prepare(args, model_config, training_args, TrainerClass=TrainerClass)

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
