import argparse
import json
import os
from glob import glob
from os.path import join
from random import sample
import random
import numpy as np
import pandas as pd
import torch
import matplotlib_inline.backend_inline
from accelerate import PartialState
from transformers import AutoTokenizer
from grad_heatmap_1 import load_model_and_tokenizer, load_and_process_dataset, compute_per_response_gradients
from tqdm import tqdm
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
random.seed(42)


def process_dataset_and_compute_gradients(model, tokenizer, dataset, batch_size=1):
    """
    Process the dataset and compute per-token gradients for positive and negative responses.
    """
    correlation_data = []
    inner_product_data = []
    grad_w_data = []
    grad_l_data = []
    for batch_idx in tqdm(range(0, len(dataset), batch_size), desc="Processing Dataset"):
        batch_samples = dataset[batch_idx:batch_idx + batch_size]

        batch_samples['chosen'] = [i[0] + i[1] for i in zip(batch_samples['prompt'], batch_samples['chosen'])]
        batch_samples['rejected'] = [i[0] + i[1] for i in zip(batch_samples['prompt'], batch_samples['rejected'])]

        batch_prompts, batch_chosen_texts, batch_rejected_texts = batch_samples['prompt'], batch_samples['chosen'], batch_samples['rejected']

        prompt_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True)
        chosen_inputs = tokenizer(batch_chosen_texts, return_tensors='pt', padding=True)
        rejected_inputs = tokenizer(batch_rejected_texts, return_tensors='pt', padding=True)
        prompt_input_ids = prompt_inputs['input_ids']
        chosen_input_ids = chosen_inputs['input_ids']
        rejected_input_ids = rejected_inputs['input_ids']
        # For each sample in batch, determine response_start_idx
        response_start_idxs = []
        for i in range(len(batch_prompts)):
            # The response starts after the length of the prompt for each sample
            prompt_len = (prompt_input_ids[i] != tokenizer.pad_token_id).sum().item()
            response_start_idxs.append(prompt_len)

        response_start_idxs = torch.tensor(response_start_idxs)

        # Compute per-response gradients
        pos_response_grads = compute_per_response_gradients(
            model, tokenizer, chosen_input_ids, response_start_idxs, keep_torch_tensor=True)
        neg_response_grads = compute_per_response_gradients(
            model, tokenizer, rejected_input_ids, response_start_idxs, keep_torch_tensor=True)

        # cosine_sim_response = cosine_similarity(pos_response_grads.reshape(1, -1),
        #                                         neg_response_grads.reshape(1, -1))[0, 0]

        inner_product = torch.dot(pos_response_grads, neg_response_grads)
        grad_w_norm = torch.norm(pos_response_grads, p=2)
        grad_l_norm = torch.norm(neg_response_grads, p=2)
        correlation = inner_product / (grad_w_norm * grad_l_norm)
        inner_product_data.append(inner_product.cpu().numpy())
        correlation_data.append(correlation.cpu().numpy())
        grad_w_data.append(grad_w_norm.cpu().numpy())
        grad_l_data.append(grad_l_norm.cpu().numpy())

    # calculate the mean cosine similarity
    condition_data = np.mean(np.array(inner_product_data) / np.min([np.array(grad_w_data) ** 2, np.array(grad_w_data) ** 2], axis=0))
    inner_product_data = np.mean(inner_product_data)
    correlation_data = np.mean(correlation_data)
    grad_w_data = np.mean(grad_w_data)
    grad_l_data = np.mean(grad_l_data)

    return inner_product_data, correlation_data, grad_w_data, grad_l_data, condition_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='outputs/l3-8b-tldr-dpo-sigmoid',
                        help="Model name or path")
    parser.add_argument("--dataset_name", type=str, default='CarperAI/openai_summarize_comparisons')
    parser.add_argument("--eval_split", type=str, default='test', help="Evaluation split")
    parser.add_argument("--output_csv", type=str, default='plots/grad/l3-dpo.csv',
                        help="Output directory for heatmaps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--full_parameters", action='store_true', help="Use full parameters")
    parser.add_argument("--start_from_inverse", action='store_true', help="Start from inverse")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset_name, split=args.eval_split, tokenizer=tokenizer)
    dataset = dataset.select(sample(range(len(dataset)), k=min(512, len(dataset))))

    state = PartialState()

    correlation_logs = []
    ckpt_list = glob(join(args.model_name_or_path, "checkpoint-*"))
    # sort by step
    ckpt_list = sorted(ckpt_list, key=lambda x: int(x.split("-")[-1]), reverse=False)
    if args.start_from_inverse:
        ckpt_list = ckpt_list[::-1]
    with state.split_between_processes(ckpt_list) as ckpt_paths:
        print(f"{state.process_index}: Run {ckpt_paths} on {state.device}.")
        for model_path in tqdm(ckpt_paths):

            # check if this step has been processed
            if os.path.exists(args.output_csv.replace(".csv", ".jsonl")):
                with open(args.output_csv.replace(".csv", ".jsonl"), "r") as f:
                    logs = [json.loads(line) for line in f.readlines()]
                if int(model_path.split("-")[-1]) in [log["Step"] for log in logs]:
                    print(f"Step {int(model_path.split('-')[-1])} has been processed.")
                    continue

            print("Loading model from", model_path)
            model, tokenizer = load_model_and_tokenizer(model_path)
            model.to(state.device)

            # set lora_weights to require grad
            required_grad_names = []
            for name, param in model.named_parameters():
                if args.full_parameters: # and ('wte' in name or "lm_head" in name):
                    param.requires_grad = True
                    required_grad_names.append(name)
                else:
                    if 'lora' in name:
                        param.requires_grad = True
                        required_grad_names.append(name)
                    else:
                        param.requires_grad = False
            print(f"Required gradients for {model_path}: {required_grad_names}")

            inner_product, correlation, grad_w_nrom, grad_l_nrom, grad_condition = process_dataset_and_compute_gradients(
                model, tokenizer, dataset, batch_size=args.batch_size)
            correlation_logs.append({
                "Step": int(model_path.split("-")[-1]),
                "InnerProduct": float(inner_product),
                "Correlation": float(correlation),
                "Grad_W_Norm": float(grad_w_nrom),
                "Grad_L_Norm": float(grad_l_nrom),
                "Grad_Condition": float(grad_condition)
            })

            # write last log to jsonl
            with open(args.output_csv.replace(".csv", ".jsonl"), "a") as f:
                f.write(json.dumps(correlation_logs[-1]) + "\n")

            # clean up
            del model
            torch.cuda.empty_cache()

    # state.wait_for_everyone()
    if state.is_main_process:
        # read josnl from file and write to csv
        with open(args.output_csv.replace(".csv", ".jsonl"), "r") as f:
            logs = [json.loads(line) for line in f.readlines()]
        pd.DataFrame(logs).to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
