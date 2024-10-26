from utils.hh_dataset import extract_dialogue
import argparse
import os
import numpy as np
import torch
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import log_softmax
from tqdm import tqdm
import json
from trl import maybe_extract_prompt, maybe_apply_chat_template

def torch_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif hasattr(torch, 'mlu') and torch.mlu.is_available():
        torch.mlu.empty_cache()
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

def load_model_and_tokenizer(model_name_or_path):
    """
    Load the pre-trained model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, tie_word_embeddings=True)
    # Set model.lm_head.weight as the model.transformer.wte.weight.T (Be careful to copy the tensor, and not just the reference)
    # model.lm_head.weight = torch.nn.Parameter(model.transformer.wte.weight.detach().clone())

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()  # Set model to evaluation mode
    return model, tokenizer

def load_and_process_dataset(dataset_name, split='train', dataset_num_proc=16, tokenizer=None):
    """
    Load and process the dataset.
    Assumes the dataset has 'prompt', 'chosen', and 'rejected' fields.
    """
    if os.path.exists(dataset_name):
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name)

        def process(row):
            row["prompt"] = [{"role": "user", "content": row["prompt"]}]
            row["chosen"] = [{"role": "assistant", "content": row["chosen"]}]
            row["rejected"] = [{"role": "assistant", "content": row["rejected"]}]
            return row

        with PartialState().local_main_process_first():
            if dataset_name == "Anthropic/hh-rlhf":
                dataset = dataset.map(extract_dialogue, num_proc=dataset_num_proc)
            else:
                dataset = dataset.map(process, num_proc=dataset_num_proc)
            dataset = dataset.map(maybe_extract_prompt, num_proc=dataset_num_proc)
            dataset = dataset.map(
                maybe_apply_chat_template, num_proc=dataset_num_proc, fn_kwargs={"tokenizer": tokenizer})
    return dataset[split]

def compute_per_token_gradients(model, tokenizer, input_ids, response_start_idx):
    """
    Compute the per-token gradients of the log probabilities with respect to model parameters.
    """
    input_ids = input_ids.to(model.device)
    input_ids = input_ids.clone().detach()
    outputs = model(input_ids)
    logits = outputs.logits
    log_probs = log_softmax(logits, dim=-1)
    shift_log_probs = log_probs[:, :-1, :].contiguous()
    shift_input_ids = input_ids[:, 1:].contiguous()
    target_log_probs = shift_log_probs.gather(2, shift_input_ids.unsqueeze(-1)).squeeze(-1)
    seq_len = target_log_probs.size(1)
    per_token_grads = []
    tokens = []
    for i in range(response_start_idx - 1, seq_len):
        model.zero_grad()
        log_p = target_log_probs[0, i]
        log_p.backward(retain_graph=True)
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().cpu().flatten())
        grad_vector = torch.cat(grads).cpu().numpy()
        per_token_grads.append(grad_vector)
        token_id = shift_input_ids[0, i]
        token_text = tokenizer.decode([token_id])
        tokens.append(token_text)
        del grad_vector
        torch_empty_cache()
    return per_token_grads, tokens

def compute_per_response_gradients(model, tokenizer, input_ids_batch, response_start_idx_batch, keep_torch_tensor=False):
    input_ids_batch = input_ids_batch.to(model.device)
    input_ids_batch = input_ids_batch.clone().detach()
    outputs = model(input_ids_batch)
    logits = outputs.logits
    log_probs = log_softmax(logits, dim=-1)
    shift_log_probs = log_probs[:, :-1, :].contiguous()
    shift_input_ids = input_ids_batch[:, 1:].contiguous()
    target_log_probs = shift_log_probs.gather(2, shift_input_ids.unsqueeze(-1)).squeeze(-1)
    # sum_log_probs = torch.sum([target_log_probs[0, i - 2:].sum() for i in response_start_idx_batch])
    sum_log_probs = torch.tensor(0.0, device=model.device)
    for idx, start_idx in enumerate(response_start_idx_batch):
        sum_log_probs += target_log_probs[idx, start_idx - 1:].sum()
    model.zero_grad()

    # last_layer_grad = None
    # def last_layer_hook(grad):
    #     nonlocal last_layer_grad
    #     last_layer_grad = grad.clone()
    #     # print(last_layer_grad.shape)
    # handle = model.lm_head.weight.register_hook(last_layer_hook)
    #
    sum_log_probs.backward(retain_graph=True)
    # handle.remove()

    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().flatten())

    # grads = last_layer_grad.detach().flatten()

    if keep_torch_tensor:
        grad_vector = torch.cat(grads) if type(grads) is list else grads
    else:
        grad_vector = (torch.cat(grads) if type(grads) is list else grads).cpu().numpy()
    return grad_vector

def process_dataset_and_compute_gradients(model, tokenizer, dataset, num_samples=3, output_dir='results'):
    """
    Process the dataset, compute gradients, compute metrics, and save results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, sample in enumerate(tqdm(dataset, desc="Processing Dataset")):
        if idx >= num_samples:
            break
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        prompt_text = prompt
        chosen_text = prompt_text + chosen
        rejected_text = prompt_text + rejected
        prompt_input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
        chosen_input_ids = tokenizer.encode(chosen_text, return_tensors='pt')
        rejected_input_ids = tokenizer.encode(rejected_text, return_tensors='pt')
        response_start_idx = prompt_input_ids.size(1)
        pos_grads, pos_tokens = compute_per_token_gradients(model, tokenizer, chosen_input_ids, response_start_idx)
        neg_grads, neg_tokens = compute_per_token_gradients(model, tokenizer, rejected_input_ids, response_start_idx)
        # pos_response_grad = compute_per_response_gradients(model, tokenizer, chosen_input_ids, response_start_idx)
        # neg_response_grad = compute_per_response_gradients(model, tokenizer, rejected_input_ids, response_start_idx)
        pos_grads_matrix = np.stack(pos_grads)
        neg_grads_matrix = np.stack(neg_grads)
        cosine_sim_matrix = cosine_similarity(pos_grads_matrix, neg_grads_matrix)
        cosine_sim_matrix_path = os.path.join(output_dir, f'sample_{idx}_cosine_sim_matrix.npy')
        np.save(cosine_sim_matrix_path, cosine_sim_matrix)
        # cosine_sim_response = cosine_similarity(pos_response_grad.reshape(1, -1), neg_response_grad.reshape(1, -1))[0, 0]
        # pos_response_grad_norm = np.linalg.norm(pos_response_grad)
        # neg_response_grad_norm = np.linalg.norm(neg_response_grad)
        # correlation_min = (np.dot(pos_response_grad, neg_response_grad) /
        #                    min(np.linalg.norm(pos_response_grad)**2,
        #                        np.linalg.norm(neg_response_grad)**2))
        metadata = {
            'sample_idx': idx,
            'prompt_text': prompt_text,
            'pos_tokens': pos_tokens,
            'neg_tokens': neg_tokens,
            # 'cosine_sim_response': float(cosine_sim_response),
            # 'correlation_min': float(correlation_min),
            # 'pos_response_grad_norm': float(pos_response_grad_norm),
            # 'neg_response_grad_norm': float(neg_response_grad_norm),
        }
        metadata_path = os.path.join(output_dir, f'sample_{idx}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        del pos_grads, neg_grads, pos_grads_matrix, neg_grads_matrix, cosine_sim_matrix
        torch_empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='gpt2', help="Model name or path")
    parser.add_argument("--dataset_name", type=str, default='data/senti_data/tweet_sentiment_preference_long')
    parser.add_argument("--eval_split", type=str, default='test', help="Evaluation split")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to process")
    parser.add_argument("--output_dir", type=str, default='results', help="Output directory for results")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset_name, split=args.eval_split)
    process_dataset_and_compute_gradients(model, tokenizer, dataset, args.num_samples, output_dir=args.output_dir)

if __name__ == '__main__':
    main()