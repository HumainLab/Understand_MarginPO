import os
from accelerate import PartialState
from trl.commands.cli_utils import SFTScriptArguments
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map, apply_chat_template,
)

from utils.configs import H4ArgumentParser


def formatting_prompts_func(example):
    return f"{example['prompt']} {example['label']}"


if __name__ == "__main__":
    parser = H4ArgumentParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    tokenizer_name_map = {
        "mistralai/Mistral-7B-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
    tokenizer_name = tokenizer_name_map.get(model_config.model_name_or_path, model_config.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    if os.path.exists(args.dataset_name):
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name)


    with PartialState().local_main_process_first():
        if "ultrafeedback_binarized" in args.dataset_name:
            dataset = dataset.remove_columns(["chosen", "rejected", "prompt"])
            dataset = dataset.map(
                apply_chat_template, num_proc=training_args.dataset_num_proc, fn_kwargs={"tokenizer": tokenizer}
            )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
        formatting_func=formatting_prompts_func if training_args.dataset_text_field is None else None,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
