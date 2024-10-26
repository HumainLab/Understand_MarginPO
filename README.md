# A Common Pitfall of Margin-based Language Model Alignment: Gradient Entanglement

## Dataset

The Sentiment Analysis dataset can be downloded from [here](https://oregonstate.box.com/s/jugqojqhpifsev9usktq32elomijfgzf).
It contains three different datasets: `long-suffix`, `short-suffix`, and `1t`.
It should be extracted to `data/senti_data/`.

The TLDR dataset is loaded from huggingface datasets. The dataset can be found [here](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons).

## Training

The scripts to run preference optimization on the TLDR task with Meta-Llama-3-8B-Instruct on different loss types.

Use `export WANDB_PROJECT=YOUR_PROJECT_NAME` to set your wandb project name.`

```shell
for loss_type in sigmoid ipo
do
export exp_name="l3-8b-tldr-dpo-${loss_type}"
accelerate launch --num_processes 2 dpo.py recipes/dpo_tldr.yaml \
  --model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
  --run_name=$exp_name --output_dir="outputs/${exp_name}" \
  --loss_type=$loss_type
done

for loss_type in pdpo rdpo
do
export exp_name="l3-8b-tldr-dpo-${loss_type}"
accelerate launch --num_processes 2 dpo-custom.py recipes/dpo_tldr.yaml \
 --model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
 --run_name=$exp_name --output_dir="outputs/${exp_name}" \
 --loss_type=$loss_type
done

export exp_name="l3-8b-tldr-simpo"
accelerate launch --num_processes 2 cpo.py recipes/dpo_tldr.yaml \
  --model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
  --run_name=$exp_name --output_dir="outputs/${exp_name}" \
  --loss_type=simpo --cpo_alpha=0

 for loss_type in rrhf
 do
 export exp_name="l3-8b-tldr-cpo-${loss_type}"
 accelerate launch --num_processes 2 cpo-custom.py recipes/dpo_tldr.yaml \
   --model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
   --run_name=$exp_name --output_dir="outputs/${exp_name}" \
   --loss_type=$loss_type
 done

```

The scripts to run preference optimization on the sentiment task with GPT-2 on different datasets.

```shell
# Run SFT on sentiment datasets
for ds in long-suffix-only short-suffix-only 1t # long-suffix short-suffix 1t
do
  exp_name="g2-sent-${ds}-sft"
  python sft.py recipes/sft_sentiment.yaml \
  --dataset_name="data/senti_data/${ds}/sft" \
  --output_dir=outputs/$exp_name \
  --run_name=$exp_name
done

# Run DPO on sentiment datasets
for ds in long-suffix-only short-suffix-only 1t # long-suffix short-suffix 1t
do
for loss_type in sigmoid
do
exp_name="g2-sent-${ds}-${loss_type}-dpo"
python dpo.py recipes/dpo_sentiment.yaml \
  --dataset_name="data/senti_data/${ds}/preference" \
  --model_name_or_path="outputs/g2-sent-${ds}-sft/checkpoint-64" \
  --run_name=$exp_name \
  --output_dir="outputs/${exp_name}" \
  --loss_type=$loss_type
done
done
```

## Gradient Evaluation

### Gradient Cosine Similarity

Evaluate Full Gradient correlation for chosen and rejected logP for gpt-2

```shell
for ds in long-suffix-only short-suffix-only 1t
do
python analysis/grad_plot.py --model_name_or_path "outputs/g2-sent-$ds-sigmoid-dpo" \
--dataset_name "data/senti_data/$ds/preference" --full_parameters \
--output_csv "plots/grad/g2-sent-$ds-sigmoid.csv"
done
```

Evaluate LoRA Gradient correlation for chosen and rejected logP for other models. This enables DDP inference for faster gradient evaluation on multiple GPUs.

```shell
for exp_name in outputs/l3-8b-*; do
    accelerate launch --num_processes 2 analysis/grad_plot.py \
      --model_name_or_path $exp_name --output_csv plots/grad/$(basename $exp_name).csv
done
```

### Token-level Gradient Heatmap

The following script will compute token-level gradient heatmaps and save results in `.npy` files.
    
```shell
for ds in long-suffix # long-suffix-only short-suffix-only 1t
do
    for rlhf in dpo # cpo
    do
        for loss_type in sigmoid # ipo
        do
            for model in outputs/g2-sent-$ds-$loss_type-$rlhf/checkpoint-*
            do
                # get checkpoint number from model path (split by '/' -> split by '-' -> take last element)
                checkpoint=$(echo $model | tr '/' '\n' | grep checkpoint | tr '-' '\n' | tail -n 1)
                python analysis/grad_heatmap_1.py --model_name_or_path=$model \
                    --dataset_name=data/senti_data/$ds/preference \
                    --output_dir=plots/heatmaps/g2-sent-$ds-$loss_type-$rlhf/$checkpoint
            done
        done
    done
done
```