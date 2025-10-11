#!/usr/bin/env bash
set -euo pipefail

##### 缓存/环境（可保持与你机器一致） #####
export HF_HOME="your_hf_home_path"
export TRANSFORMERS_CACHE="your_transformer_cache_path"
export HF_DATASETS_CACHE="your_datasets_cache_path"

export NCCL_DEBUG=warn
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

##### W&B #####
run_id="decent-field-17" # For task trivia_qa
# run_id="deft-monkey-51" # For task coqa
echo "[INFO] WANDB_RUN_ID: $run_id"
export WANDB_PROJECT="nlg_uncertainty"
export WANDB_RESUME="allow"

MODELS=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-3B-Instruct"

    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-8B-Base"
    "Qwen/Qwen3-14B-Base"
    "Qwen/Qwen3-4B-Instruct-2507"

    "mistralai/Mistral-7B-v0.1"
    "mistral-community/Mistral-7B-v0.2"
    "mistralai/Mistral-7B-v0.3"
    "mistralai/Mistral-Nemo-Base-2407"
    "mistralai/Mistral-7B-Instruct-v0.1"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Mistral-Nemo-Instruct-2407"
    "mistralai/Mixtral-8x7B-v0.1"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)

NUM_SHARDS=2
NUM_GENERATIONS_PER_PROMPT=5
FRACTION_OF_DATA=0.02
TEMPERATURE=0.5
NUM_BEAMS=1
TOP_P=1.0

mkdir -p logs

for model in "${MODELS[@]}"; do
  tag="$(basename "$model")"
  logdir="logs/${tag}"
  mkdir -p "$logdir"

  echo "[INFO] ==== Start model: ${model} | run_id=${run_id} ===="
  
  shard_run_id="${run_id}-shard"
  export WANDB_RUN_ID="${shard_run_id}"
  echo "[INFO] Launch shard on GPU (WANDB_RUN_ID=$WANDB_RUN_ID)"
   nohup python generate.py \
    --num_generations_per_prompt "${NUM_GENERATIONS_PER_PROMPT}" \
    --model "${model}" \
    --fraction_of_data_to_use "${FRACTION_OF_DATA}" \
    --run_id "${run_id}" \
    --temperature "${TEMPERATURE}" \
    --num_beams "${NUM_BEAMS}" \
    --top_p "${TOP_P}" \
    > "${logdir}/gen.log" 2>&1 &
  
  wait
  echo "[INFO] ==== Finished model generation: ${model} ===="

    python clean_generated_strings.py \
      --generation_model "$model" \
      --run_id "$run_id"
    python get_semantic_similarities.py \
      --generation_model "$model" \
      --run_id "$run_id"
    python get_likelihoods.py \
      --evaluation_model "$model" \
      --generation_model "$model" \
      --run_id "$run_id"

  python get_prompting_based_uncertainty.py \
    --generation_model "$model" \
    --run_id_for_few_shot_prompt "$run_id" \
    --run_id_for_evaluation "$run_id" \
    > "${logdir}/prompting_based_uncertainty.log" 2>&1 &

  python compute_confidence_measure.py \
    --generation_model "$model" \
    --evaluation_model "$model" \
    --run_id "$run_id" \
    > "${logdir}/output_entropy.log" 2>&1

  echo "[INFO] ==== Finished model: ${model} ===="
done

echo "[INFO] All models finished."
