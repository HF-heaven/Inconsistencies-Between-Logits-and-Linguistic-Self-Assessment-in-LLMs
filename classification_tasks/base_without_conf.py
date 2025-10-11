import json
import csv
import os
import re
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run base model inference (without linguistic confidence)."
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face model ID, e.g., meta-llama/Llama-2-13b-hf")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Prompt file name, e.g., prompts_base_withoutConfidence.json")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name, e.g., CoLA or MMLU")
    parser.add_argument("--token", type=str, default=None,
                        help="(Optional) Hugging Face token")
    parser.add_argument("--data_root", type=str, default="/mnt/ssd/bingquan/llama/data_formatted",
                        help="Root directory for task data")
    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    start_time = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_id = args.model_id
    prompt_file = args.prompt_file
    task = args.task
    data_root = args.data_root

    print(f"\nLoading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_attentions=True
    )
    model.config.return_dict = True

    model_name = model_id.split("/")[-1]
    input_file_path = os.path.join(data_root, task, prompt_file)
    prompt_base = os.path.splitext(prompt_file)[0]
    output_file_name = f"{model_name}_{prompt_base}.csv"
    output_file_path = os.path.join(data_root, task, "base", output_file_name)

    if os.path.exists(output_file_path):
        print(f"Output file {output_file_path} already exists. Skipping...")
        return

    with open(input_file_path, "r", encoding="utf-8") as file:
        prompts = json.load(file)

    data = [["Answer", "Logits"]]
    print(f"\nProcessing model: {model_id}, task: {task}, prompt file: {prompt_file}")

    for input_text in tqdm(prompts):
        # Append “(” to trigger generation
        input_text += " ("

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        generate_length = 5

        generated_outputs = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids).to(model.device),
            max_new_tokens=generate_length,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )

        position_of_ans = 0
        generated_sequences = generated_outputs.sequences
        answer_token = generated_sequences[0][len(input_ids[0]) + position_of_ans]
        answer = tokenizer.decode([answer_token], skip_special_tokens=True)

        all_logits = generated_outputs.scores
        probabilities = torch.nn.functional.softmax(all_logits[position_of_ans][0], dim=0)
        probability_answer = probabilities[answer_token].cpu().item()

        data.append([answer, probability_answer])

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"Saved results to {output_file_path}")
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
