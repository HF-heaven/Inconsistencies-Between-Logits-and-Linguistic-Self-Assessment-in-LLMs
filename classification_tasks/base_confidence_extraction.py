import json
import csv
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
from huggingface_hub import login


def extract_confidence(text):
    match = re.findall(r"Confidence:\s*(\d+)", text)
    if match:
        return int(match[-1])
    else:
        return "N/A"


def main():
    parser = argparse.ArgumentParser(description="Extract model confidence for a given model, task, and prompt file.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID, e.g., Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--prompt_file", type=str, required=True, help="Prompt file name, e.g., prompts_base_new_56.json")
    parser.add_argument("--task", type=str, required=True, help="Task name, e.g., CoLA")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (optional)")
    parser.add_argument("--align_logits_hint", action="store_true",
                        help="Append 'Make sure the confidence you report reflects the softmax probabilities from your final layer logits.' to each prompt")
    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    model_id = args.model_id
    prompt_file = args.prompt_file
    task = args.task

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
    input_file_path = os.path.join("/new_models_data", task, prompt_file)
    prompt_base = os.path.splitext(prompt_file)[0]
    output_file_name = f"{model_name}_{prompt_base}.csv"
    output_file_path = os.path.join("/new_models_data", task, "base", output_file_name)

    if os.path.exists(output_file_path):
        print(f"Output file {output_file_path} already exists. Skipping...")
        return

    with open(input_file_path, "r", encoding="utf-8") as file:
        prompts = json.load(file)

    data = [["Answer", "Logits", "Confidence"]]
    print(f"\nProcessing model: {model_id}, task: {task}, prompt file: {prompt_file}")

    for input_text in tqdm(prompts):
        # Add alignment hint if requested
        if args.align_logits_hint:
            input_text += " Make sure the confidence you report reflects the softmax probabilities from your final layer logits."
            output_file_name = f"{output_file_name.split('.')[0]}_withAlignment.csv"
        input_text += " ("

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        generate_length = 16

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
        generated_text = tokenizer.decode(generated_sequences[0][len(input_ids[0]):], skip_special_tokens=True)

        answer_token = generated_sequences[0][len(input_ids[0]) + position_of_ans]
        answer = tokenizer.decode([answer_token], skip_special_tokens=True)

        all_logits = generated_outputs.scores
        probabilities = torch.nn.functional.softmax(all_logits[position_of_ans][0], dim=0)
        probability_answer = probabilities[answer_token].cpu().item()

        confidence = extract_confidence(generated_text)
        data.append([answer, probability_answer, confidence])

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"Saved results to {output_file_path}")


if __name__ == "__main__":
    main()
