# save as: run_ins_no_conf.py
import os
import re
import json
import csv
import math
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# -----------------------
# Defaults / switches
# -----------------------
MAX_SAMPLES = 200
MAX_NEW_TOKENS = 8
DO_SAMPLE = False
PRINT_TOKEN_DETAILS = True
MAX_TOKENS_TO_PRINT = 100
UNIFORM_TOL = 0.0  # treat near-uniform label probs as N/A

# -----------------------
# Task groups: decide option count by task name
# -----------------------
TWO_OPTION_TASKS = {"cause_and_effect", "CoLA", "QNLI", "QQP"}
FOUR_OPTION_TASKS = {"conceptual_combinations", "ruin_names", "temporal_sequences", "MMLU"}


def get_labels_from_targets(target_choices):
    """
    Try to derive labels like (a) (b) ... from multiple_choice_targets if present.
    """
    labels = []
    for t in target_choices:
        m = re.match(r'^\s*\(([a-d])\)\s+', t.strip(), flags=re.IGNORECASE)
        if m:
            labels.append(m.group(1).lower())
    return labels if labels else target_choices


def normalize_text_for_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace('confidence', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s


def extract_choice_label_robust(full_response: str):
    """
    Robustly parse 'a'/'b'/'c'/'d' from free-form model output.
    """
    if not full_response:
        return None
    t = normalize_text_for_label(full_response)
    m = re.search(r'\(\s*([abcd])[^a-z0-9]{0,5}\)', t)
    if m:
        return m.group(1)
    m = re.search(r'\b(answer|option|choice|is)\b[^a-d]{0,10}\b([abcd])\b', t[:120])
    if m:
        return m.group(2)
    head = t[:60]
    m = re.search(r'(?<![a-z])([abcd])(?![a-z])', head)
    if m:
        return m.group(1)
    return None


@torch.no_grad()
def continuation_logprob_with_details(model, tokenizer, input_ids, continuation_text: str):
    """
    Compute log P(continuation | prompt) and return token-level details.
    """
    device = input_ids.device
    cont_ids = tokenizer.encode(continuation_text, add_special_tokens=False, return_tensors='pt').to(device)
    full_ids = torch.cat([input_ids, cont_ids], dim=1)

    outputs = model(full_ids)
    logits = outputs.logits  # [1, seq_len, vocab]
    input_len = input_ids.shape[1]
    cont_len = cont_ids.shape[1]

    step_logits = logits[:, input_len-1: input_len+cont_len-1, :]
    log_probs = torch.log_softmax(step_logits, dim=-1)

    target_ids = full_ids[:, input_len: input_len+cont_len]
    token_logprobs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    total_logprob = token_logprobs.sum(dim=1).item()

    ids = target_ids[0].tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    logs = token_logprobs[0].tolist()
    probs = [math.exp(x) for x in logs]

    return {
        "tokens": toks,
        "ids": ids,
        "logprobs": logs,
        "probs": probs,
        "total_logprob": total_logprob,
        "text": continuation_text,
    }


def decide_allowed_labels(task_name: str, task_item: dict | None):
    """
    Decide allowed label set (2-choice or 4-choice) by task name; if task data
    provides multiple_choice_targets with explicit (a)-(d), honor that.
    """
    if task_name in TWO_OPTION_TASKS:
        base = ['a', 'b']
    elif task_name in FOUR_OPTION_TASKS:
        base = ['a', 'b', 'c', 'd']
    else:
        # Fallback to 4-choice unless multiple_choice_targets says otherwise
        base = ['a', 'b', 'c', 'd']

    try:
        if task_item and "multiple_choice_targets" in task_item:
            inferred = get_labels_from_targets(task_item["multiple_choice_targets"])
            if all(x in ['a', 'b', 'c', 'd'] for x in inferred):
                base = [x for x in base if x in inferred] or base
    except Exception:
        pass

    return base


def main():
    parser = argparse.ArgumentParser(description="Unified 2/4-choice MCQ runner (no linguistic confidence).")
    parser.add_argument("--model_id", required=True, type=str,
                        help="HF model id, e.g., Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--task", required=True, type=str,
                        help="Task name, decides option count if known (2 or 4).")
    parser.add_argument("--prompt_file", required=True, type=str,
                        help="Prompt JSON file under data_root/{task}/")
    parser.add_argument("--data_root", type=str, default="data_formatted",
                        help="Data root directory (default: data_formatted)")
    parser.add_argument("--token", type=str, default=None,
                        help="(Optional) HF access token; or set env HF_TOKEN")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES,
                        help="Max number of samples to run (default: 200)")
    args = parser.parse_args()

    # Login
    token = args.token or os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)

    # Load model & tokenizer
    print(f"\n=== Loading model: {args.model_id} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=False,
        output_attentions=False
    )

    # Paths
    task_name = args.task
    input_file_path = os.path.join(args.data_root, task_name, args.prompt_file)
    task_data_path = os.path.join(args.data_root, task_name, "data.json")

    # Output dir: instruct1 if file name contains 'instruct1', else base
    output_subfolder = "instruct1" if "instruct1" in args.prompt_file else "base"
    os.makedirs(os.path.join(args.data_root, task_name, output_subfolder), exist_ok=True)
    file_suffix = os.path.splitext(os.path.basename(args.prompt_file))[0].replace("prompts_instruct_", "")
    model_name = args.model_id.split("/")[-1]
    output_file_path = os.path.join(args.data_root, task_name, output_subfolder, f"{model_name}_{file_suffix}.csv")

    # Load data
    with open(task_data_path, "r", encoding="utf-8") as f:
        task_data_all = json.load(f)
    task_split = task_data_all["validation"] if isinstance(task_data_all, dict) and "validation" in task_data_all else task_data_all

    with open(input_file_path, "r", encoding="utf-8") as f:
        prompts_all = json.load(f)

    n = min(args.max_samples, len(prompts_all), len(task_split))
    prompts = prompts_all[:n]
    task_items = task_split[:n]
    print(f"\n--- Task: {task_name} | Prompt file: {args.prompt_file} | Samples: {n} ---")

    # Setup candidate variants
    candidate_variants = {
        'a': [' (a)', '(a)', '(a', ' a', 'a', 'answer: a', 'choice: a', 'is a'],
        'b': [' (b)', '(b)', '(b', ' b', 'b', 'answer: b', 'choice: b', 'is b'],
        'c': [' (c)', '(c)', '(c', ' c', 'c', 'answer: c', 'choice: c', 'is c'],
        'd': [' (d)', '(d)', '(d', ' d', 'd', 'answer: d', 'choice: d', 'is d'],
    }

    data = [["Answer", "Logits"]]  # 'Logits' column holds P(answer) (i.e., probability)

    for idx, conv in enumerate(prompts):
        print(f"\n================ SAMPLE {idx + 1}/{n} ================")
        task_item = task_items[idx]

        # Build input via chat template
        try:
            input_ids = tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
        except Exception as e:
            print(f"[WARN] apply_chat_template failed at idx {idx}: {e}")
            data.append(['N/A', 'N/A'])
            continue

        # EOS setup
        terminators = []
        if tokenizer.eos_token_id is not None:
            terminators.append(tokenizer.eos_token_id)
        try:
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != tokenizer.eos_token_id:
                terminators.append(eot_id)
        except Exception:
            pass
        terminators = terminators or None

        # Generate
        try:
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=terminators,
                do_sample=DO_SAMPLE,
                return_dict_in_generate=True,
                output_scores=True
            )
            response_tokens = outputs.sequences[0][input_ids.shape[-1]:]
            full_response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[ERROR] generate failed at idx {idx}: {e}")
            data.append(['N/A', 'N/A'])
            continue

        # Debug prints
        if PRINT_TOKEN_DETAILS:
            try:
                prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            except Exception:
                prompt_text = "<decode failed>"
            print("\n[DEBUG] === PROMPT (after chat template) ===")
            print(prompt_text)
            print("\n[DEBUG] === GENERATED ANSWER (decoded) ===")
            print(full_response if len(full_response) <= 2000 else full_response[:2000] + "... <truncated>")
            try:
                gen_ids = response_tokens.tolist()
                gen_toks = tokenizer.convert_ids_to_tokens(gen_ids)
                step_scores = outputs.scores
                print("\n[DEBUG] === GENERATED TOKENS (index | token | id | prob at step) ===")
                to_print = min(len(gen_ids), MAX_TOKENS_TO_PRINT)
                for i in range(to_print):
                    logits_step = step_scores[i][0].float()
                    probs_step = torch.softmax(logits_step, dim=-1)
                    tid = gen_ids[i]
                    p = probs_step[tid].item()
                    tk = gen_toks[i]
                    safe_tk = tk.replace("\n", "\\n").replace("\t", "\\t")
                    print(f"{i:03d}: {safe_tk:>12s} | {tid:>6d} | {p:.6f}")
                if len(gen_ids) > to_print:
                    print(f"... ({len(gen_ids)-to_print} more tokens truncated)")
            except Exception as e:
                print(f"[DEBUG] failed to print generated token probs: {e}")

        # Decide labels (2 or 4)
        allowed_labels = decide_allowed_labels(task_name, task_item)

        # Score each candidate by conditional logprob; use best variant per label
        label_logprobs = {}
        label_best_details = {}
        try:
            for lab, variants in candidate_variants.items():
                if lab not in allowed_labels:
                    continue
                best_lp = -1e30
                best_detail = None
                for v in variants:
                    detail = continuation_logprob_with_details(model, tokenizer, input_ids, v)
                    lp = detail["total_logprob"]
                    if lp > best_lp:
                        best_lp = lp
                        best_detail = detail
                label_logprobs[lab] = best_lp
                label_best_details[lab] = best_detail

            if not label_logprobs:
                data.append(['N/A', 'N/A'])
                continue

            # Softmax over labels -> probabilities
            max_lp = max(label_logprobs.values())
            exp_sum = sum(math.exp(lp - max_lp) for lp in label_logprobs.values())
            label_probs = {lab: math.exp(lp - max_lp) / exp_sum for lab, lp in label_logprobs.items()}
            # Keep only allowed_labels and (re)normalize
            label_probs = {k: v for k, v in label_probs.items() if k in allowed_labels}
            if len(label_probs) < 2:
                data.append(['N/A', 'N/A'])
                continue
            renorm = sum(label_probs.values())
            label_probs = {k: v / renorm for k, v in label_probs.items()}

            # Uniform check (e.g., 0.5/0.5 or 0.25x4)
            L = len(label_probs)
            uniform_p = 1.0 / L
            if max(abs(p - uniform_p) for p in label_probs.values()) <= UNIFORM_TOL:
                print(f"[UNIFORM] near-uniform probs -> { {k: round(v,3) for k,v in label_probs.items()} } ; write N/A")
                data.append(['N/A', 'N/A'])
                continue

            # Parse label from text; fall back to argmax
            answer_label = extract_choice_label_robust(full_response)
            best_label = max(label_probs, key=label_probs.get)
            if (answer_label not in label_probs) or (label_probs[answer_label] < label_probs[best_label]):
                answer_label = best_label

            probability_answer = label_probs.get(answer_label, 'N/A')

            # Debug
            debug_prob_str = " | ".join(f"{k}:{label_probs.get(k,0):.3f}" for k in ['a','b','c','d'] if k in label_probs)
            debug_logp_str = " | ".join(f"{k}:{label_logprobs.get(k):.2f}" for k in ['a','b','c','d'] if k in label_logprobs)
            print(f"\n[PROBS] {debug_prob_str} -> pick: {answer_label} (p={probability_answer if probability_answer=='N/A' else f'{probability_answer:.3f}'})")
            print(f"[LOGP ] {debug_logp_str}")

            data.append([answer_label, probability_answer])

        except Exception as e:
            print(f"[WARN] scoring failed at idx {idx}: {e}")
            data.append(['N/A', 'N/A'])

    with open(output_file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print(f"\nSaved results to: {output_file_path}")


if __name__ == "__main__":
    main()
