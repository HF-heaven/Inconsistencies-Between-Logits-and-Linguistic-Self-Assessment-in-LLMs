import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize
import re
import random
from typing import Optional
import accelerate
import config
import datasets
import evaluate
import numpy as np
import torch
import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
login(token="your_hf_huggingface_token")
from rouge_score import rouge_scorer, scoring
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--num_beams', type=int, default='5')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--with_conf', choices=["True", "False"], default="True")
args = parser.parse_args()

args.with_conf = args.with_conf == "True"

wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')

run_name = wandb.run.name

device = 'cuda'

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

model = AutoModelForCausalLM.from_pretrained(f"{args.model}",
                                             torch_dtype=torch.float16,
                                             cache_dir=config.hf_cache_dir,
                                             device_map="auto")

if args.model == 'opt-30b':
    accelerate.dispatch_model(model, device_map=config.device_map)

tokenizer = AutoTokenizer.from_pretrained(f"{args.model}", use_fast=False, cache_dir=config.hf_cache_dir)

opt_models = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

if args.dataset == 'coqa':
    dataset = datasets.load_from_disk(f'{config.output_dir}/coqa_dataset')
    id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
elif 'trivia_qa' in args.dataset:
    dataset = datasets.load_from_disk(f'{config.output_dir}/{args.dataset}')

if args.fraction_of_data_to_use < 1.0:
    train_dataset = dataset.train_test_split(train_size=200, seed=seed_value)['train']
else:
    train_dataset = dataset

def add_global_instruction_and_conf(text: str,
                                    seed: Optional[int] = 42,
                                    conf_min: int = 0,
                                    conf_max: int = 10) -> str:
    """
    Insert a global instruction at the beginning of the text, and rewrite all answer segments to include a confidence score.
    """
    if seed is not None:
        random.seed(seed)

    # 1) Modify all A: <answer> to 'A: <answer> Confidence: <int>.'
    def a_repl(m):
        ans = m.group(1).strip()
        conf = random.randint(conf_min, conf_max)  # 闭区间 [conf_min, conf_max]
        return f"A: {ans} Confidence: {conf}. \n"

    rewritten = re.sub(r"A:\s*([^Q]+?)(?=(?:Q:|$))", a_repl, text, flags=re.DOTALL)

    # 2) Insert a global instruction before the first Q:
    instruction = (
        "\n\nRead the question, provide your answer and your confidence in this answer. "
        "Note: The confidence indicates how likely you think your answer is true. "
        "0 means you don't believe this answer at all, 10 means you trust this answer must be true. "
        "Use the following format to answer: A: <your answer>. "
        "Confidence: <Your confidence level, please only include the numerical number in the range of 0-10>.\n"
        "Only the answer and confidence, don’t give me the explanation.\n"
    )
    # repeat_instruction = "You must ONLY answer in the following format (do not output anything else): \nA: <your answer>. Confidence: <number>\n"

    first_q = re.search(r"\bQ:\s*", rewritten)
    if first_q:
        insert_pos = first_q.start()
        return rewritten[:insert_pos] + instruction + rewritten[insert_pos:]
    else:
        # If there's no question found, just prepend the instruction
        return rewritten + instruction
    

def encode(examples):
    # print("--------------------------------")
    if args.with_conf:
        story = add_global_instruction_and_conf(examples['story'])
        text = story + 'Q: ' + examples['question'] + ' A:'
    else:
        text = examples['story'] + ' Q: ' + examples['question'] + ' A:'
    # print("After adding instruction:", text)
    return tokenizer(text, truncation=False, padding=False)


def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def extract_confidence(text):
    match = re.search(r'Confidence:\s*(\d+)', text)
    return int(match.group(1)) if match else 'N/A'

if args.dataset == 'coqa':
    questions = encode_and_format_dataset(train_dataset)
elif 'trivia_qa' in args.dataset:
    questions = train_dataset

dataloader = torch.utils.data.DataLoader(questions, batch_size=1)

period_token_id = tokenizer('. ')['input_ids'][1]
eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
question_framing_ids = []
for eos_token in eos_tokens:
    input_ids = tokenizer(eos_token, add_special_tokens=False)["input_ids"]
    if len(input_ids) == 1:
        question_framing_ids.append([input_ids[0]])
    else:
        question_framing_ids.append([input_ids[1]])
squad_metric = evaluate.load("squad")
rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")


def get_generations(model, dataloader, number_of_generations):
    """For a given model, produce a number of generation """

    with torch.no_grad():
        max_length_of_generated_sequence = 50
        sequences = []
        for batch in tqdm.tqdm(dataloader):
            # input_ids = torch.cat(batch['input_ids']).to(device).reshape(
            #     1, -1) if args.dataset == 'trivia_qa' else batch['input_ids'].to(device)
            input_ids = batch['input_ids'].to(device)
            # print("Input IDs:", input_ids)
            # print("Input prompt:", tokenizer.decode(input_ids[0], skip_special_tokens=True))
            # attention_mask = torch.cat(batch['attention_mask']).to(device).reshape(
            #     1, -1) if args.dataset == 'trivia_qa' else batch['attention_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if args.decoding_method == 'beam_search':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=5,
                                                        num_return_sequences=2,
                                                        do_sample=False,
                                                        # max_length=input_ids.shape[1] +
                                                        # max_length_of_generated_sequence,
                                                        max_new_tokens=max_length_of_generated_sequence,
                                                        attention_mask=attention_mask,
                                                        pad_token_id=tokenizer.pad_token_id,
                                                        eos_token_id=period_token_id,
                                                        bad_words_ids=question_framing_ids)
            elif args.decoding_method == 'greedy':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=1,
                                                        do_sample=False,
                                                        # max_length=input_ids.shape[1] +
                                                        # max_length_of_generated_sequence,
                                                        max_new_tokens=max_length_of_generated_sequence,
                                                        attention_mask=attention_mask,
                                                        pad_token_id=tokenizer.pad_token_id,
                                                        eos_token_id=period_token_id,
                                                        bad_words_ids=question_framing_ids)

            input_length = input_ids.shape[1] if args.dataset == 'trivia_qa' else batch['input_ids'].shape[1]
            generations = torch.ones((number_of_generations, input_length + max_length_of_generated_sequence + 10),
                                     dtype=torch.long,
                                     device=device)
            

            if args.with_conf:
                pattern = re.compile(r'Confidence:\s*(\d+)')
                # max_attempts = 15

                generated_texts = []
                # can_match = False
                # first_gen = torch.ones((number_of_generations, input_length + max_length_of_generated_sequence),
                #                      dtype=torch.long,
                #                      device=device)
                # first_texts = []

                for i in range(number_of_generations):
                    # print("bad_words_ids=None")
                    generation = model.generate(
                            input_ids,
                            do_sample=True,
                            num_return_sequences=1,
                            num_beams=args.num_beams,
                            # max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                            max_new_tokens=max_length_of_generated_sequence,
                            attention_mask=attention_mask,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=period_token_id,
                            temperature=args.temperature,
                            bad_words_ids=None,
                            top_p=args.top_p,
                        )
                    generated_text = tokenizer.decode(generation[0][input_length:], skip_special_tokens=True)
                    match = pattern.search(generated_text)
                    if match:
                        # print(match.group(1), generated_text)
                        generations[i, :generation.shape[1]] = generation
                        generated_texts.append(generated_text)
                    else:
                        strings_to_filter_on = [
                            '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
                            'ANSWER:'
                        ]
                        cleaned_generated_text = generated_text
                        for string in strings_to_filter_on:
                            if string in generated_text:
                                cleaned_generated_text = generated_text.split(string)[0]
                        cleaned_generated_text += '. Confidence:'
                        cleaned_ids = tokenizer(cleaned_generated_text)['input_ids']
                        new_input_ids = torch.cat([input_ids[0], torch.tensor(cleaned_ids, device=device)]).unsqueeze(0)
                        if tokenizer.pad_token_id is not None:
                            new_attention_mask = (new_input_ids != tokenizer.pad_token_id).long()
                        else:
                            new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.long)
                        generation = model.generate(
                            new_input_ids,
                            do_sample=True,
                            num_return_sequences=1,
                            num_beams=args.num_beams,
                            # max_length=input_ids.shape[1] + max_length_of_generated_sequence + 5,
                            max_new_tokens=input_ids.shape[1] + max_length_of_generated_sequence + 10 - new_input_ids.shape[1],
                            attention_mask=new_attention_mask,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=period_token_id,
                            temperature=args.temperature,
                            bad_words_ids=question_framing_ids,
                            top_p=args.top_p,
                        )
                        generated_text = tokenizer.decode(generation[0][input_length:], skip_special_tokens=True)
                        generations[i, :generation.shape[1]] = generation
                        generated_texts.append(generated_text)
                        # print("After cleaning:", generated_text)


                    # if i > 0 and not can_match:
                    #     max_attempts = 1
                    # for attempt in range(max_attempts):
                    #     generation = model.generate(
                    #         input_ids,
                    #         do_sample=True,
                    #         num_return_sequences=1,
                    #         num_beams=args.num_beams,
                    #         max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                    #         eos_token_id=period_token_id,
                    #         temperature=args.temperature,
                    #         bad_words_ids=question_framing_ids,
                    #         top_p=args.top_p,
                    #     )
                    #     generated_text = tokenizer.decode(generation[0][input_length:], skip_special_tokens=True)
                    #     match = pattern.search(generated_text)
                    #     if i == 0 and attempt < number_of_generations:
                    #         first_gen[attempt, :generation.shape[1]] = generation
                    #         first_texts.append(generated_text)
                    #     if match:
                    #         if i == 0:
                    #             can_match = True   
                    #             break
                    #         else:
                    #             break
                    # 
                    # if not can_match:
                    #     if i == 0:
                    #         generations = first_gen
                    #         generated_texts = first_texts
                    #         if max_attempts > number_of_generations:
                    #             break
                    #     else:
                    #         generations[i + max_attempts - 1, :generation.shape[1]] = generation
                    #         generated_texts.append(generated_text)
                    #         if len(generated_texts) == number_of_generations:
                    #             break
                    # else:
                    #     generations[i, :generation.shape[1]] = generation
                    #     generated_texts.append(generated_text)


            else:
                generated_texts = []
                for i in range(number_of_generations):
                    generation = model.generate(input_ids,
                                                do_sample=True,
                                                num_return_sequences=1,
                                                num_beams=args.num_beams,
                                                max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                                                eos_token_id=period_token_id,
                                                temperature=args.temperature,
                                                bad_words_ids=question_framing_ids,
                                                top_p=args.top_p)
                    generated_text = tokenizer.decode(generation[0][input_length:], skip_special_tokens=True)
                    print(f"Raw Generation {i + 1}: {generated_text}")
                    generations[i, :generation.shape[1]] = generation
                    generated_texts.append(generated_text)



            generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
            for i in range(generations.shape[0]):

                if args.dataset == 'coqa':
                    sequence_dict = {
                        'prompt': batch['input_ids'][i].to('cpu'),
                        'generations': generations[i].to('cpu'),
                        'id': batch['id'],
                        'question': id_to_question_mapping[batch['id'][0]]
                    }
                elif 'trivia_qa' in args.dataset:
                    few_shot_question = tokenizer.decode(input_ids[0])
                    question = few_shot_question.split('Question: ')[-1].split('Answer: ')[0]
                    sequence_dict = {
                        'prompt': input_ids[0],
                        'generations': generations[i],
                        'id': batch['question_id'],
                        'few_shot_question': tokenizer.decode(input_ids[0]),
                        'question': question
                    }

                # generated_texts = []
                # for generation in generations[i]:
                #     generated_texts.append(
                #         tokenizer.decode(generation[len(batch['input_ids'][i]):], skip_special_tokens=True))

                # for idx, gen_text in enumerate(generated_texts):
                #     print(f"Generation {idx + 1}: {gen_text}")
                #     print(f"Extracted Confidence: {extract_confidence(gen_text)}")

                sequence_dict['generated_texts'] = generated_texts
                sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                sequence_dict['most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[0][len(batch['input_ids'][i]):], skip_special_tokens=True)

                sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                sequence_dict['second_most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[1][len(batch['input_ids'][i]):], skip_special_tokens=True)

                sequence_dict['semantic_variability_reference_answers'] = batch[
                    'semantic_variability'] if 'semantic_variability' in batch else None
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
                for rouge_type in rouge_types:
                    if rouge_type in batch:
                        sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]

                    else:
                        sequence_dict[rouge_type + '_reference_answers'] = None

                    sequence_dict[rouge_type + '_to_target'] = 0.0

                sequence_dict['answer'] = batch['answer']['text'] if args.dataset == 'coqa' else batch['answer']
                sequence_dict['additional_answers'] = [x[0] for x in batch['additional_answers']
                                                      ] if args.dataset == 'coqa' else None

                sequence_dict['exact_match'] = 0.0

                reference_answers = batch['answer']['text'] + [x[0] for x in batch['additional_answers']
                                                              ] if args.dataset == 'coqa' else batch['answer']

                for answer in reference_answers:
                    predictions = [sequence_dict['most_likely_generation'].lstrip().split('. ')[0]]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                         references=references,
                                                         ignore_case=True,
                                                         ignore_punctuation=True)
                    sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
                    # rouge_results = rouge.compute(predictions=predictions, references=references)
                    
                    aggregator = scoring.BootstrapAggregator()
                    for pred, ref in zip(predictions, references):
                        # print("Pred:", pred, "Ref:", ref)
                        # print(scorer.score(ref, pred))
                        aggregator.add_scores(scorer.score(ref, pred))
                    rouge_results = aggregator.aggregate()
                    for rouge_type in rouge_types:
                        sequence_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type].mid.fmeasure,
                                                                       sequence_dict[rouge_type + '_to_target'])
                        # print(rouge_type, sequence_dict[rouge_type + '_to_target'])
                sequences.append(sequence_dict)

    return sequences


sequences = get_generations(model, dataloader, args.num_generations_per_prompt)

pathlib.Path(f'{config.output_dir}/sequences/' + run_name).mkdir(parents=True, exist_ok=True)

with open(f'{config.output_dir}/sequences/{run_name}/{args.model.split("/")[-1]}_generations.pkl', 'wb') as outfile:
    pickle.dump(sequences, outfile)