import argparse
import os
import pickle
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
login(token="your_hf_huggingface_token")
from tqdm import tqdm

import wandb



parser = argparse.ArgumentParser()
parser.add_argument('--evaluation_model', type=str, default='opt-350m')
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
args = parser.parse_args()

device = 'cuda'
import config

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

model = AutoModelForCausalLM.from_pretrained(args.evaluation_model,
                                             torch_dtype=torch.float16,
                                             cache_dir=config.data_dir,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.evaluation_model,
                                          use_fast=False,
                                          cache_dir=config.data_dir)

wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')

run_name = wandb.run.name

opt_models = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

with open(f'{config.output_dir}/sequences/{run_name}/{args.generation_model.split("/")[-1]}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'{config.output_dir}/sequences/{run_name}/{args.generation_model.split("/")[-1]}_generations_similarities.pkl', 'rb') as infile:
    similarities_dict = pickle.load(infile)


def get_neg_loglikelihoods(model, sequences):

    with torch.no_grad():
        result = []
        for sample in tqdm(sequences):
            result_dict = {}
            prompt = sample['prompt']
            
            if 'cleaned_generations' in sample:
                generations = sample['cleaned_generations'].to(device)
            else:
                generations = sample['generations'].to(device)
            id_ = sample['id']
            average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
            pointwise_mutual_information = torch.zeros((generations.shape[0],))
            sequence_embeddings = []

            for generation_index in range(generations.shape[0]):
                prompt = prompt[prompt != tokenizer.eos_token_id]
                generation_row = generations[generation_index]
                generation = generation_row[generation_row != tokenizer.eos_token_id]
                # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
                target_ids = generation.clone()
                target_ids[:len(prompt)] = -100
                model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
                generation_only = generation.clone()[len(prompt):]
                if len(generation_only) == 0:
                    generation_only = torch.tensor([tokenizer.eos_token_id], device=device)
                # print("generation_only", generation_only)
                unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                   labels=generation_only,
                                                   output_hidden_states=True)
                hidden_states = model_output['hidden_states']
                average_neg_log_likelihood = model_output['loss']

                average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']
                average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
                average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood
                neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
                neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * (
                    len(generation) - len(prompt))
                pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                    generation_index] + neg_unconditioned_log_likelihoods[generation_index]

                average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
                sequence_embeddings.append(average_of_last_layer_token_embeddings)

            if 'cleaned_most_likely_generation_ids' in sample:
                most_likely_generation = sample['cleaned_most_likely_generation_ids'].to(device)
            else:
                most_likely_generation = sample['most_likely_generation_ids'].to(device)
            most_likely_generation = torch.as_tensor(most_likely_generation, device=model.device, dtype=torch.long)
            target_ids = most_likely_generation.clone()
            target_ids[:len(prompt)] = -100
            model_output = model(torch.reshape(most_likely_generation, (1, -1)),
                                 labels=target_ids,
                                 output_hidden_states=True)
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood_of_most_likely_gen = model_output['loss']
            most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

            if 'cleaned_second_most_likely_generation_ids' in sample:
                second_most_likely_generation = sample['cleaned_second_most_likely_generation_ids'].to(device)
            else:
                second_most_likely_generation = sample['second_most_likely_generation_ids'].to(device)
            second_most_likely_generation = torch.as_tensor(second_most_likely_generation, device=model.device, dtype=torch.long)
            target_ids = second_most_likely_generation.clone()
            target_ids[:len(prompt)] = -100
            model_output = model(torch.reshape(second_most_likely_generation, (1, -1)),
                                 labels=target_ids,
                                 output_hidden_states=True)
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood_of_second_most_likely_gen = model_output['loss']
            second_most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

            neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (
                len(most_likely_generation) - len(prompt))

            sequence_embeddings = torch.stack(sequence_embeddings)
            result_dict['prompt'] = prompt
            result_dict['generations'] = generations
            result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
            result_dict['neg_log_likelihoods'] = neg_log_likelihoods
            result_dict['sequence_embeddings'] = most_likely_generation_embedding
            result_dict['most_likely_sequence_embedding'] = most_likely_generation
            result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
            result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
            result_dict['pointwise_mutual_information'] = pointwise_mutual_information
            result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen
            result_dict[
                'average_neg_log_likelihood_of_second_most_likely_gen'] = average_neg_log_likelihood_of_second_most_likely_gen
            result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen
            result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_[0]]['semantic_set_ids'], device=device)
            result_dict['id'] = id_
            result.append(result_dict)
        return result


likelihoods = get_neg_loglikelihoods(model, sequences)

with open(f'{config.data_dir}/sequences/{run_name}/{args.generation_model.split("/")[-1]}_generations_{args.evaluation_model.split("/")[-1]}_likelihoods.pkl',
          'wb') as outfile:
    pickle.dump(likelihoods, outfile)
