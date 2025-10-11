import json
import pandas as pd
import numpy as np
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_all(tasks, filenames, model_info):
    all_dfs = []
    for task in tasks:
        task_name = f'new_models_data/{task}'
        for filename in filenames:
            if not os.path.exists(f'{task_name}/{filename}'):
                print(f'File {task_name}/{filename} does not exist, skipping.')
                continue
            loaded_df = pd.read_csv(f'{task_name}/{filename}')
            loaded_df['Task'] = task
            loaded_df['Num_choices'] = tasks[task]
            loaded_df['Model_name'] = loaded_df['Model'].str.partition("_")[0]
            loaded_df['Prompt'] = loaded_df['Model'].str.partition("_").iloc[:, -1]
            # print(f'Processing {model_name} for task {task}')
            loaded_df[["Model_family", "Model_version", "Model_type", "Model_size"]] = (loaded_df['Model_name'].map(model_info).apply(pd.Series))
            all_dfs.append(loaded_df)
    return pd.concat(all_dfs, ignore_index=True)


def load_without_conf(tasks, filenames, model_info):
    all_dfs = []
    for task in tasks:
        task_name = f'new_models_data/{task}'
        for filename in filenames:
            if not os.path.exists(f'{task_name}/{filename}'):
                print(f'File {task_name}/{filename} does not exist, skipping.')
                continue
            loaded_df = pd.read_csv(f'{task_name}/{filename}')
            loaded_df = loaded_df[["Model","Accuracy", "Norm_acc","Overall_logits_mean","Overall_logits_std", "ECE_logits"]]
            loaded_df['Task'] = task
            loaded_df['Num_choices'] = tasks[task]
            loaded_df['Model_name'] = loaded_df['Model'].str.partition("_")[0]
            loaded_df['Prompt'] = loaded_df['Model'].str.partition("_").iloc[:, -1]
            # print(f'Processing {model_name} for task {task}')
            loaded_df[["Model_family", "Model_version", "Model_type", "Model_size"]] = (loaded_df['Model_name'].map(model_info).apply(pd.Series))
            all_dfs.append(loaded_df)
    return pd.concat(all_dfs, ignore_index=True)


def calculate_ece(confidences, correct):
    M = 10 # 11 bins: 0,1,...,10
    ece = 0.0
    n = len(confidences)
    bins = np.linspace(0.0, 10.0, M+1)

    for i in range(M):
        bin_lower, bin_upper = bins[i], bins[i+1]
        if M == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            conf_avg = np.mean(confidences[in_bin])
            norm_conf_avg = conf_avg / 10.0
            acc_avg = np.mean(correct[in_bin])
            ece += (bin_size / n) * np.abs(acc_avg - norm_conf_avg)

    return ece


def normalize_accuracy(accuracy, num_choices):
    if num_choices <= 1:
        return 0.0
    return (accuracy - 1/num_choices) / (1 - 1/num_choices)


def calculate_acc(task, file):
    loaded_df = pd.read_csv(f'{task}/{file}')
    with open(f'{task}/data.json') as f:
        data = json.load(f)
    ground_truth = [data["validation"][k]["targets"][1] for k in range(len(data["validation"]))]
    loaded_df['ground_truth'] = ground_truth
    loaded_df.replace('N/A', pd.NA, inplace=True)
    valid_answers = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    loaded_df.loc[~loaded_df['Answer'].isin(valid_answers), 'Answer'] = pd.NA
    loaded_df.dropna(subset=['Answer'], inplace=True)
    if loaded_df.empty:
        return None, None
    match_rows = loaded_df[loaded_df['Answer'] == loaded_df['ground_truth']]
    accuracy = len(match_rows)/len(loaded_df)
    num_of_options = len(data['train'][0]["multiple_choice_targets"])
    norm_acc = normalize_accuracy(accuracy, num_of_options)
    mean_std = loaded_df[['Logits']].agg(['mean', 'std'])
    logits_ece = calculate_ece(loaded_df['Logits'].to_numpy()*10, (loaded_df['Answer'] == loaded_df['ground_truth']).astype(int).to_numpy())
    return accuracy, norm_acc, mean_std['Logits']['mean'], mean_std['Logits']['std'], logits_ece


def calculate_coefficient(task, file):
    loaded_df = pd.read_csv(f'{task}/{file}')
    with open(f'{task}/data.json') as f:
        data = json.load(f)
    num_of_options = len(data['train'][0]["multiple_choice_targets"])
    ground_truth = [data["validation"][k]["targets"][1] for k in range(len(data["validation"]))]
    loaded_df['ground_truth'] = ground_truth

    loaded_df['Confidence'] = pd.to_numeric(loaded_df['Confidence'], errors='coerce')
    loaded_df.replace('N/A', pd.NA, inplace=True)
    valid_answers = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    loaded_df.loc[~loaded_df['Answer'].isin(valid_answers), 'Answer'] = pd.NA
    loaded_df.dropna(subset=['Answer', 'Logits', 'Confidence'], inplace=True)
    if loaded_df.empty:
        return None, None, None, None, None, None, None, None, None, None,

    correlation = loaded_df['Logits'].corr(loaded_df['Confidence'])
    mean_std = loaded_df[['Logits', 'Confidence']].agg(['mean', 'std'])
    # logits_ece = calculate_ece(loaded_df['Logits'].to_numpy(), (loaded_df['Answer'] == loaded_df['ground_truth']).astype(int).to_numpy())
    conf_ece = calculate_ece(loaded_df['Confidence'].to_numpy(), (loaded_df['Answer'] == loaded_df['ground_truth']).astype(int).to_numpy())
    conf_logits_ece = calculate_ece(loaded_df['Confidence'].to_numpy(), loaded_df['Logits'].to_numpy())
    logits_ece = calculate_ece(loaded_df['Logits'].to_numpy()*10, (loaded_df['Answer'] == loaded_df['ground_truth']).astype(int).to_numpy())
    dist_logits_conf = np.linalg.norm(loaded_df['Confidence'].to_numpy()/10 - loaded_df['Logits'].to_numpy())/np.sqrt(len(loaded_df['Logits']))

    match_rows = loaded_df[loaded_df['Answer'] == loaded_df['ground_truth']]
    accuracy = len(match_rows)/len(loaded_df)
    norm_acc = normalize_accuracy(accuracy, num_of_options)
    mismatch_rows = loaded_df[loaded_df['Answer'] != loaded_df['ground_truth']]
    corr_match = match_rows['Logits'].corr(match_rows['Confidence'])
    mean_std_match = match_rows[['Logits', 'Confidence']].agg(['mean', 'std'])
    corr_mismatch = mismatch_rows['Logits'].corr(mismatch_rows['Confidence'])
    mean_std_mismatch = mismatch_rows[['Logits', 'Confidence']].agg(['mean', 'std'])
    return accuracy, norm_acc, correlation, conf_ece, conf_logits_ece, logits_ece, dist_logits_conf, mean_std['Logits']['mean'], mean_std['Logits']['std'], mean_std['Confidence']['mean'], mean_std['Confidence']['std'], corr_match, mean_std_match['Logits']['mean'], mean_std_match['Logits']['std'], mean_std_match['Confidence']['mean'], mean_std_match['Confidence']['std'], corr_mismatch, mean_std_mismatch['Logits']['mean'], mean_std_mismatch['Logits']['std'], mean_std_mismatch['Confidence']['mean'], mean_std_mismatch['Confidence']['std']


if __name__ == "__main__":
    
    model_base = [
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-14B-Base",
        "mistralai/Mistral-7B-v0.1",
        "mistral-community/Mistral-7B-v0.2",
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mistral-Nemo-Base-2407",
        "mistralai/Mixtral-8x7B-v0.1",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-3.2-3B",
    ]
    model_instruct = [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen3-4B-Instruct-2507",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
        "meta-llama/Llama-2-7b-chat-hf", 
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        "meta-llama/Llama-3.2-3B-Instruct", 
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]
    tasks = {
        "conceptual_combinations": 4,
        "ruin_names": 4, 
        "temporal_sequences": 4, 
        "MMLU": 4,
        "cause_and_effect": 2,
        "CoLA": 2,
        "QNLI": 2,
        "QQP": 2
    }
    
    for task_name in tasks:
        print(f"Processing {task_name}...")
        task_name = f'new_models_data/{task_name}'
        data = [["Model","Accuracy", "Norm_acc","Overall_logits_mean","Overall_logits_std", "ECE_logits"]]
        for model in model_base:
            model = model.split("/")[-1]
            for prompt in ["withoutConfidence"]:
                if not os.path.exists(f'{task_name}/base/{model}_prompts_base_{prompt}.csv'):
                    print(f"File not found: {task_name}/base/{model}_prompts_base_{prompt}.csv")
                    continue
                data.append([f"{model}_base_{prompt}"] + list(calculate_acc(task_name,f"base/{model}_prompts_base_{prompt}.csv")))  
        
        with open(f'{task_name}/base_withoutConfidence.csv', "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
    
    for task_name in tasks:
        print(f"Processing {task_name}...")
        task_name = f'new_models_data/{task_name}'
        data = [["Model","Accuracy", "Norm_acc","Overall_logits_mean","Overall_logits_std", "ECE_logits"]]
        for model in model_instruct:
            model = model.split("/")[-1]
            for prompt in ["withoutConfidence"]:
                if not os.path.exists(f'{task_name}/instruct/{model}_{prompt}.csv'):
                    print(f"File not found: {task_name}/instruct/{model}_{prompt}.csv")
                    continue
                data.append([f"{model}_instruct_{prompt}"] + list(calculate_acc(task_name,f"instruct/{model}_{prompt}.csv")))  
        
        with open(f'{task_name}/instruct_withoutConfidence.csv', "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
            
            
    for task_name in tasks:
        print(f"Processing {task_name}...")
        task_name = f'new_models_data/{task_name}'
        data = [["Model","Accuracy", "Norm_acc","Overall_corr","ECE_conf", "ECE_conf-logits", "ECE_logits", "Dist_conf-logits","Overall_logits_mean","Overall_logits_std","Overall_confidence_mean","Overall_confidence_std","Match_corr","Match_logits_mean","Match_logits_std","Match_confidence_mean","Match_confidence_std","Mismatch_corr","Mismatch_logits_mean","Mismatch_logits_std","Mismatch_confidence_mean","Mismatch_confidence_std"]]
        for model in model_base:
            model = model.split("/")[-1]
            for prompt in ["45", "new_56"]:
                # print(f"Processing {model} with prompt {prompt}...")
                if not os.path.exists(f'{task_name}/base/{model}_prompts_base_{prompt}.csv'):
                    print(f"File not found: {task_name}/base/{model}_prompts_base_{prompt}.csv")
                    continue
                data.append([f"{model}_base_{prompt}"] + list(calculate_coefficient(task_name,f"base/{model}_prompts_base_{prompt}.csv")))  
        
        with open(f'{task_name}/base_prompt_results.csv', "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
            
    for task_name in tasks:
        print(f"Processing {task_name}...")
        task_name = f'new_models_data/{task_name}'
        data = [["Model","Accuracy", "Norm_acc","Overall_corr","ECE_conf", "ECE_conf-logits", "ECE_logits", "Dist_conf-logits","Overall_logits_mean","Overall_logits_std","Overall_confidence_mean","Overall_confidence_std","Match_corr","Match_logits_mean","Match_logits_std","Match_confidence_mean","Match_confidence_std","Mismatch_corr","Mismatch_logits_mean","Mismatch_logits_std","Mismatch_confidence_mean","Mismatch_confidence_std"]]
        for model in model_instruct:
            model = model.split("/")[-1]
            for prompt in ["2"]:
                if not os.path.exists(f'{task_name}/instruct/{model}_{prompt}.csv'):
                    print(f"File not found: {task_name}/instruct/{model}_{prompt}.csv")
                    continue
                data.append([f"{model}_instruct_{prompt}"] + list(calculate_coefficient(task_name,f"instruct/{model}_{prompt}.csv")))  
        
        with open(f'{task_name}/instruct2_prompt_results.csv', "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
            
            
    model_base = [
        "Qwen/Qwen2.5-7B",
        "mistralai/Mistral-7B-v0.3",
        "meta-llama/Llama-3.1-8B",
    ]
    model_instruct = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    ]
    emotion_prompts = [
        "criticism1", 
        "criticism2", 
        "criticism3", 
        "approval1", 
        "approval2", 
        "approval3", 
        "nonsense1",
        "nonsense2",
        "nonsense3",
        "withAlignment",
    ]
    for task_name in tasks:
        print(f"Processing {task_name}...")
        task_name = f'new_models_data/{task_name}'
        data = [["Model","Accuracy", "Norm_acc","Overall_corr","ECE_conf", "ECE_conf-logits", "ECE_logits", "Dist_conf-logits", "Overall_logits_mean","Overall_logits_std","Overall_confidence_mean","Overall_confidence_std","Match_corr","Match_logits_mean","Match_logits_std","Match_confidence_mean","Match_confidence_std","Mismatch_corr","Mismatch_logits_mean","Mismatch_logits_std","Mismatch_confidence_mean","Mismatch_confidence_std"]]
        for model in model_base:
            model = model.split("/")[-1]
            for prompt in ["45", "new_56"]:
                data.append([f"{model}_base_{prompt}"] + list(calculate_coefficient(task_name,f"base/{model}_prompts_base_{prompt}.csv")))  
                for emotion in emotion_prompts:
                    emo_prompt = f"{prompt}_{emotion}"
                    if not os.path.exists(f'{task_name}/base/{model}_prompts_base_{emo_prompt}.csv'):
                        print(f"File not found: {task_name}/base/{model}_prompts_base_{emo_prompt}.csv")
                        continue
                    data.append([f"{model}_base_{emo_prompt}"] + list(calculate_coefficient(task_name,f"base/{model}_prompts_base_{emo_prompt}.csv")))  
        
        with open(f'{task_name}/base_prompt_pertubation.csv', "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
    
    for task_name in tasks:
        print(f"Processing {task_name}...")
        task_name = f'new_models_data/{task_name}'
        data = [["Model","Accuracy", "Norm_acc","Overall_corr","ECE_conf", "ECE_conf-logits", "ECE_logits", "Dist_conf-logits","Overall_logits_mean","Overall_logits_std","Overall_confidence_mean","Overall_confidence_std","Match_corr","Match_logits_mean","Match_logits_std","Match_confidence_mean","Match_confidence_std","Mismatch_corr","Mismatch_logits_mean","Mismatch_logits_std","Mismatch_confidence_mean","Mismatch_confidence_std"]]
        for model in model_instruct:
            model = model.split("/")[-1]
            for prompt in ["2"]:
                data.append([f"{model}_instruct_{prompt}"] + list(calculate_coefficient(task_name,f"instruct/{model}_{prompt}.csv")))
                for emotion in emotion_prompts:
                    emo_prompt = f"{prompt}_{emotion}"
                    if not os.path.exists(f'{task_name}/instruct/{model}_{emo_prompt}.csv'):
                        print(f"File not found: {task_name}/instruct/{model}_{emo_prompt}.csv")
                        continue
                    data.append([f"{model}_instruct_{emo_prompt}"] + list(calculate_coefficient(task_name,f"instruct/{model}_{emo_prompt}.csv")))  
        
        with open(f'{task_name}/instruct_prompt_pertubation.csv', "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
            
            
    model_info = {
        "Qwen2.5-14B-Instruct": ("qwen", "2.5", "instruct", 14),
        "Qwen2.5-7B-Instruct": ("qwen", "2.5", "instruct", 7),
        "Qwen2.5-7B": ("qwen", "2.5", "base", 7),
        "Qwen2.5-14B": ("qwen", "2.5", "base", 14),
        "Qwen2.5-3B": ("qwen", "2.5", "base", 3),
        "Qwen2.5-3B-Instruct": ("qwen", "2.5", "instruct", 3),
        "Qwen3-4B-Base": ("qwen", "3", "base", 4),
        "Qwen3-8B-Base": ("qwen", "3", "base", 8),
        "Qwen3-14B-Base": ("qwen", "3", "base", 14),
        "Qwen3-4B-Instruct-2507": ("qwen", "3", "instruct", 4),
        "Mistral-7B-v0.1": ("mistral", "0.1", "base", 7),
        "Mistral-7B-v0.2": ("mistral", "0.2", "base", 7),
        "Mistral-7B-v0.3": ("mistral", "0.3", "base", 7),
        "Mistral-Nemo-Base-2407": ("mistral", "nemo", "base", 12),
        "Mistral-7B-Instruct-v0.1": ("mistral", "0.1", "instruct", 7),
        "Mistral-7B-Instruct-v0.2": ("mistral", "0.2", "instruct", 7),
        "Mistral-7B-Instruct-v0.3": ("mistral", "0.3", "instruct", 7),
        "Mistral-Nemo-Instruct-2407": ("mistral", "nemo", "instruct", 12),
        "Mixtral-8x7B-v0.1": ("mistral", "0.1", "base", 13),
        "Mixtral-8x7B-Instruct-v0.1": ("mistral", "0.1", "instruct", 13),
        "Mixtral-8x22B-v0.1": ("mistral", "0.1", "base", 44),
        "Mixtral-8x22B-Instruct-v0.1": ("mistral", "0.1", "instruct", 44),
        "Llama-3.1-8B": ("llama", "3.1", "base", 8),
        "Llama-2-7b-hf": ("llama", "2", "base", 7),
        "Llama-2-13b-hf": ("llama", "2", "base", 13),
        "Llama-3.2-3B": ("llama", "3.2", "base", 3),
        "Meta-Llama-3-8B": ("llama", "3", "base", 8),
        "Llama-2-7b-chat-hf": ("llama", "2", "instruct", 7),
        "Llama-2-13b-chat-hf": ("llama", "2", "instruct", 13),
        "Meta-Llama-3.1-8B-Instruct": ("llama", "3.1", "instruct", 8),
        "Llama-3.2-3B-Instruct": ("llama", "3.2", "instruct", 3),
        "Meta-Llama-3-8B-Instruct": ("llama", "3", "instruct", 8),
    }
    
    filenames = ['base_prompt_results.csv', 'instruct2_prompt_results.csv']
    all_data = load_all(tasks, filenames, model_info)
    for i in range(len(all_data)):
        print(all_data['Model'][i], all_data['Model_size'][i], all_data['Model_family'][i], all_data['Model_version'][i], all_data['Model_type'][i], all_data['Task'][i], all_data['Prompt'][i], all_data['Accuracy'][i])
    
    # Save the combined dataframe to a CSV file
    all_data.to_csv('new_models_data/all_model_results.csv', index=False)
    
    filenames = ['base_prompt_results.csv', 'instruct2_prompt_results.csv', 'base_withoutConfidence.csv', 'instruct_withoutConfidence.csv']
    data_with_withoutConf = load_without_conf(tasks, filenames, model_info)
    data_with_withoutConf.to_csv('new_models_data/with&withoutConf.csv', index=False)
    
    choices = ["criticism", "approval", "nonsense", "withAlignment"]
    data_pertubation = []
    for model_type in ["base", "instruct"]:
        for task in tasks:
            df = pd.read_csv(f'new_models_data/{task}/{model_type}_prompt_pertubation.csv')
            conditions = [
                df['Model'].str.contains("criticism", case=False, na=False),
                df['Model'].str.contains("approval", case=False, na=False),
                df['Model'].str.contains("nonsense", case=False, na=False),
                df['Model'].str.contains("withAlignment", case=False, na=False)
            ]
            df['Task'] = task
            df['pertubation'] = np.select(conditions, choices, default="origin")
            data_pertubation.append(df)
        
        data_pertubation = pd.concat(data_pertubation, ignore_index=True)
        data_pertubation.to_csv(f'new_models_data/{model_type}_pertubation.csv', index=False)