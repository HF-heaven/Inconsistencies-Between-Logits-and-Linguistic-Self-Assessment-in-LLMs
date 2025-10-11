# Correlation between Confidence Measurements

## Classification tasks
Codes for classification tasks can be found in ```/classification_tasks```.

Use ```base_confidence_extraction.py``` to get the logits-based and linguistic confidence results for base models, for example:

```bash
python base_confidence_extraction.py \
  --model_id Qwen/Qwen2.5-14B \        # Hugging Face model name
  --prompt_file prompts_base_new_56.json \      # Prompt file under /new_models_data/{task}/, new_56, 45 for Prompt 1, 2
  --task CoLA \                                 # Task name (folder name)
  --token hf_xxx_your_token \          # (Optional) Hugging Face access token
  --align_logits_hint                           # (Optional) Append a note asking model to align confidence with logits
```
Use ```instruct_confidence_extraction.py``` to get the logits-based and linguistic confidence results for instruction-tuned models, for example:
```bash
python instruct_confidence_extraction.py \
  --model_id Qwen/Qwen2.5-14B-Instruct \
  --prompt_file prompts_instruct_2.json \
  --task CoLA \
  --token hf_xxx_your_token \
  --align_logits_hint
```
Use ```base_without_conf.py``` to get only logits-based confidence results for base models, whose prompt doesn't have the requirement for linguistic confidence. For example:
```bash
python base_without_conf.py \
  --model_id meta-llama/Llama-2-13b-hf \        
  --prompt_file prompts_base_withoutConfidence.json \  
  --task CoLA \                            
  --token hf_xxx_your_token           
```
Use ```instruct_without_conf.py``` to get only logits-based confidence results for instruction-tuned models, whose prompt doesn't have the requirement for linguistic confidence. For example:
```bash
python instruct_without_conf.py \
  --model_id Qwen/Qwen2.5-14B-Instruct \   
  --prompt_file prompts_instruct_withoutConfidence.json \ 
  --task CoLA \                          
  --token hf_xxx_your_token                  
```
Run ```results_analysis.py``` to get the results.

## Generation tasks
Codes for generation tasks can be found in ```/generation_tasks```.

Use ```parse_coqa.py``` and ```parse_triviaqa.py``` to preprocess the datasets.

Use ```model_pipeline``` to get all results.

After that, use ```result_analysis.ipynb``` to analyze the correlation of semantic uncertainty and linguistic confidence.