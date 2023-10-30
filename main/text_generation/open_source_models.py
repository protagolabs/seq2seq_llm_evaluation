"""
This script is used to generate the outputs for all open-source models using the HuggingFace transformers library,
for all three NLP tasks: text summarisation, simplification and grammatical error correction.
NOTE: depending on your OS and set up, you may need to run this terminal command before running this script:
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"""
import json
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
import pandas as pd
from tqdm import tqdm
from accelerate import init_empty_weights, infer_auto_device_map
import torch
import pytz


models_to_run = [
    'google/flan-t5-xxl',
    'google/flan-ul2',
    'facebook/opt-iml-max-30b',
    'bigscience/T0pp'
]


no_split_layers = [
    'T5Block',
    'T5Block',
    'OPTDecoderLayer',
    'T5Block'
]


simplification_prompts = [
    "Simplify the following text. [...]",
    "Simplify the following text. [...] \n The simplified version is: ",
    "Explain this to a 5 year old. [...]",
    "Explain this to a 5 year old. [...] \n The explanation to a 5 year old could be: ",
    "This is the main story: [...]\n The simplified version of the story is: "
]

summarisation_prompts = [
    "Summarize the following text.[...] \n The summary is:",
    "[...] \n Summarize the text above.",
    "Summarize the following text. [...] \n The very short summary is:",
    "This is the main story: [...]\n The summarized version of the story is:"
]


gec_prompts = [
    "Reply with a corrected version of the input sentence with all grammatical and spelling errors fixed. If there "
    "are no errors, reply with a copy of the original sentence. \n\n Input sentence: [...] \n Corrected sentence: ",
    "Correct the following to standard English: \n\n Sentence: [...] \n Correction:",
]


temperatures = [0.01, 0.5, 0.7]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="The task to use. Can be 'Summarisation', 'Simplification' or 'GEC'.",
        choices=["Summarisation", "Simplification", "GEC"], required=True
    )
    args = parser.parse_args()
    TASK = args.task  # change task as appropriate
    time_now = datetime.now(pytz.timezone('Europe/London')).strftime('%Y-%m-%dT%H:%M:%S')
    if TASK == 'Simplification':
        with open('data/newsela-auto/newsela-auto/ACL2020/test_dedup_sample.src', 'r') as f:
            inputs = f.readlines()
            inputs = [i.strip("\n") for i in inputs]
        prompts = simplification_prompts
    elif TASK == 'Summarisation':
        df = pd.read_csv('data/CNNDailyMail/test_subset_3000.csv')
        inputs = df['article']
        # inputs_1506 = df['article_trunc1506']
        prompts = summarisation_prompts
    elif TASK == 'GEC':
        with open('data/BEA_website/Write & Improve/origin.txt', 'r') as f:
            inputs = f.readlines()
            inputs = [i.strip("\n") for i in inputs]
        prompts = gec_prompts
    else:
        raise ValueError(f"TASK should be 'Summarisation', 'Simplification' or 'GEC'. Got '{TASK}' instead.")

    results = {
        "model": [],
        "prompt": [],
        "input": [],
        "detokenized_input": [],
        "original_input": [],
        "output": [],
        "temperature": []
    }

    for model_name, no_split in zip(models_to_run, no_split_layers):
        print(f"INFERENCE ON MODEL {model_name}")
        max_memory = {i: "24GiB" for i in range(6)}
        max_memory[0] = "10GiB"  # to fit lm_head to the same device as the inputs
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        with init_empty_weights():
            try:
                model = AutoModelForCausalLM.from_config(config)
            except ValueError:
                model = AutoModelForSeq2SeqLM.from_config(config)
            device_map = infer_auto_device_map(model, no_split_module_classes=[no_split], max_memory=max_memory)
            device_map['lm_head'] = 0
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, load_in_8bit=True)
        except ValueError:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map, load_in_8bit=True)

        for text_input in tqdm(inputs):
            for prompt in prompts:
                for temperature in temperatures:
                    results['model'].append(model_name)
                    results['prompt'].append(prompt)
                    results['temperature'].append(temperature)
                    text = prompt.replace("[...]", text_input)
                    results['input'].append(text)
                    results['original_input'].append(text_input)
                    detokenized_text = tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True)
                    results['detokenized_input'].append(detokenized_text)
                    input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=128).to(0)
                    beam_outputs = model.generate(
                        input_ids,
                        max_length=128 if model_name != 'facebook/opt-iml-max-30b' else 256,  # default 50
                        # 'facebook/opt-iml-max-30b' includes input in the output so length must be longer
                        top_p=1.0,
                        temperature=temperature,
                        do_sample=True,
                    )
                    results["output"].append(
                        tokenizer.decode(beam_outputs[0], skip_special_tokens=True).replace(detokenized_text, '')
                    )
            with open(f"data/outputs/output_{TASK.lower()}_{time_now}.json", "w") as f:
                json.dump(results, f)
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame.from_dict(results)
    df.to_csv(f"data/outputs/output_{TASK.lower()}.csv", index=False)
