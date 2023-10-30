"""
This script is used to merge together the outputs of simplification and summarisation
for all models. It can also easily be adapted for the GEC (grammatical error correction) task.
"""
import json


if __name__ == '__main__':
    with open('data/outputs/output_simplification.json', 'r') as f:
        simp_huggingface_models = json.load(f)
    with open('data/outputs/output_openai_simplification.json', 'r') as f:
        simp_openai = json.load(f)
    with open('data/outputs/output_chatgpt_simplification_2023-04-13T15:53:49.json', 'r') as f:
        simp_chatgpt = json.load(f)
    with open('data/outputs/output_summarisation.json', 'r') as f:
        summ_huggingface_models = json.load(f)
    with open('data/outputs/output_openai_summarisation.json', 'r') as f:
        summ_openai = json.load(f)
    with open('data/outputs/output_chatgpt_summarisation_2023-04-14T12:05:05.json', 'r') as f:
        summ_chatgpt = json.load(f)

    simp_openai['detokenized_input'] = ['N/A'] * len(simp_openai['model'])
    summ_openai['detokenized_input'] = ['N/A'] * len(summ_openai['model'])
    simp_chatgpt['detokenized_input'] = ['N/A'] * len(simp_chatgpt['model'])
    summ_chatgpt['detokenized_input'] = ['N/A'] * len(summ_chatgpt['model'])

    simplification_all = {
        'model': [],
        'prompt': [],
        'temperature': [],
        'input': [],
        'detokenized_input': [],
        'original_input': [],
        'output': []
    }

    for key in simp_huggingface_models.keys():
        simplification_all[key].extend(simp_huggingface_models[key])
        simplification_all[key].extend(simp_openai[key])
        simplification_all[key].extend(simp_chatgpt[key])

    summarisation_all = {
        'model': [],
        'prompt': [],
        'temperature': [],
        'input': [],
        'detokenized_input': [],
        'original_input': [],
        'output': []
    }

    for key in summ_huggingface_models.keys():
        summarisation_all[key].extend(summ_huggingface_models[key])
        summarisation_all[key].extend(summ_openai[key])
        summarisation_all[key].extend(summ_chatgpt[key])

    with open('data/outputs/output_simplification_all.json', 'w') as f:
        json.dump(simplification_all, f)
    with open('data/outputs/output_summarisation_all.json', 'w') as f:
        json.dump(summarisation_all, f)
