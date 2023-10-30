"""
This script calculates the T-test (significance test) to assess whether the differences in the distributions of
automatic metrics scores between different models are statistically significant. The distributions which are
significantly better (p < 0.001) than others are detailed in section 4.1 of our paper.
"""
import json
import argparse
import scipy
from scipy.stats import gmean
import evaluate
import pandas as pd
import numpy as np
from tqdm import tqdm
from main.automatic_evaluation.eval_errant_GEC import compute_score_by_individual_sample


def calculate_metrics(metric: str, df: pd.DataFrame, inputs_references: list[str], output_columns: str = 'output_1'):
    """
    This function is very similar in purpose to calculate_sari_or_rouge from evaluate_sari_or_rouge.py (see this
    function for a complete documentation), with the difference that it also returns sample-by-sample scores,
    which are necessary to calculate the T-test.
    """
    models = set(df['model'])
    prompts = set(df['prompt'])
    assert 'temperature' in df.columns, \
        "This code is only designed to work when the input data contain temperature param"
    temperatures = set(df['temperature'])
    metrics = {
        'by_model': {},
        'by_prompt': {},
        'by_temperature': {},
        'by_model_and_prompt': {},
        'by_model_and_temperature': {},
        'by_prompt_and_temperature': {},
        'by_model_prompt_temperature': {},
        'one_by_one': {}
    }

    score = evaluate.load(metric)  # 'sari' or 'rouge'

    for model in models:
        for prompt in prompts:
            for temperature in temperatures:
                subset_df = df[(df['model'] == model) & (df['prompt'] == prompt) & (df['temperature'] == temperature)]
                inputs_ = subset_df['original_input'].tolist()
                references_ = [inputs_references[i] for i in subset_df['original_input']]
                predictions_ = subset_df[output_columns].tolist()
                try:
                    if metric == 'sari':
                        result = score.compute(
                            sources=inputs_, predictions=predictions_, references=[[ref] for ref in references_]
                        )
                        metrics['by_model_prompt_temperature'][(model, prompt, temperature)] = result['sari']
                    elif metric == 'rouge':
                        result = score.compute(
                            predictions=predictions_, references=references_, use_stemmer=False, use_aggregator=False
                        )
                        result_mean = {k: np.mean(v) for k, v in result.items()}
                        result_gmean = gmean([result_mean['rouge1'], result_mean['rouge2'], result_mean['rougeL']])
                        metrics['by_model_prompt_temperature'][(model, prompt, temperature)] = result_gmean
                except (ZeroDivisionError, IndexError):
                    continue
                metrics['one_by_one'][(model, prompt, temperature)] = []
                # for significance testing, we need the score of individual samples
                for inpp, reff, predd in tqdm(zip(inputs_, references_, predictions_)):
                    try:
                        if metric == 'sari':
                            result = score.compute(
                                sources=[inpp], predictions=[predd], references=[[ref] for ref in [reff]]
                            )
                            metrics['one_by_one'][(model, prompt, temperature)].append(result['sari'])
                        elif metric == 'rouge':
                            result = score.compute(
                                predictions=[predd], references=[reff], use_stemmer=False, use_aggregator=False
                            )
                            result_mean = {k: np.mean(v) for k, v in result.items()}
                            result_gmean = gmean([result_mean['rouge1'], result_mean['rouge2'], result_mean['rougeL']])
                            metrics['one_by_one'][(model, prompt, temperature)].append(result_gmean)
                    except (ZeroDivisionError, IndexError):
                        continue
    printable_df = pd.DataFrame.from_dict(
        {'by_model_prompt_temperature': metrics['by_model_prompt_temperature']}
    )
    return metrics, printable_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="The task to use. Can be 'Summarisation', 'Simplification' or 'GEC'.",
        choices=["Summarisation", "Simplification", "GEC"], required=True
    )
    args = parser.parse_args()
    TASK = args.task  # change task as appropriate
    if TASK == 'Simplification':
        with open('data/newsela-auto/newsela-auto/ACL2020/test_dedup_sample.src', 'r') as f:
            inputs = f.readlines()
            inputs = [i.strip("\n") for i in inputs]
        with open('data/newsela-auto/newsela-auto/ACL2020/test_dedup_sample.dst', 'r') as f:
            references = f.readlines()
            references = [i.strip("\n") for i in references]
        with open('data/outputs/output_simplification_all.json', 'r') as f:
            model_outputs = json.load(f)
        df = pd.DataFrame.from_dict(model_outputs)
    elif TASK == 'Summarisation':
        df = pd.read_csv('data/CNNDailyMail/test_subset_3000.csv')
        inputs = df['article']
        references = df['highlights']
        with open('data/outputs/output_summarization_all.json', 'r') as f:
            model_outputs = json.load(f)
        df = pd.DataFrame.from_dict(model_outputs)
    elif TASK == 'GEC':
        # The design for the GEC part is slightly different, and we don't at the moment have a streamlined elegant
        # solution as for the other tasks. Therefore you will need to decide a priori which model output you want to
        # use (out of the various models and hyperparameter combinations you tried) and load it one by one.
        # The file "target.txt" (see line below) will always correspond to a single combination of model settings.
        hyp_path = 'data/BEA_website/Write & Improve/target.txt'
        ref_path = 'data/BEA_website/Write & Improve/gold.txt'
    else:
        raise ValueError(f"TASK should be 'Summarisation', 'Simplification' or 'GEC'. Got '{TASK}' instead.")

    assert len(inputs) == len(references)
    inputs_references = {k: v for k, v in zip(inputs, references)}
    if TASK == 'Simplification' or TASK == 'Summarisation':
        metrics, printable_df = calculate_metrics('rouge', df, inputs_references, output_columns='output')

        individual_scores = pd.DataFrame.from_dict({'by_model_prompt_temperature': metrics['one_by_one']})
        printable_df['one_by_one'] = individual_scores['by_model_prompt_temperature']
        printable_df['index'] = range(len(printable_df))
        print(printable_df)
        # Now you need to observe the `printable_df` and decide which rows you want to compare, as the T-test is always
        # a comparison between two distributions. For example, if you want to compare rows 0 and 4, you can do
        print(scipy.stats.ttest_ind(
            individual_scores['by_model_prompt_temperature'][0], individual_scores['by_model_prompt_temperature'][4]
        ))
    elif TASK == 'GEC':
        F_0_5_list = compute_score_by_individual_sample(hyp_path, ref_path)
        print(F_0_5_list)
