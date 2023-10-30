"""
Script used to calculate the SARI score for text simplification and the ROUGE score for text summarisation.
OPTIONAL: if you set GENERATE_PLOT == True by using the "-p" argument, this file also generates a heatmap
with the SARI or ROUGE scores for each model and prompt. This was used to visually inspect the performance
of the various model/prompt combinations, but it is not necessary to reproduce the results in the paper.
"""
import json
import argparse
import evaluate
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import gmean
from matplotlib import pyplot as plt


def calculate_sari_or_rouge(metric: str, df: pd.DataFrame, inputs_references: dict, output_columns: str = 'output_1'):
    """Compute the metrics, either SARI or ROUGE F1 geometric mean, given
    inputs data, reference outputs and model predictions.

    Args:
        metric (str): either "sari" or "rouge"
        df (pd.DataFrame): the results dataframe, as obtained by converting
        the json model output
        inputs_references (dict): the gold references, required for SARI scores
        output_columns (str, optional): The output column to use, in case there are multiple outputs
        due to beam search. Defaults to 'output_1'

    Returns:
        _type_: _description_
    """
    assert any(metric == i for i in ("sari", "rouge")), f"metric must be 'sari' or 'rouge', got '{metric}' instead."
    models = set(df['model'])
    prompts = set(df['prompt'])
    if 'temperature' in df.columns:
        temperatures = set(df['temperature'])
        metrics = {
            'by_model': {},
            'by_prompt': {},
            'by_temperature': {},
            'by_model_and_prompt': {},
            'by_model_and_temperature': {},
            'by_prompt_and_temperature': {},
            'by_model_prompt_temperature': {}
        }
    else:
        metrics = {'by_model': {}, 'by_prompt': {}, 'by_model_and_prompt': {}}
    score = evaluate.load(metric)  # 'sari' or 'rouge'

    for model in models:
        subset_df = df[df['model'] == model]
        inputs_ = subset_df['original_input'].tolist()
        references_ = [inputs_references[i] for i in subset_df['original_input']]
        predictions_ = subset_df[output_columns].tolist()
        try:
            if metric == 'sari':
                result = score.compute(
                    sources=inputs_, predictions=predictions_, references=[[ref] for ref in references_]
                )
                metrics['by_model'][model] = result['sari']
            elif metric == 'rouge':
                result = score.compute(
                    predictions=predictions_, references=references_, use_stemmer=False, use_aggregator=False
                )
                result_mean = {k: np.mean(v) for k, v in result.items()}
                result_gmean = gmean([result_mean['rouge1'], result_mean['rouge2'], result_mean['rougeL']])
                metrics['by_model'][model] = result_gmean
        except (ZeroDivisionError, IndexError):
            continue

    for prompt in prompts:
        subset_df = df[df['prompt'] == prompt]
        inputs_ = subset_df['original_input'].tolist()
        references_ = [inputs_references[i] for i in subset_df['original_input']]
        predictions_ = subset_df[output_columns].tolist()
        try:
            if metric == 'sari':
                result = score.compute(
                    sources=inputs_, predictions=predictions_, references=[[ref] for ref in references_]
                )
                metrics['by_prompt'][prompt] = result['sari']
            elif metric == 'rouge':
                result = score.compute(
                    predictions=predictions_, references=references_, use_stemmer=False, use_aggregator=False
                )
                result_mean = {k: np.mean(v) for k, v in result.items()}
                result_gmean = gmean([result_mean['rouge1'], result_mean['rouge2'], result_mean['rougeL']])
                metrics['by_prompt'][prompt] = result_gmean
        except (ZeroDivisionError, IndexError):
            continue

    if 'temperature' in df.columns:
        for temperature in temperatures:
            subset_df = df[df['temperature'] == temperature]
            inputs_ = subset_df['original_input'].tolist()
            references_ = [inputs_references[i] for i in subset_df['original_input']]
            predictions_ = subset_df[output_columns].tolist()
            try:
                if metric == 'sari':
                    result = score.compute(
                        sources=inputs_, predictions=predictions_, references=[[ref] for ref in references_]
                    )
                    metrics['by_temperature'][temperature] = result['sari']
                elif metric == 'rouge':
                    result = score.compute(
                        predictions=predictions_, references=references_, use_stemmer=False, use_aggregator=False
                    )
                    result_mean = {k: np.mean(v) for k, v in result.items()}
                    result_gmean = gmean([result_mean['rouge1'], result_mean['rouge2'], result_mean['rougeL']])
                    metrics['by_temperature'][temperature] = result_gmean
            except (ZeroDivisionError, IndexError):
                continue

    for model in models:
        metrics['by_model_and_prompt'][model] = {}
        for prompt in prompts:
            subset_df = df[(df['model'] == model) & (df['prompt'] == prompt)]
            inputs_ = subset_df['original_input'].tolist()
            references_ = [inputs_references[i] for i in subset_df['original_input']]
            predictions_ = subset_df[output_columns].tolist()
            try:
                if metric == 'sari':
                    result = score.compute(
                        sources=inputs_, predictions=predictions_, references=[[ref] for ref in references_]
                    )
                    metrics['by_model_and_prompt'][model][prompt] = result['sari']
                elif metric == 'rouge':
                    result = score.compute(
                        predictions=predictions_, references=references_, use_stemmer=False, use_aggregator=False
                    )
                    result_mean = {k: np.mean(v) for k, v in result.items()}
                    result_gmean = gmean([result_mean['rouge1'], result_mean['rouge2'], result_mean['rougeL']])
                    metrics['by_model_and_prompt'][model][prompt] = result_gmean
            except (ZeroDivisionError, IndexError):
                continue

    if 'temperature' in df.columns:
        for model in models:
            metrics['by_model_and_temperature'][model] = {}
            for temperature in temperatures:
                subset_df = df[(df['model'] == model) & (df['temperature'] == temperature)]
                inputs_ = subset_df['original_input'].tolist()
                references_ = [inputs_references[i] for i in subset_df['original_input']]
                predictions_ = subset_df[output_columns].tolist()
                try:
                    if metric == 'sari':
                        result = score.compute(
                            sources=inputs_, predictions=predictions_, references=[[ref] for ref in references_]
                        )
                        metrics['by_model_and_temperature'][model][temperature] = result['sari']
                    elif metric == 'rouge':
                        result = score.compute(
                            predictions=predictions_, references=references_, use_stemmer=False, use_aggregator=False
                        )
                        result_mean = {k: np.mean(v) for k, v in result.items()}
                        result_gmean = gmean([result_mean['rouge1'], result_mean['rouge2'], result_mean['rougeL']])
                        metrics['by_model_and_temperature'][model][temperature] = result_gmean
                except (ZeroDivisionError, IndexError):
                    continue

        for prompt in prompts:
            metrics['by_prompt_and_temperature'][prompt] = {}
            for temperature in temperatures:
                subset_df = df[(df['prompt'] == prompt) & (df['temperature'] == temperature)]
                inputs_ = subset_df['original_input'].tolist()
                references_ = [inputs_references[i] for i in subset_df['original_input']]
                predictions_ = subset_df[output_columns].tolist()
                try:
                    if metric == 'sari':
                        result = score.compute(
                            sources=inputs_, predictions=predictions_, references=[[ref] for ref in references_]
                        )
                        metrics['by_prompt_and_temperature'][prompt][temperature] = result['sari']
                    elif metric == 'rouge':
                        result = score.compute(
                            predictions=predictions_, references=references_, use_stemmer=False, use_aggregator=False
                        )
                        result_mean = {k: np.mean(v) for k, v in result.items()}
                        result_gmean = gmean([result_mean['rouge1'], result_mean['rouge2'], result_mean['rougeL']])
                        metrics['by_prompt_and_temperature'][prompt][temperature] = result_gmean
                except (ZeroDivisionError, IndexError):
                    continue

        for model in models:
            for prompt in prompts:
                for temperature in temperatures:
                    subset_df = df[
                        (df['model'] == model) & (df['prompt'] == prompt) & (df['temperature'] == temperature)
                    ]
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
                                predictions=predictions_,
                                references=references_,
                                use_stemmer=False,
                                use_aggregator=False
                            )
                            result_mean = {k: np.mean(v) for k, v in result.items()}
                            result_gmean = gmean([result_mean['rouge1'], result_mean['rouge2'], result_mean['rougeL']])
                            metrics['by_model_prompt_temperature'][(model, prompt, temperature)] = result_gmean
                    except (ZeroDivisionError, IndexError):
                        continue
        printable_df = pd.DataFrame.from_dict({'by_model_prompt_temperature': metrics['by_model_prompt_temperature']})

    return metrics, printable_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="The task to use. Can be 'Summarisation' or 'Simplification'",
        choices=["Summarisation", "Simplification"], required=True
    )
    parser.add_argument(
        "-p", "--generate_plot",
        help="Whether to generate a heatmap with the SARI or ROUGE scores for each model and prompt.",
        action='store_true',  # Store True when flag is present
        default=False  # Default value is False
    )
    args = parser.parse_args()
    TASK = args.task  # change task as appropriate
    GENERATE_PLOT = args.generate_plot
    if TASK == 'Simplification':
        METRIC = 'sari'
        with open('data/newsela-auto/newsela-auto/ACL2020/test_dedup_sample.src', 'r') as f:
            inputs = f.readlines()
            inputs = [i.strip("\n") for i in inputs]
        with open('data/newsela-auto/newsela-auto/ACL2020/test_dedup_sample.dst', 'r') as f:
            references = f.readlines()
            references = [i.strip("\n") for i in references]
        assert len(inputs) == len(references)
        with open('data/outputs/output_simplification_all.json', 'r') as f:
            model_outputs = json.load(f)
    elif TASK == 'Summarisation':
        METRIC = 'rouge'
        df_summ = pd.read_csv('data/CNNDailyMail/test_subset_3000.csv')
        inputs = df_summ['article']
        references = df_summ['highlights']
        with open('data/outputs/output_summarization_all.json', 'r') as f:
            model_outputs = json.load(f)
    else:
        raise ValueError(f"TASK must be 'Summarisation' or 'Simplification', git '{TASK}' instead.")
    inputs_references_ = {k: v for k, v in zip(inputs, references)}
    df_ = pd.DataFrame.from_dict(model_outputs)
    metrics_, printable_df_ = calculate_sari_or_rouge(METRIC, df_, inputs_references_)
    print(metrics_)
    if GENERATE_PLOT:
        df_ = pd.DataFrame.from_dict({k: [v] for k, v in metrics_['by_model'].items()})
        fig, axs = plt.subplots(ncols=1, nrows=3, gridspec_kw=dict(width_ratios=[5]))
        sns.heatmap(df_.transpose(), annot=True, fmt=".2f", cmap=sns.cm.rocket_r, ax=axs[0])
        df_ = pd.DataFrame.from_dict({k: [v] for k, v in metrics_['by_prompt'].items()})
        sns.heatmap(df_.transpose(), annot=True, fmt=".2f", cmap=sns.cm.rocket_r, ax=axs[1])
        sns.heatmap(
            pd.DataFrame.from_dict(metrics_['by_model_and_prompt']), annot=True, fmt=".2f", cmap=sns.cm.rocket_r,
            ax=axs[2]
        )
        plt.gcf().set_size_inches(4 * 1, 4 * 3)
        plt.show()
