"""
This module contains the code to reproduce the results shown in Table 6 of the paper. The outputs are already saved
in the data/outputs/potato_aggregated folder as three csv files, one per task. Table 6 represents the combination
of these csv files.
"""
import os
import ndjson
import pandas as pd
import numpy as np


def generate_stats_csv(
        main_directory: str,
        data_annotation_folders: list[str],
        task: str,
        no_of_studies: int,
        annotator_ids: list[str],
        metrics: dict[str, list[str]],
        models: dict[str, list[str]]
) -> pd.DataFrame:
    """Generates a csv file with the results shown in Table 6 of the paper, i.e. the average
    and standard deviation of the scores given by each human annotator and GPT-4 per model,
    task and metrics, across all analysed samples. Needs to be run once per task

    Args:
        main_directory (str): the directory containing all the outputs from the human and GPT-4 evaluation study
        data_annotation_folders (list[str]): a list of subfolders for each specific task (the human annotation study
        was split into multiple one-hour chunks, and each sub-study has results in a different folder.
        task (str): the task, either "summarization", "simplification" or "gec"
        no_of_studies (int): how many folders there are for the selected task (one per human evaluation study) - we
        used 4 one-hour studies for simplification and GEC and 8 one-hour studies for summarisation
        annotator_ids (list[str]): a list of annotator IDs (from the Prolific platform)
        metrics (dict[str, list[str]]): a dictionary containing as keys the possible tasks, and as value a list
        with all the metrics used for each task for the human evaluation study
        models (dict[str, list[str]]): a dictionary containing as keys the possible tasks, and as value a list
        with all the models used for each task for the human evaluation study

    Returns:
        pd.DataFrame: the DataFrame containing the results used for Table 6 in the paper, which can be saved as a csv
    """
    assert any(task == i for i in ["summarization", "simplification", "gec"])
    with open(
            os.path.join(main_directory, f"{task}_evaluation_1_of_{no_of_studies}/gpt4_as_annotator/output.jsonl"), "r"
    ) as f:
        # unlike for human annotators, for GPT-4 annotations, all samples are in the first folder
        gpt4_outputs = ndjson.load(f)
    annotations = {}
    for annotator in annotator_ids:
        results = {
            key: []
            for key in [item for list_ in [
                [metric + ('_' + str(i + 1) if i > 0 else '') for metric in metrics[task]] for i in range(4)
            ] for item in list_]
        }  # dictionary of empty list as values, which will be filled with annotator evaluations

        full_outputs = []
        for folder in data_annotation_folders:
            with open(os.path.join(main_directory, folder, annotator, 'annotated_instances.jsonl'), 'r') as f:
                human_annotators_outputs = ndjson.load(f)
            id_outputs = {i['id']: i for i in human_annotators_outputs}
            with open(os.path.join(main_directory, folder, annotator, 'annotation_order.txt'), 'r') as f:
                order = [i.strip('\n') for i in f.readlines()]
                order = [i for i in order if not i.endswith('.html') and not i.startswith('attention')]
            ordered_outputs = [id_outputs[key] for key in order]
            full_outputs.extend(ordered_outputs)
        for sample in full_outputs:
            for metric in results.keys():
                rating = sample['label_annotations'][metric]
                assert len(rating) == 1
                assert list(rating.values())[0] in ['1', '2', '3', '4', '5']
                results[metric].append(int(list(rating.values())[0]))
        annotations[annotator] = results

    gpt4_annotations = {
        key: []
        for key in [item for list_ in [
            [metric + ('_' + str(i + 1) if i > 0 else '') for metric in metrics[task]] for i in range(4)
        ] for item in list_]
    }  # dictionary of empty list as values, which will be filled with annotator evaluations
    list_for_iteration = [elem for j in [[model] * len(metrics[task]) for model in models[task]] for elem in j]
    for sample in gpt4_outputs:
        for metric, model in zip(gpt4_annotations.keys(), list_for_iteration):
            if metric.split("_")[0] not in ['grammaticality', 'overcorrection']:
                # these two metrics have non-numerical scales
                gpt4_annotations[metric].append(round(sample[model][metric.split("_")[0]]))
            else:
                gpt4_annotations[metric].append(sample[model][metric.split("_")[0]])

    annotations['gpt4_model'] = gpt4_annotations
    if task == 'gec':  # required as GEC task has different evaluation criteria for grammaticality and overcorrection
        for annotator, results_ in annotations.items():
            for metric, values in results_.items():
                if metric.startswith('grammaticality') and annotator != 'gpt4_model':
                    array = np.array(values)
                    assert set(array) - {1, 2, 3} == set()
                    array[array == 1] = 0
                    array[array == 2] = 1
                    array[array == 3] = 2
                    annotations[annotator][metric] = array.tolist()
                if metric.startswith('grammaticality') and annotator == 'gpt4_model':
                    array = np.array(values)
                    assert set(array) - {'0', '1', '2 or more'} == set()
                    array[array == '0'] = 0
                    array[array == '1'] = 1
                    array[array == '2 or more'] = 2
                    annotations[annotator][metric] = [eval(i) for i in array]

                if metric.startswith('overcorrection') and annotator != 'gpt4_model':
                    array = np.array(values)
                    assert set(array) - {1, 2, 3, 4} == set()
                    array[array == 1] = 0
                    array[array == 2] = 1
                    array[array == 3] = 2
                    array[array == 4] = 3
                    annotations[annotator][metric] = array.tolist()
                if metric.startswith('overcorrection') and annotator == 'gpt4_model':
                    array = np.array(values)
                    assert set(array) - {
                        'No', 'Minor over-correction', 'Moderate over-correction', 'Substantial over-correction'
                    } == set()
                    array[array == 'No'] = 0
                    array[array == 'Minor over-correction'] = 1
                    array[array == 'Moderate over-correction'] = 2
                    array[array == 'Substantial over-correction'] = 3
                    annotations[annotator][metric] = [eval(i) for i in array]
    averages = {}
    for annotator, annotation in annotations.items():
        averages['Average ' + annotator[:5]] = {k: np.mean(v) for k, v in annotation.items()}
        averages['Standard Deviation ' + annotator[:5]] = {k: np.std(v) for k, v in annotation.items()}
    table = pd.DataFrame.from_dict(averages)
    return table


if __name__ == "__main__":
    annotator_ids = ['57e2b7c38e00270001b43aba', '5995e3cf9845ef00014de380', '61102f9292f5e6301b5bea45']
    metrics = {
        "summarization": ["relevance", "fluency", "coherence", "consistency"],
        "simplification": ["semantics", "fluency", "simplicity"],
        "gec": ["semantics", "grammaticality", "overcorrection"]
    }
    models = {
        "summarization": ['gold_reference', 'bigscience/T0pp', 'text-davinci-003', 'gpt-3.5-turbo'],
        "simplification": ['gold_reference', 'google/flan-t5-xxl', 'davinci-instruct-beta', 'gpt-3.5-turbo'],
        "gec": ['gold_reference', 'opt-iml-max-30b', 'text-davinci-003', 'gpt-3.5-turbo']
    }
    rename_tables = {  # this is a table to convert the nomenclature used in the backend of the Potato tool used for
        # human annotation, to a pretty and easily understandable name.
        "summarization": {
            'relevance': 'RELEVANCE - gold reference',
            'relevance_2': 'RELEVANCE - bigscience/T0pp',
            'relevance_3': 'RELEVANCE - text-davinci-003',
            'relevance_4': 'RELEVANCE - gpt-3.5-turbo',

            'fluency': 'FLUENCY - gold reference',
            'fluency_2': 'FLUENCY - bigscience/T0pp',
            'fluency_3': 'FLUENCY - text-davinci-003',
            'fluency_4': 'FLUENCY - gpt-3.5-turbo',

            'coherence': 'COHERENCE - gold reference',
            'coherence_2': 'COHERENCE - bigscience/T0pp',
            'coherence_3': 'COHERENCE - text-davinci-003',
            'coherence_4': 'COHERENCE - gpt-3.5-turbo',

            'consistency': 'CONSISTENCY - gold reference',
            'consistency_2': 'CONSISTENCY - bigscience/T0pp',
            'consistency_3': 'CONSISTENCY - text-davinci-003',
            'consistency_4': 'CONSISTENCY - gpt-3.5-turbo',
        },
        "simplification": {
            'semantics': 'SEMANTICS - gold reference',
            'semantics_2': 'SEMANTICS - google/flan-t5-xxl',
            'semantics_3': 'SEMANTICS - davinci-instruct-beta',
            'semantics_4': 'SEMANTICS - gpt-3.5-turbo',

            'fluency': 'FLUENCY - gold reference',
            'fluency_2': 'FLUENCY - google/flan-t5-xxl',
            'fluency_3': 'FLUENCY - davinci-instruct-beta',
            'fluency_4': 'FLUENCY - gpt-3.5-turbo',

            'simplicity': 'SIMPLICITY - gold reference',
            'simplicity_2': 'SIMPLICITY - google/flan-t5-xxl',
            'simplicity_3': 'SIMPLICITY - davinci-instruct-beta',
            'simplicity_4': 'SIMPLICITY - gpt-3.5-turbo',
        },
        "gec": {
            'semantics': 'SEMANTICS - gold reference',
            'semantics_2': 'SEMANTICS - opt-iml-max-30b',
            'semantics_3': 'SEMANTICS - text-davinci-003',
            'semantics_4': 'SEMANTICS - gpt-3.5-turbo',

            'grammaticality': 'GRAMMATICALITY - gold reference',
            'grammaticality_2': 'GRAMMATICALITY - opt-iml-max-30b',
            'grammaticality_3': 'GRAMMATICALITY - text-davinci-003',
            'grammaticality_4': 'GRAMMATICALITY - gpt-3.5-turbo',

            'overcorrection': 'OVERCORRECTION - gold reference',
            'overcorrection_2': 'OVERCORRECTION - opt-iml-max-30b',
            'overcorrection_3': 'OVERCORRECTION - text-davinci-003',
            'overcorrection_4': 'OVERCORRECTION - gpt-3.5-turbo',
        }
    }

    for task in ["summarization", "simplification", "gec"]:
        number_of_studies = 8 if task == "summarization" else 4
        # summarization samples are longer and we split them into 8
        directory = "data/outputs/potato_and_GPT4_full_annotations"
        folders = [
            f"{task}_evaluation_{i + 1}_of_{number_of_studies}/annotation_output" for i in range(number_of_studies)
        ]
        df = generate_stats_csv(directory, folders, task, number_of_studies, annotator_ids, metrics, models)
        df.rename(index=rename_tables[task], inplace=True)
        df.to_csv(f"data/outputs/potato_aggregated/{task}.csv")
