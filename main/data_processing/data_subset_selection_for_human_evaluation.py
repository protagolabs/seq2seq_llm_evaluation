"""
This script takes the output generated after running merge_outputs.py and creates 3 csv files (one per task / dataset)
containing 100 random samples each. These csv files can then be used for the human evaluation, and loaded onto
the Potato software which will generate a UI for human evaluators to use.
"""
import json
import random
import argparse
import pandas as pd
import tiktoken


def create_df_for_human_eval(
        output_df: pd.DataFrame, task_model_prompt_temperature_dict: dict, task: str
) -> pd.DataFrame:
    """Returns a dataframe which can be used to generate the final input to the Potato annotation tool
    which will display the samples to human annotators

    Args:
        output_df (pd.DataFrame): the outputs obtained after running merge_outputs.py, for example
        "output_summarisation_all.json"
        task_model_prompt_temperature_dict (dict): a dictionary containing the combination of model, prompt and
        temperature for each human evaluation task (the exact dictionary structure is shown below after
        if __name__ == "__main__": statement)
        task (str): the task. Should be 'Summarisation', 'Simplification' or 'GEC'

    Returns:
        pd.DataFrame: the required dataframe.
    """
    selected_model_prompt_temperature = task_model_prompt_temperature_dict[task]
    for model, prompt, temperature in selected_model_prompt_temperature:
        sub_df = output_df[
            (output_df['model'] == model) & (output_df['prompt'] == prompt) & (output_df['temperature'] == temperature)
        ].reset_index(drop=True)
        try:
            df_for_human_eval[model] = sub_df['output']
        except NameError:  # create df_for_human_eval if it doesn't exist
            df_for_human_eval = sub_df.rename({'output': model}, axis=1)
    return df_for_human_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="The task to use. Can be 'Summarisation', 'Simplification' or 'GEC'.",
        choices=["Summarisation", "Simplification", "GEC"], required=True
    )
    args = parser.parse_args()
    TASK = args.task  # change task as appropriate
    enc = tiktoken.encoding_for_model("text-davinci-003")
    NUM_SAMPLES = 100  # the number of random samples to be used for human evaluation per dataset and task
    summarisation_prompt = 'Summarize the following text. [...] \n The summary is:'
    simplification_prompt = 'Simplify the following text. [...] \n The simplified version is: '
    gec_prompt = "Reply with a corrected version of the input sentence with all grammatical and spelling errors " \
                 "fixed. If there are no errors, reply with a copy of the original sentence. \n\n " \
                 "Input sentence: [...] \n Corrected sentence:"
    selected_task_model_prompt_temperature = {
        "Summarisation":
            [
                ('bigscience/T0pp', summarisation_prompt, 0.01),
                ('text-davinci-003', summarisation_prompt, 0.0),
                ('gpt-3.5-turbo', summarisation_prompt, 0.0)
            ],
        "Simplification":
            [
                ('google/flan-t5-xxl', simplification_prompt, 0.01),
                ('davinci-instruct-beta', simplification_prompt, 0.0),
                ('gpt-3.5-turbo', simplification_prompt, 0.0)
            ],
        "GEC":
            [
                ('facebook/opt-iml-max-30b', gec_prompt, 0.01),
                ('text-davinci-003', gec_prompt, 0.0),
                ('gpt-3.5-turbo', gec_prompt, 0.2)
            ]
    }
    if TASK == "Summarisation":
        with open('data/outputs/output_summarization_all.json', 'r') as f:
            summ = json.load(f)
        df = pd.DataFrame.from_dict(summ)
        cnn_dm = pd.read_csv('data/CNNDailyMail/test_subset_3000.csv')

        df_for_human_eval = create_df_for_human_eval(df, selected_task_model_prompt_temperature, TASK)
        df_for_human_eval['gold_reference'] = cnn_dm['highlights']
        assert cnn_dm['article'].tolist() == df_for_human_eval['original_input'].tolist()

        df_for_human_eval = df_for_human_eval[
            ['original_input', 'gold_reference', 'bigscience/T0pp', 'text-davinci-003', 'gpt-3.5-turbo']
        ]  # remove any column that may be unwanted

        summ_inputs = df_for_human_eval['original_input']
        appropriate_indices_for_summ = [i for i, j in enumerate(summ_inputs) if len(enc.encode(j)) <= 500]
        # this step is required for summarisation only, as the dataset has a tail of very long examples; in order to
        # enable reviewers to complete the task in a reasonable time we only ask them to review samples below 500 tokens
        indices = random.sample(range(len(appropriate_indices_for_summ)), NUM_SAMPLES)
        selected_indices_for_summ = [appropriate_indices_for_summ[i] for i in indices]
        subset_summ = df_for_human_eval.iloc[selected_indices_for_summ]
        subset_summ.to_csv('data/potato/text_summarization_potato.csv', index=False)

    elif TASK == "Simplification":
        with open('data/outputs/output_simplification_all.json', 'r') as f:
            simp = json.load(f)
        df = pd.DataFrame.from_dict(simp)
        with open('data/newsela-auto/newsela-auto/ACL2020/test_dedup_sample.src', 'r') as f:
            inputs = f.readlines()
            inputs = [i.strip("\n") for i in inputs]
        with open('data/newsela-auto/newsela-auto/ACL2020/test_dedup_sample.dst', 'r') as f:
            references = f.readlines()
            references = [i.strip("\n") for i in references]
        df_for_human_eval = create_df_for_human_eval(df, selected_task_model_prompt_temperature, TASK)
        df_for_human_eval['gold_reference'] = references

        assert inputs == df_for_human_eval['original_input'].tolist()

        df_for_human_eval = df_for_human_eval[
            ['original_input', 'gold_reference', 'google/flan-t5-xxl', 'davinci-instruct-beta', 'gpt-3.5-turbo']
        ]  # remove any column that may be unwanted
        simp_inputs = df_for_human_eval['original_input']
        indices = random.sample(range(3000), NUM_SAMPLES)  # all 3000 samples are short enough unlike for summarisation
        subset_simp = df_for_human_eval.iloc[indices]
        subset_simp.to_csv('data/potato/text_simplification_potato.csv', index=False)
    elif TASK == 'GEC':
        with open('data/outputs/output_gec_all.json', 'r') as f:
            gec = json.load(f)
        df = pd.DataFrame.from_dict(gec)
        with open('data/BEA_website/Write & Improve/origin.txt', 'r') as f:
            inputs = f.readlines()
            inputs = [i.strip("\n") for i in inputs]
        with open('data/BEA_website/Write & Improve/target.txt', 'r') as f:
            references = f.readlines()
            references = [i.strip("\n") for i in references]
        df_for_human_eval = create_df_for_human_eval(df, selected_task_model_prompt_temperature, TASK)
        df_for_human_eval['gold_reference'] = references

        assert inputs == df_for_human_eval['original_input'].tolist()

        df_for_human_eval = df_for_human_eval[
            ['original_input', 'gold_reference', 'facebook/opt-iml-max-30b', 'text-davinci-003', 'gpt-3.5-turbo']
        ]  # remove any column that may be unwanted
        gec_inputs = df_for_human_eval['original_input']
        indices = random.sample(range(3000), NUM_SAMPLES)  # all 3000 samples are short enough unlike for summarisation
        subset_gec = df_for_human_eval.iloc[indices]
        subset_gec.to_csv('data/potato/GEC_potato.csv', index=False)
    else:
        raise ValueError(f"TASK should be 'Summarisation', 'Simplification' or 'GEC'. Got '{TASK}' instead.")
