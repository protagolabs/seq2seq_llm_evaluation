"""
This script loads the aggregated data from the different human reviewers and GPT-4 and calculates the
inter-annotator agreement expressed through the Krippendorff alpha coefficient.
It also generates the tables with the human and GPT-4 annotation rankings.
Outputs are saved in data/outputs/potato_aggregated
Outputs from this script are shown in Tables 2 and 7 of the paper.
NOTE: We report each annotator using the initials of their Prolific IDs, for example 57e2b, 5995e and 61102.
These and are just unique identifiers for each anonymous annotator.
"""
import json
import pandas as pd
import krippendorff


if __name__ == "__main__":
    krippendorff_alpha_scores = {"summarization": {}, "simplification": {}, "gec": {}}
    rankings = {  # creating placeholders for DataFrames to be populated later
        "summarization": {"RELEVANCE": ..., "FLUENCY": ..., "COHERENCE": ..., "CONSISTENCY": ...},
        "simplification": {"SEMANTICS": ..., "FLUENCY": ..., "SIMPLICITY": ...},
        "gec": {"SEMANTICS": ..., "GRAMMATICALITY": ..., "OVERCORRECTION": ...}
    }

    for task in krippendorff_alpha_scores.keys():
        df = pd.read_csv(f'data/outputs/potato_aggregated/{task}.csv', index_col=0)
        df_to_concat = []
        for metric in rankings[task].keys():
            rankings[task][metric] = df.loc[
                [i for i in df.index if i.startswith(metric)]
            ][
                [j for j in df.columns if j.startswith("Average")]
            ]
            df_to_concat.append(
                rankings[task][metric].rank(ascending=True if metric in ["GRAMMATICALITY", "OVERCORRECTION"] else False)
            )
        pd.concat(df_to_concat).to_csv(f'data/outputs/potato_aggregated/{task}_ranks.csv')
        for label, data in rankings[task].items():
            krippendorff_alpha_scores[task][label] = {}
            reliability_data = [
                data.rank(ascending=False)[i].tolist() for i in ['Average 57e2b', 'Average 5995e', 'Average 61102']
            ]
            reliability_data_withgpt = [
                data.rank(ascending=False)[i].tolist() for i in df.columns if i.startswith("Average")
            ]
            for level in ['nominal', 'ordinal', 'interval']:
                krippendorff_alpha_scores[task][label][f"{level} without GPT-4"] = round(
                    krippendorff.alpha(reliability_data=reliability_data, level_of_measurement=level), 4
                )
                krippendorff_alpha_scores[task][label][f"{level} with GPT-4"] = round(
                    krippendorff.alpha(reliability_data=reliability_data_withgpt, level_of_measurement=level), 4
                )
    with open('data/outputs/potato_aggregated/krippendorff_alpha_scores.json', 'w') as f:
        json.dump(krippendorff_alpha_scores, f, indent=4)
