"""
This script adds a unique ID to each sample used for human evaluation. This is required in the Potato annotation
tool so that we can match the scores given by reviewers to each sample.
"""
import hashlib
import pandas as pd


def get_filename(sentence: str) -> str:
    # assuming leading/trailing whitespace doesn't matter, nor does case
    sentence_norm = sentence.lower().strip()
    return hashlib.sha256(sentence_norm.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    df_simp = pd.read_csv('data/potato/text_simplification_potato.csv')
    df_simp['id'] = df_simp['original_input'].apply(lambda x: get_filename(x))
    df_simp.to_csv('data/potato/text_simplification_potato.csv', index=False)

    df_summ = pd.read_csv('data/potato/text_summarization_potato.csv')
    df_summ['id'] = df_summ['original_input'].apply(lambda x: get_filename(x))
    df_summ.to_csv('data/potato/text_summarization_potato.csv', index=False)

    df_gec = pd.read_csv('data/potato/gec_potato.csv')
    df_gec['id'] = df_gec['original_input'].apply(lambda x: get_filename(x))
    df_gec.to_csv('data/potato/gec_potato.csv', index=False)
