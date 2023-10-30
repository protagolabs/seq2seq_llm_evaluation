"""
This script splits the data for the human evaluation study into small chunks which can be annotated in one hour each.
We allocated 25 samples per hour for text simplification and GEC (equivalent 4 hours per task), and 12-13 sample
per hours for text summarisation due to the much longer average sample length (equivalent to 8 hours for this task).
"""
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('data/potato/text_simplification_potato.csv')
    chunk_size = 100 / 4
    for i in range(4):
        df.iloc[int(chunk_size * i):int(chunk_size * (i + 1))].to_csv(
            f'data/potato/text_simplification_potato_part-{i+1}-of-4.csv', index=False
        )
    df_concat = pd.concat(
        [pd.read_csv(f'data/potato/text_simplification_potato_part-{i+1}-of-4.csv') for i in range(4)],
        ignore_index=True
    )
    pd.testing.assert_frame_equal(df, df_concat)

    df = pd.read_csv('data/potato/text_summarization_potato.csv')
    chunk_size = 100 / 8
    for i in range(8):
        df.iloc[int(chunk_size * i):int(chunk_size * (i + 1))].to_csv(
            f'data/potato/text_summarization_potato_part-{i+1}-of-8.csv', index=False
        )
    df_concat = pd.concat(
        [pd.read_csv(f'data/potato/text_summarization_potato_part-{i+1}-of-8.csv') for i in range(8)],
        ignore_index=True
    )
    pd.testing.assert_frame_equal(df, df_concat)

    df = pd.read_csv('data/potato/gec_potato.csv')
    df.drop('davinci-instruct-beta', axis=1, inplace=True)
    chunk_size = 100 / 4
    for i in range(4):
        df.iloc[int(chunk_size * i):int(chunk_size * (i + 1))].to_csv(
            f'data/potato/gec_potato_part-{i + 1}-of-4.csv', index=False
        )
    df_concat = pd.concat(
        [pd.read_csv(f'data/potato/gec_potato_part-{i + 1}-of-4.csv') for i in range(4)],
        ignore_index=True
    )
    pd.testing.assert_frame_equal(df, df_concat)
