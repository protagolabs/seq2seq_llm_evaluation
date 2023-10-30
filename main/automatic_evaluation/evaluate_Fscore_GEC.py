"""
Script used to calculate the F0.5 ERRANT score for grammatical error correction.
"""
from main.automatic_evaluation.eval_errant_GEC import write_m2, compute_score
from main.automatic_evaluation.tokenize_function_GEC import re_tokenize


def calculate_f05_score(orig_path: str, cor_path: str, right_path: str) -> float:
    """Function to evaluate the F0.5 ERRANT score.

    Args:
        orig_path (str): the .txt path to the file with the original (uncorrected) sentences
        cor_path (str): the .txt path to the file with the sentences corrected by the model
        right_path (str): the .txt path to the gold standard corrections

    Returns:
        float: The F0.5 ERRANT score
    """
    out_path = cor_path.replace(".txt", ".m2")
    write_m2(orig_path, cor_path, out_path)

    out_path_orig = right_path.replace(".txt", ".m2")
    write_m2(orig_path, right_path, out_path_orig)

    hyp_path = out_path

    score = compute_score(hyp_path, out_path_orig)

    return score


if __name__ == "__main__":
    # Customize
    # The original text
    orig_path = "origin.txt"

    # The corrected text
    right_path = "target.txt"

    # the gold-text
    cor_path = "gold.txt"

    # Step 1: Retokenize
    re_tokenize(orig_path, orig_path)
    re_tokenize(cor_path, right_path)
    re_tokenize(right_path, cor_path)

    # Step 2: Compute the F0.5 score by using the ERRANT
    f05_score = calculate_f05_score(orig_path, cor_path, right_path)

    print(f"The F0.5 score is {f05_score}")
