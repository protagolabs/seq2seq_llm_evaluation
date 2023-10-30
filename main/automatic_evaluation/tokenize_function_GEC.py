"""
This file contains a helper function used to tokenize text in order to compute ERRANT scores for the
grammatical error correction task. For more info see https://www.cl.cam.ac.uk/research/nl/bea2019st/
"""
import spacy


def re_tokenize(file_path: str, out_path: str):
    """Tokenize text in order to compute ERRANT scores for grammatical error correction task.
    For more info see https://www.cl.cam.ac.uk/research/nl/bea2019st/

    Args:
        file_path (str): the prediction .txt file to be tokenized, with one sentence per line
        out_path (str): the tokenized output as a .txt file to be written to disk. One sentence per line, each
        token is separated by a whitespace
    """
    file_path = file_path
    out_path = out_path

    nlp = spacy.load("en_core_web_sm")    # xx_ent_wiki_sm

    with open(file_path, 'r') as f:
        sents = f.readlines()

    new_text = []

    for sent in sents:

        doc = nlp(sent)
        new_sent = ""

        for i in range(len(doc)):
            if i == len(doc) - 1:
                new_sent += doc[i].text
            else:
                new_sent += doc[i].text + " "
        new_text.append(new_sent)

    long_text = ''
    for i in range(len(new_text)):
        long_text += new_text[i]

    with open(out_path, "w") as f:

        f.write(long_text)
