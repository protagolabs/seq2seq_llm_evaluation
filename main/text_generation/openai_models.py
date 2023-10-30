"""
This script is used to generate the outputs for all OpenAI models except ChatGPT, for all three NLP tasks:
text summarisation, simplification and grammatical error correction. To generate outputs using ChatGPT,
see main/text_generation/chatgpt.py
"""
import os
import json
import time
import argparse
from datetime import datetime
import openai
import pandas as pd
import pytz

openai.api_key = os.getenv("OPENAI_API_KEY")

models_to_run = [
    "text-davinci-003",
    # "text-davinci-001",
    "davinci-instruct-beta",
    # "curie-instruct-beta"
]

simplification_prompts = [
    "Simplify the following text. [...]",
    "Simplify the following text. [...] \n The simplified version is: ",
    "Explain this to a 5 year old. [...]",
    "Explain this to a 5 year old. [...] \n The explanation to a 5 year old could be: ",
    "This is the main story: [...]\n The simplified version of the story is: "
]

summarisation_prompts = [
    "Summarize the following text.[...] \n The summary is:",
    "[...] \n Summarize the text above.",
    "Summarize the following text. [...] \n The very short summary is:",
    "This is the main story: [...]\n The summarized version of the story is:"
]

gec_prompts = [
    "Reply with a corrected version of the input sentence with all grammatical and spelling errors fixed. If there "
    "are no errors, reply with a copy of the original sentence. \n\n Input sentence: [...] \n Corrected sentence: ",
    "Correct the following to standard English: \n\n Sentence: [...] \n Correction:",
]


temperatures = [0, 0.5, 0.7]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="The task to use. Can be 'Summarisation', 'Simplification' or 'GEC'.",
        choices=["Summarisation", "Simplification", "GEC"], required=True
    )
    args = parser.parse_args()
    TASK = args.task  # change task as appropriate
    time_now = datetime.now(pytz.timezone('Europe/London')).strftime('%Y-%m-%dT%H:%M:%S')
    if TASK == 'Simplification':
        with open('data/newsela-auto/newsela-auto/ACL2020/test_dedup_sample.src', 'r') as f:
            inputs = f.readlines()
            inputs = [i.strip("\n") for i in inputs]
        prompts = simplification_prompts
    elif TASK == 'Summarisation':
        df = pd.read_csv('data/CNNDailyMail/test_subset_3000.csv')
        inputs = df['article']
        # inputs_1506 = df['article_trunc1506']
        prompts = summarisation_prompts
    elif TASK == 'GEC':
        with open('data/BEA_website/Write & Improve/origin.txt', 'r') as f:
            inputs = f.readlines()
            inputs = [i.strip("\n") for i in inputs]
        prompts = gec_prompts
    else:
        raise ValueError(f"TASK should be 'Summarisation', 'Simplification' or 'GEC'. Got '{TASK}' instead.")

    results = {"model": [], "prompt": [], "temperature": [], "input": [], "original_input": [], "output": []}

    total_requests = 0
    for i, text_input in enumerate(inputs):
        print(f"INFERENCE ON SAMPLE {i} OF {len(inputs)}")
        for model_name in models_to_run:
            for prompt in prompts:
                for temperature in temperatures:
                    results['model'].append(model_name)
                    results['prompt'].append(prompt)
                    results['temperature'].append(temperature)
                    text = prompt.replace("[...]", text_input)
                    results['input'].append(text)
                    results['original_input'].append(text_input)
                    inference_not_done = True
                    while inference_not_done:
                        try:
                            response = openai.Completion.create(
                                model=model_name,
                                prompt=text,
                                temperature=temperature,
                                max_tokens=128,
                                # top_p=1,
                                # frequency_penalty=0.1,
                                # presence_penalty=0
                            )
                            inference_not_done = False
                        except Exception as e:  # pragmatic catch all exception is not ideal, but works well for now as
                            # we don't know which error OpenAI API will throw (it is still unstable and can throw many
                            # different errors). We retry after 10 minutes as often the OpenAI server will start
                            # working again
                            print(f"Waiting 10 minutes, current time: "
                                  f"{datetime.now(pytz.timezone('Europe/London')).isoformat()}")
                            print(f"Error was: {e}")
                            time.sleep(600)
                    results["output"].append(response['choices'][0]['text'].strip("\n"))
                    total_requests += 1
                    if total_requests % 18 == 0:
                        time.sleep(150)  # to avoid reaching rate limit
        with open(f"data/outputs/output_openai_{TASK.lower()}_{time_now}.json", "w") as f:
            json.dump(results, f)
    df = pd.DataFrame.from_dict(results)
    df.to_csv(f"data/outputs/output_openai_{TASK.lower()}.csv", index=False)
