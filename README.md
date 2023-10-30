# seq2seq_llm_evaluation

This project contains the code used to produce the results for the paper 
[_"Evaluation Metrics in the Era of GPT-4: Reliably
Evaluating Large Language Models on Sequence to Sequence Tasks"_](https://arxiv.org/abs/2310.13800), 
which has been accepted at EMNLP 2023, main conference. It also contains the full instructions provided to human reviewers 
and GPT-4 for model evaluation in the `main/human_and_gpt4_evaluation/instructions_to_human_reviewers_and_gpt4` folder.
For any questions on the code please contact [Andrea Sottana](mailto:andrea.sottana@netmind.ai).

Please use the following bibtex when referencing this paper. This is currently based on the arXiv preprint, 
and will be updated once the peer-reviewed publication link becomes available.
```
@article{sottana2023evaluation,
      title={Evaluation Metrics in the Era of GPT-4: Reliably Evaluating Large Language Models on Sequence to Sequence Tasks}, 
      author={Andrea Sottana and Bin Liang and Kai Zou and Zheng Yuan},
      journal={arXiv preprint arXiv:2310.13800},
      year={2023},
      eprint={2310.13800},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

The project is structured as follows; each python module has a docstring at the top explaining its use case.

- `data`: This folder hosts all the data. As mentioned in the notes below, we have not included raw data for the full study, only some processed data for the human and GPT-4 evaluation sub-study.
- `main`: The folder hosting the main code in the subfolders below.
  - `text_generation`: This folder contains the modules to prompt the LLMs to generate the main outputs to be 
  evaluated for the three tasks, text summarisation, simplification and grammatical error correction.
  - `data_processing`: This folder contains all utils and miscellaneous modules used for data preprocessing. In particular, `newsela_preprocessing.py` 
should be run before running any files in the `text_generation` folder, `merge_outputs.py` should
be run after running the files in the `text_generation` folder and before running the files in the `automatic_evaluation` folder.
Every other file in this folder is used to prepare the data for the human evaluation study using the [Potato annotation tool](https://github.com/davidjurgens/potato).
  - `automatic_evaluation`: This folder contains the code used to reproduce the automatic metrics results, including calculating the T-test between the various distributions.
  - `human_and_gpt4_evaluation`: This folder contains the code used to prompt GPT-4 to evaluate LLMs outputs, and to 
generate the statistics of the human evaluation study which are displayed in the paper, as well as the inter-annotator
agreement.
    - `instructions_to_human_reviewers_and_gpt4`: This subfolder contains the instructions given to human reviewers,
and the prompts used for GPT-4 model-to-model evaluation. The instructions to human reviewers are reported in html files,
as they were included in the UI of the Potato annotation tool. 
The code for the human evaluation UI, which is the Potato code with some minor modifications, is not included in this repository.
  
## Notes
- This codebase as it stands is not sufficient to reproduce our results in full without modifications. 
Sometimes manually changing minor parameters (such as a model's temperature) will be required to reproduce 
the full spectrum of results.

- We have not included the raw data. You will need to source the data files yourself and place 
them in the `data` folder before running the code. All datasets used are open-source, but some must be requested 
directly to the original owners. We have, however, included the human and GPT-4 evaluation outputs in the 
`data/outputs` folder; these can be used to reproduce the human evaluation statistics discussed in our paper.
  - The CNN/Daily Mail dataset for text summarisation can be downloaded from Kaggle (link 
  [here](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)) or HuggingFace (link 
  [here](https://huggingface.co/datasets/cnn_dailymail)).
  - The Newsela dataset for text simplification must be requested via [this link](https://newsela.com/data/).
  - The BEA-2019 Shared Task for grammatical error correction can be downloaded via 
  [this link](https://www.cl.cam.ac.uk/research/nl/bea2019st/). Note that this dataset requires some processing before it
  can be used, and you should follow the instructions in the link above in order to generate the M2 file. We have not 
  included the data processing code where this is provided by the project's authors, unless we made specific 
  modifications required to reproduce our results. We will however expect anyone downloading the BEA-2019 dataset to 
  independently carry out the preprocessing steps as described in the link above and at 
  [this link](https://github.com/chrisjbryant/errant), before attempting to reproduce our results.

- In order to generate a UI for the human evaluation study, we used the [Potato annotation tool](https://github.com/davidjurgens/potato).
We had to make some minor front-end modifications in order to suit our use case. However, as they were minor changes 
largely embedded within a fork of their project, we have not reported this code here. This does not affect
results reproducibility, as the Potato code only generates the UI to display the samples for the human evaluation
study, and every researcher can use their preferred tool for this purpose.
- This project is released under the MIT Licence.
