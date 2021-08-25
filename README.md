# ALBERT for Mathematical Answer Retrieval

We will publish the code for the paper "An ALBERT-based Similarity Measure for Mathematical Answer Retrieval" [SIGIR 2021] in this repository soon. 

Currently, we are re-writing the code using huggingface for a better compatibility and adaptability. Let me know, if you have any questions!

# Pre-Training
## Data Pre-Processing

1. Execute `mkdir raw_data` inside this repository
2. Put ARQMath XML files in the directory `raw_data`
3. Execute 
   1. `pip install -r requirements.txt`
   2. `preprocess_pretrain.sh` (Might take some time)

This will build the data needed for pre-training in both set ups (normal and separated). The data will be saved in the directory `data_processing`. Besides a json with all posts, the files contain the pre-processed posts, separated by a new line between each post. This data can be used by the Google ALBERT repository scripts for tokenization and pre-training.

## Training
   [WIP]

# Task 1
## Data Pre-Processing

0. **Skip, if you already ran preprocess_pretain.sh:**
   1. Put ARQMath XML files in the directory `raw_data`
   2. Execute:
      ```
      pip install -r requirements.txt
      cd preprocessing_scripts
      git clone https://github.com/AnReu/ARQMathCode.git
      python3 get_clean_json.py
      ```
1. Execute `preprocess_task1_train.sh` to generate the training data.
2. Put the topics XML file in `raw_data` and adjust the year in the files `get_cleaned_topics_1.py` and `get_topic_answer_files_task1.py`
 in the directory `preprocessing_scripts` if you want to build the evaluation directory for the 2020 topics
3. Execute `preprocess_task1_eval.sh` to build the evaluation directory.

## Training
[WIP]