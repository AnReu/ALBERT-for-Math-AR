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

## Training
[WIP]