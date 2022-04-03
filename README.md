# ALBERT for Mathematical Answer Retrieval

This repository contains the code for the paper "An ALBERT-based Similarity Measure for Mathematical Answer Retrieval" [SIGIR 2021]. 
We have re-written the code using huggingface for a better compatibility and adaptability. Let me know, if you have any questions!

# Pre-Training
## Data Pre-Processing

1. Execute `mkdir raw_data` inside this repository
2. Put ARQMath XML files in the directory `raw_data`
3. Execute 
   1. `pip install -r requirements.txt`
   2. `preprocess_pretrain.sh` (Might take some time)

This will build the data needed for pre-training in both set ups (normal and separated). The data will be saved in the directory `data_processing`. 

Besides a json with all posts, the files contain the pre-processed posts, separated by a new line between each post. This data can also be used by the Google ALBERT repository scripts for tokenization and pre-training.

## Training

1. Execute:
   ```
   cd training
   python3 get_hf_dataset.py albert_data_unprocessed albert_data_tokenized # generates SOP dataset from line-separated pretraining data
   python3 pretrain.py # starts pretraining ALBERT   
   ```

`get_hf_dataset.py` will create the SOP labels used during pre-training as well as tokenize the data and convert it to tensors. This step will take some time (several hours) depending on your setup. In the script currently the ALBERT tokenizer is used. If you want to train BERT or any other model, you need to change it in this script.

As input it takes a directory which potentially can contain multiple file. All these files will then be processed as input data. 
The script will output two datasets, one validation set and one training set. They will be created in the output_dir. It will also output an info.json which contains the parameters to generate the dataset such as the input datapath and the tokenizer.

Options: 
* `get_hf_dataset.py <input dir> <output dir>`, both dirs should/will be located in `data_processing/`

`pretrain.py` will start the pretraining of ALBERT. It will automatically load the tokenizer that was used to create the dataset. The dataset along with other information such as which model to use are specified in the script itself.
In order to change the model, make sure that the SOP labels for your model are actually called sentence_order_label otherwise the huggingface Trainer will throw an error. For example, BERT calls them next_sentence_label. Apart from this, you can simply use any huggingface model, that has a masked language model head and classification head. 

The `tokenized_data_dir` needs to be the output dir of the get_hf_dataset.py above as it relies on the files outputted byt the script.

If `use_comet` is provided as a flag the experiment will be logged to your comet.ml dashboard. After the experiment is done, it will upload all hyperparamters to the experiment as well. Note that you need comet_ml installed and the environment variables for your project and api key set before you run the script.

Options: 
* Edit `pretrain.py` to change pretraining hyperparameters
* `pretrain.py use_comet` to log to comet_ml (needs to be installed)
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

This creates the data for fine-tuning on Task 1: a csv file with one column for the question, one for an answer and one for the label (0 = non relevant and 1 = relevant).
For details on how we chose the non relevant answers, please refer to one of the papers.
## Training
Execute `finetune_task1.sh`, we use the fine-tuning parameters of our two publications. For adjustments like the model or data paths edit `training/finetune.py`. As for the pre-training logging with comet.ml is also available using the `use_comet` option. 


The model gets the question-answer pairs created above as its input. Its objective is to classify whether the answer is relevant to the question or not. 
## Evaluation
Once the model is fine-tuned, you can evaluate it on the ARQMath data. 
1. Execute:
```
cd task1
python evaluate.py <path to your model> <model identifier> <topic id>
```
This will evaluate the topic with the specified topic id against all answers in the corpus.
It will write out a file that contains the relevance scores for each answer in the order they were passed to the model. 

2. Execute:
```
python get_best_for_each_topic.py
```
to get a ranking in the submission format for ARQMath. It will be in the submission directory. In the script you need to specify which model you want to create the ranking file for (`model = <model identifier>`).

## Use the fine-tuned model yourself

We re-trained our ALBERT Base 750k model using the code in this repository and uploaded it [here](https://huggingface.co/AnReu/albert-for-math-ar-base-ft).
If you use our uploaded model or your own finetuned model, you can simply use the huggingface transformer library:

```python
# based on https://huggingface.co/docs/transformers/main/en/task_summary#sequence-classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AutoModelForSequenceClassification.from_pretrained("AnReu/albert-for-math-ar-base-ft")

classes = ["non relevant", "relevant"]

sequence_0 = "How can I calculate x in $3x = 5$"
sequence_1 = "Just divide by 3: $x = \\frac{5}{3}$"
sequence_2 = "The general rule for squaring a sum is $(a+b)^2=a^2+2ab+b^2$"

# The tokenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to
# the sequence, as well as compute the attention masks.
irrelevant = tokenizer(sequence_0, sequence_2, return_tensors="pt")
relevant = tokenizer(sequence_0, sequence_1, return_tensors="pt")

irrelevant_classification_logits = model(**irrelevant).logits
relevant_classification_logits = model(**relevant).logits

irrelevant_results = torch.softmax(irrelevant_classification_logits, dim=1).tolist()[0]
relevant_results = torch.softmax(relevant_classification_logits, dim=1).tolist()[0]

# Should be irrelevant
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(irrelevant_results[i] * 100))}%")

# Should be relevant
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(relevant_results[i] * 100))}%")
```

This code first tokenizes the query (sequence_0) together with either a relevant answer and a non relevant answer. The two question-answer pairs are provided as model input. The model outputs logits which need to be converted in to probabilities (using the softmax function). The results for both answers are shown in the end.

We also created a Google Colab with this inference code: [Colab Link](https://colab.research.google.com/drive/11XYJ5KHNkndD2Cup6X3Q1EOoxOqXkFFe?usp=sharing)
## Reference

If you find the code in this repository useful or use the provided model, please consider referencing our papers:

```
@inproceedings{reusch2021tu_dbs,
  title={TU\_DBS in the ARQMath Lab 2021, CLEF},
  author={Reusch, Anja and Thiele, Maik and Lehner, Wolfgang},
  year={2021},
  organization={CLEF}
}

@inproceedings{reusch2021albert,
  title={An albert-based similarity measure for mathematical answer retrieval},
  author={Reusch, Anja and Thiele, Maik and Lehner, Wolfgang},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1593--1597},
  year={2021}
}
```

