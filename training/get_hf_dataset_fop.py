from math_datasets import LineByLineWithFOPFormulaDataset
from transformers import AlbertTokenizerFast
from os import makedirs
import sys

dataset = LineByLineWithFOPFormulaDataset
untrained_model_path = '/scratch/ws/1/s8252120-polbert/Slurm-for-ALBERT_Math/ALBERT-for-Math-AR/untrained_models'
model_path = f'{untrained_model_path}/tokenizer_albert-base-v2_with_latex'# huggingface model path, e.g., 'albert-base-v2'
tokenizer = AlbertTokenizerFast.from_pretrained(model_path)

data_dir_path = '../data_processing'
input_data_dir = data_dir_path+'/'+sys.argv[1] if len(sys.argv) > 1 else f"{data_dir_path}/test_data"
dataset_path = data_dir_path+'/'+sys.argv[2] if len(sys.argv) > 2 else input_data_dir + '_tokenized'
makedirs(dataset_path, exist_ok=True)


tokenized_data = dataset(
    tokenizer=tokenizer,
    file_dir=input_data_dir,
    block_size=512,
)

tokenized_data.save_train_eval_splits(dataset_path, eval_p=0.05)
