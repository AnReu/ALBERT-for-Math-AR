from transformers import AutoModelForSequenceClassification, AlbertTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader

import torch
from os import makedirs
from sys import argv

# call: python finetune.py model_patb model_id topic_id

data_path = 'evaluation_files/one/2020/' + argv[3]
raw_dataset = load_dataset('csv', data_files={'test': f'{data_path}/data/test.csv'})
model_path = argv[1]

albert_path = '/scratch/ws/1/s8252120-polbert/Slurm-for-ALBERT_Math/ALBERT-for-Math-AR'
model = AutoModelForSequenceClassification.from_pretrained(f'{albert_path}/{model_path}')
tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')

def tokenize_function(examples):
    return tokenizer(examples["question"], examples['answer'], padding="max_length", truncation=True)

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['answer', 'id_a', 'id_q', 'question'])
tokenized_dataset.set_format("torch")
eval_dataloader = DataLoader(tokenized_dataset['test'], batch_size=8)

sanatized_model_path = model_path.replace('/', '_')
out_path = f'{data_path}/{argv[2]}'
makedirs(out_path, exist_ok=True)
import json
json.dump({'model':model_path, 'tokenizer':'albert-base-v2'}, open(f'{out_path}/info.json', 'w'))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

results = []
with open(f'{out_path}/results.txt', 'w') as out_file:
	for batch in eval_dataloader:
		batch = {k: v.to(device) for k, v in batch.items()}
		with torch.no_grad():
			outputs = model(**batch)

		for logit in outputs.logits:
			out_file.write(f'{logit[0].cpu()}\t{logit[1].cpu()}\n')
