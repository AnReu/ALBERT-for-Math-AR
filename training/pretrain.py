import sys
use_comet = sys.argv[1] == 'use_comet' if len(sys.argv) > 1 else False
if use_comet:
    import comet_ml


import datetime

import numpy as np
from datasets import load_metric
from transformers import AlbertTokenizerFast, DataCollatorForLanguageModeling
from transformers import AlbertForPreTraining, AlbertConfig
from transformers import Trainer, TrainingArguments
import json

from math_datasets import LineByLineWithSOPTextDataset

data_dir = '../data_processing'
tokenized_data_dir = f'{data_dir}/albert_data_tokenized'

dataset = LineByLineWithSOPTextDataset

model_path = 'albert-base-v2'

tokenizer_info = json.load(open(tokenized_data_dir+'/train/info.json'))
tokenizer_path = tokenizer_info['tokenizer_path']

experiment_start = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S:%f")
out_dir = 'models/pretrain/' + experiment_start

tokenizer = AlbertTokenizerFast.from_pretrained(tokenizer_path)
metric = load_metric("accuracy")

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions_mlm = np.argmax(logits[0], axis=-1)
    labels_filter_mlm = labels[0] != -100
    acc_mlm = metric.compute(predictions=predictions_mlm[labels_filter_mlm], references=labels[0][labels_filter_mlm])

    predictions_sop = np.argmax(logits[1], axis=-1)
    acc_sop = metric.compute(predictions=predictions_sop, references=labels[1])

    return {'acc_mlm': acc_mlm['accuracy'], 'acc_sop': acc_sop['accuracy']}

dataset_sop = dataset(
    load_from_path=tokenized_data_dir+'/train'
)
dataset_sop_eval = dataset(
    load_from_path=tokenized_data_dir+'/eval'
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=f"./trainer/pretrain_{experiment_start}",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=False,
    evaluation_strategy="epoch",
    # eval_steps=50,
    label_names=['labels', 'sentence_order_label']
)

# ALBERT base config: https://tfhub.dev/google/albert_base/1
config = AlbertConfig(attention_probs_dropout_prob=0.1,
                      hidden_act='gelu',
                      hidden_dropout_prob=0.1,
                      embedding_size=128,
                      hidden_size=768,
                      initializer_range=0.02,
                      intermediate_size=3072,
                      max_position_embeddings=512,
                      num_attention_heads=12,
                      num_hidden_layers=12,
                      num_hidden_groups=1,
                      net_structure_type=0,
                      gap_size=0,
                      num_memory_blocks=0,
                      inner_group_num=1,
                      down_scale_factor=1,
                      type_vocab_size=2,
                      vocab_size=30000)
model = AlbertForPreTraining(config)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_sop,
    compute_metrics=compute_metrics,
    eval_dataset=dataset_sop_eval
)
trainer.train()
print(trainer.evaluate())

if use_comet:
    experiment = comet_ml.config.get_global_experiment()
    experiment.log_parameters({
        'model_path': model_path,
        'tokenized_data_dir': tokenized_data_dir,
        'model_save_dir': out_dir,
        'experiment_start': experiment_start
    })
    experiment.log_parameters(tokenizer_info, prefix='dataset/')
    experiment.log_parameters(training_args.to_sanitized_dict(), prefix='train_args/')


model.save_pretrained(out_dir, push_to_hub=False)
