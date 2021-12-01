import sys
use_comet = sys.argv[1] == 'use_comet' if len(sys.argv) > 1 else False
if use_comet:
    import comet_ml

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from datasets import load_metric
import datetime

experiment_start = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S:%f")
out_dir = 'models/task1/' + experiment_start

model_path = 'albert-base-v2'
tokenizer_path = 'albert-base-v2'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

dataset = load_dataset('csv', data_files={'train': '../task1/training_files/arqmath_task1_train.csv',
                                          'dev': '../task1/training_files/arqmath_task1_dev.csv'})
print('-- First training example --')
print(dataset['train'][0])
print('-- First evaluation example --')
print(dataset['dev'][0])


def tokenize(examples):
    return tokenizer(examples["question"], examples['answer'], padding="max_length", truncation=True)


metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenized_dataset = dataset.map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

training_args = TrainingArguments("task1_trainer", evaluation_strategy="epoch",
                                  warmup_steps=200,
                                  per_device_train_batch_size=32,
                                  per_device_eval_batch_size=32,
                                  learning_rate=2e-05,
                                  disable_tqdm=True,
                                  num_train_epochs=3)

trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["dev"], compute_metrics=compute_metrics
)


trainer.train()
trainer.evaluate()
import sys
use_comet = sys.argv[1] == 'use_comet' if len(sys.argv) > 1 else False
if use_comet:
    import comet_ml
if use_comet:
    experiment = comet_ml.config.get_global_experiment()
    experiment.log_parameters({
        'model_save_dir': out_dir,
        'experiment_start': experiment_start
    })
    experiment.log_parameters(training_args.to_sanitized_dict(), prefix='train_args/')

model.save_pretrained(out_dir, push_to_hub=False)
