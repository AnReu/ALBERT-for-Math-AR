from transformers import AutoTokenizer, AutoModel
import torch

additional_token_file_name = 'latex_tokens.txt' # file with newline separated characters
model_path = 'albert-base-v2' # huggingface model path, e.g. 'bert-base-cased'

latex_token = open(additional_token_file_name).read().split('\n')
print(len(latex_token), latex_token[:5])

tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

tok.add_tokens(latex_token)
model.resize_token_embeddings(len(tok))

tok.save_pretrained(f'tokenizer_{model_path}_with_latex')
model.save_pretrained(f'model_{model_path}_with_latex')
