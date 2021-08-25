import json
from os import makedirs

from custom_tokenize import split, divide

data_path = '../data_processing'
makedirs(f'{data_path}/albert_data_unprocessed', exist_ok=True)
data = json.load(open(f'{data_path}/cleaned_with_links.json', encoding='utf-8'))
out_path = data_path + '/albert_data_unprocessed/albert_data_{i}.txt'

puncts = ['?', '!', ':', ';']  # punctuation except . as end of sentence symbol
one_thread_one_doc = False  # whether we add a new line between answers of one question or not (False = add newline, one post one doc)
splits = 1

batched_data = divide(data, splits)
for i, data_split in enumerate(batched_data):
    print(f'Processing split {i}/{splits}')
    with open(out_path.format(i=i), 'w', encoding='utf-8') as f:
        for elem in data_split:
            title = elem['title']
            for p in puncts:  # remove punctuations also from title
                title = title.replace(p, '')
            question_sp = split(elem['question'])
            if 'answers' in elem:
                answers_sp = [split(a) for a in elem['answers']]
            else:
                answers_sp = []
            f.write(title + '\n')
            for sent in question_sp:
                f.write(sent)
            for answer_sp in answers_sp:
                if not one_thread_one_doc:
                    f.write('\n')
                for answer in answer_sp:
                    f.write(answer)
            f.write('\n')  # emtpy line between docs
