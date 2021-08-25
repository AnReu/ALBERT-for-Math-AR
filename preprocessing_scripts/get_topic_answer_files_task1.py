import json
from pathlib import Path

data_path = '../data_processing'
task1_path = '../task1'
year = '2021'

topics_file_path = f'{task1_path}/topics_task1_{year}.json'
tag_matching = 'one'  # possible values: one, all, none

topics = json.load(open(topics_file_path, encoding='utf-8'))
cleaned_data = json.load(open(f'{data_path}/cleaned_with_links.json', encoding='utf-8'))

line_pattern = '0\t{id_a}\t{id_b}\t{text_a}\t{text_b}\n'


def matching_tags(tags_a, tags_b):
    # returns whether tags_a and tags_b match or not
    if tag_matching == 'none':
        return True
    elif tag_matching == 'one':
        for tag in tags_a:
            if tag in tags_b:
                return True
        return False
    elif tag_matching == 'all':
        for t_a, t_b in zip(sorted(tags_a), sorted(tags_b)):
            if t_a != t_b:
                return False
        return True
    else:
        raise Exception(
            'Please specify the tag_matching parameter: either match at least one tag (one), all or disable matching '
            'generate all question-answer pairs (none). ')


nl_count = 0
tab_count = 0
p_count = 0
for topic in topics:
    id_a = topic['id']
    out_path = f'{task1_path}/evaluation_files/{tag_matching}/{year}/{id_a.replace(" ", "_")}/MRPC/'
    Path(out_path).mkdir(parents=True, exist_ok=True)
    question = f"{topic['title']} {topic['question']}"
    tags = topic['tags'].split(',')
    lines = ['label\tid_a\tid_b\ttext_a\ttext_b\n']  # first line contains header
    for q in cleaned_data:
        if 'answers' not in q or not matching_tags(tags, q['tags']):
            continue
        for id_b, answer in zip(q['answer_ids'], q['answers']):
            if '\n' in question or '\n' in answer:
                nl_count += 1
                print(topic)
            if '\u2029' in question or '\u2029' in answer:
                p_count += 1
                print(topic['id'])
            if '\t' in question or '\t' in answer:
                tab_count += 1
                question = question.replace('\t', ' ')
                answer = answer.replace('\t', ' ')
                print(topic)
            lines.append(line_pattern.format(id_a=id_a, id_b=id_b, text_a=question, text_b=answer))
    open(f'{out_path}/test.tsv', 'w', encoding='utf-8').writelines(lines)
print(nl_count)
print(tab_count)
print(p_count)
