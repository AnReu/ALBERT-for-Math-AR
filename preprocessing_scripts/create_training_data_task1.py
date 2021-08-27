import json
import random
from random import shuffle
from collections import defaultdict
from tqdm import tqdm
from os import makedirs
import pandas as pd

# create training set: one question - two answers, one is correct answer, one is from a different question.
# to make it harder for the model to decide: take question that shares at least one category with the original question

data_path = '../data_processing'
out_path = '../task1/training_files'
train_p = 0.9  # --> valid_p = 1 - train_p, no test set, because ARQMath provides test set

data = json.load(open(f'{data_path}/cleaned_with_links.json', encoding='utf-8'))
makedirs(out_path, exist_ok=True)

# 1. Remove questions without answers
# 2. Group questions by tag
questions_with_answers = defaultdict(list)
for q in data:
    if 'answers' not in q:
        continue  # we only want questions with answers
    for tag in q['tags']:
        questions_with_answers[tag].append(q)

# 3. Check number of questions for each tag
print('Questions with answers, sizes by tag:')
for tag in questions_with_answers:
    print(tag, len(questions_with_answers[tag]))

# 4. For each questions: get one correct answer (random out of all answers of this question) and one incorrect answer with at least one common tag
correct_pairs = []
wrong_pairs = []
for d in tqdm(data):
    if 'answers' in d:
        correct_answer = random.choice(d['answers'])
        correct_pairs.append((d['title'] + ' ' + d['question'] , correct_answer, '1')) # Label 1 for correct question-answer pairs
    tag_a = random.choice(d['tags'])
    try:
        # We choose another question by chance. From this question we will choose the incorrect answer from.
        d_b = random.choice(questions_with_answers[tag_a])
    except:
        # when we find a tag that does not have any questions with answers, then random.choice will throw an error.
        print(tag_a)
        tag_a = random.choice(list(questions_with_answers.keys()))
        d_b = random.choice(questions_with_answers[tag_a])
    # Resolve problem, when the original question is chosen by chance
    while d['post_id'] == d_b['post_id']:
        if len(questions_with_answers[tag_a]) == 1 and len(d['tags']) > 1:
            # try another tag of the question, if the current tag has only one question (which is the question we are trying to find an incorrect answer for)
            tag_a = random.choice(d['tags'])
            if len(questions_with_answers[tag_a]) == 1:
                # if the new tag still has an empty question set: get random tag
                tag_a = random.choice(list(questions_with_answers.keys()))
        elif len(questions_with_answers[tag_a]) == 1:
            # if the question has only one tag, which has only one question: use another random tag
            tag_a = random.choice(list(questions_with_answers.keys()))
        d_b = random.choice(questions_with_answers[tag_a])
    wrong_answer = random.choice(d_b['answers'])
    wrong_pairs.append((d['title'] + ' ' + d['question'], wrong_answer, '0')) # Label 0 for correct question-answer pairs


# 5. Shuffle data and save splits to file
all_pairs = [*correct_pairs, *wrong_pairs]
shuffle(all_pairs)

no_all = len(all_pairs)
no_train = int(no_all * 0.9)
no_val = no_all - no_train


def build_split(split, data_pairs):
    df = pd.DataFrame(data_pairs, columns=['question', 'answer', 'label'])
    df.to_csv(f'{out_path}/arqmath_task1_{split}.csv', index_label='idx')

build_split('train', all_pairs[:no_train])
build_split('dev', all_pairs[no_train:])
build_split('test', [])

print('Done creating training data for task 1.')