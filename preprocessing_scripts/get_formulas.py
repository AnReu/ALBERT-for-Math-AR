from ARQMathCode.post_reader_record import DataReaderRecord

from datetime import datetime
import bs4 as bs
from collections import defaultdict
import json


data_path = '../raw_data'
out_dir = '../data_processing'

reader = DataReaderRecord(data_path, version='1.2')
questions = {}
answers = defaultdict(list)


def get_formulas(body):
    soup = bs.BeautifulSoup(body, "lxml")
    formulas = []
    for math in soup.find_all('span', {'class': "math-container"}):
        formulas.append(math.text)
    return formulas


cleaned = []
for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
    print(f'Year - {year}')
    now = datetime.now()
    questions = reader.get_list_of_questions_posted_in_a_year(year)
    for q in questions:
        thread = {}
        thread['question_formulas'] = get_formulas(q.body)
        thread['post_id'] = q.post_id
        try:
            thread['title_formulas'] = get_formulas(q.title)
        except:
            print('could not parse', q.title)
        thread['tags'] = q.tags
        if q.answers:
            thread['answer_ids'] = []
            thread['answer_formulas'] = []
            for a in q.answers:
                thread['answer_ids'].append(a.post_id)
                thread['answer_formulas'].append(get_formulas(a.body))
        cleaned.append(thread)
    print(f'{year} took: {datetime.now() - now}')
del reader
del questions

json.dump(cleaned, open(f'{out_dir}/cleaned_formulas.json', 'w', encoding='utf-8'))
