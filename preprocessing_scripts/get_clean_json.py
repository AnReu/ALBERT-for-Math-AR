from ARQMathCode.post_reader_record import DataReaderRecord

from datetime import datetime
from utils import clean_body
from collections import defaultdict
import json

data_path = '../raw_data'
out_dir = '../data_processing'
reader = DataReaderRecord(data_path, version='1.2')
answers = defaultdict(list)


cleaned = []
for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
    print(f'Processing Year - {year}')
    now = datetime.now()
    questions = reader.get_list_of_questions_posted_in_a_year(year)
    for q in questions:
        thread = {'question': clean_body(q.body), 'post_id': q.post_id}
        try:
            thread['title'] = clean_body(q.title)
        except:
            thread['title'] = q.title
            print('could not parse', q.title)
        thread['tags'] = q.tags
        if q.answers:
            thread['answer_ids'] = []
            thread['answers'] = []
            for a in q.answers:
                thread['answer_ids'].append(a.post_id)
                thread['answers'].append(clean_body(a.body))
        thread['related_posts'] = []
        thread['duplicates'] = []
        for p_id, is_duplicate in q.related_post:
            if is_duplicate:
                thread['duplicates'].append(p_id)
            else:
                thread['related_posts'].append(p_id)
        cleaned.append(thread)
    print(f'{year} took: {datetime.now() - now}')
del reader
del questions
json.dump(cleaned, open(f'{out_dir}/cleaned_with_links.json', 'w', encoding='utf-8'))
print('Done saving cleaned json.')
