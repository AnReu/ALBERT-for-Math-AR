from lxml import etree
import json
from utils import clean_body

year = '2020'
raw_data_path = '../raw_data'
task1_path = '../task1'
if year == '2020':
    topics_file = 'Topics_V2.0.xml'
elif year == '2021':
    topics_file = 'Topics_Task1_2021_V1.1.xml'

tree = etree.parse(f'{raw_data_path}/{topics_file}')
root = tree.getroot()


topics = []
for topic in root:
    topic_content = {'id': topic.get('number')}
    for elem in topic:
        if elem.tag not in 'Tags':
            content = clean_body(elem.text)
        else:
            content = elem.text
        topic_content[elem.tag.lower()] = content
    topics.append(topic_content)

json.dump(topics, open(f'{task1_path}/topics_task1_{year}.json', 'w', encoding='utf-8'))
