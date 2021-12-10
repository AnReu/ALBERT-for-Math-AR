import pandas as pd
import os
from tqdm import tqdm

model = '13B' # name of the dir of the model outputs we want to evaluate
skip=[]
out_dir = 'submission'
data_path = 'evaluation_files/one/2020'
out_file_name = f'{model}-task1-auto-text.tsv'
topics = [f'A.{i}' for i in [1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 29, 30, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 72, 74, 75, 77, 79, 80, 83, 85, 86, 87, 88, 89, 90, 93, 96, 98, 99]]
complete_df = None
progress = tqdm(topics)
for topic in progress:
	if topic in skip:
		continue
	progress.set_description(topic)
	df_results = pd.read_csv(f'{data_path}/{topic}/{model}/results.txt', sep='\t', header=None)
	df_test_set = pd.read_csv(open(f'{data_path}/{topic}/data/test.csv', encoding='utf-8'))
	if len(df_results) != len(df_test_set):
		print('different length', topic)
		exit()
	df = pd.concat([df_results, df_test_set], axis=1)
	df['rank'] = df[1].rank(ascending=False).astype(int)
	df['id_a'] = df['id_a'].astype(int)
	df = df[df['rank']<1000]
	df['run'] = model
	df = df[['id_q', 'id_a', 'rank', 1, 'run']]
	if complete_df is not None:
		complete_df = pd.concat([complete_df, df], axis=0)
	else:
		complete_df = df
complete_df.to_csv(f'{out_dir}/{out_file_name}', sep='\t', header=False, index=False)
