from custom_tokenize import divide, LatexTokenizer

from os import makedirs

data_path = '../data_processing'
data_file = open(f'{data_path}/albert_data_unprocessed/albert_data_0.txt', encoding='utf-8')
out_path = f'{data_path}/albert_data_text_latex_separated_unprocessed'
makedirs(out_path, exist_ok=True)

splits = 1

tok = LatexTokenizer()

separated_lines = []
formula = ''
for j, line in enumerate(data_file.readlines()):
        if '$' not in line:
                separated_lines.append(line)
                continue
        separated_line = []
        before_too_short = False # indicator for short formulas
        line = line.replace('$$','$') # $$ -> $
        line = line.replace('\n', '')
        line = line[:-1] if line.endswith('$') else line
        splitted = line.split('$')
        for i, part in enumerate(splitted):
                if i == 0:
                        separated_line.append(part)
                        continue
                if i%2 == 1: # every second part is a formula
                        if tok.num_of_non_whitespace_tokens(part) > 3: # put latex line in $..$ only if length > 3
                                separated_line.append(f'${part}$') # wrap formulas in $ to indicate that this is forumla, might facilitate formula recognition by the model during finetuning
                                before_too_short = False
                        else:
                                separated_line[-1] += f'${part}$'
                                before_too_short = True
                else:
                        if before_too_short:
                                before_too_short = False
                                separated_line[-1] += part
                        elif len(part.strip()) < 10:
                                separated_line[-1] += part
                        else:
                                separated_line.append(part)
        separated_lines.extend([l.strip() + '\n' for l in separated_line])


final_lines = []

for i, line in enumerate(separated_lines):
        if line.strip() == '\\' and separated_lines[i-1].strip() != '':
                final_lines[-1] = final_lines[-1][:-1] + line
        else:
                final_lines.append(line)

# divide data in 16 files and save to: albert_data_text_latex_separated_unprocessed
# divide(lst, n): # divide list in n parts of equal length (more or less)
print('number of lines in orig. file:', j)
print('number of lines in final files:', len(separated_lines))
for i, batch in enumerate(divide(separated_lines, splits)):
                open(f'{out_path}/albert_data_{i}.txt', 'w', encoding='utf-8').writelines(batch)
