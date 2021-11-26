# one sentence = one formula
# one document = one formula chain?

from os import makedirs
import json
import re

# from tex_lex
t_REL_CLASS = r'=|:=|\\[dD]oteq|\\dot=|\\approxeq|\\backsimeq|\\circeq|\\cong|\\backsim|\\curlyeqprec|\\curlyeqsucc|\\eqslantgtr|\\eqslantless|\\equiv|\\gnsim|\\triangleq|\\eqsim|\\thicksim|\\sim|\\simeq|\\nsim|\\neq|\\not(=|\\equiv)|\\frown|\\between|\\eqcirc|\\smallfrown|\\smallsmile|\\approx|\\asymp|\\ge|\\geq|\\geqq|\\geqslant|\\gg|\\gnapprox|\\gt|>|\\gtrapprox|\\gtrdot|\\gtreqless|\\gtreqqless|\\gtrless|\\gtrsim|\\le^[f]|\\leq|\\leqq|\\leqslant|\\lessapprox|\\lessdot|\\lesssim|\\ll|\\lnapprox|\\lneq|\\lneqq|\\lt|<|\\lvertneqq|\\ncong|\\ne|\\ngeq|\\ngeqq|\\ngeqslant|\\nleq|\\nleqq|\\nleqslant|\\nless|\\nprec|\\npreceq|\\nsucc|\\nsucceq|\\prec|\\preceq|\\succ|\\succapprox|\\succcurlyeq|\\thickapprox|\\trianglelefteq|\\trianglerighteq|\\succeq|\\succnapprox|\\succneqq|\\succnsim|\\succsim|\\unlhd|\\unrhd|\\gneq|\\gneqq|\\gvertneqq|\\ggg|\\gggtr|\\ngtr|\\precapprox|\\preccurlyeq|\\precnapprox|\\precneqq|\\precnsim|\\precsim|\\Cap|\\cap|\\Cup|\\cup|\\curlyvee|\\dashv|\\curlywedge|\\land|\\lor|\\sqcap|\\sqcup|\\vee|\\veebar|\\wedge|\\Join|\\bowtie|\\Subset|\\Supset|\\nsubseteq|\\nsupseteq|\\supseteq|\\subset|\\sqsubset|\\sqsubseteq|\\sqsupset|\\sqsupseteq|\\subseteq|\\subseteqq|\\subsetneq|\\subsetneqq|\\supset|\\supseteq|\\supseteqq|\\supsetneq|\\supsetneqq|\\varsubsetneq|\\varsubsetneqq|\\varsupsetneq|\\varsupsetneqq|\\in|\\ni|\\not\\in|\\owns|\\nparallel|\\parallel|\\propto'

rel_pattern = re.compile(t_REL_CLASS)


data_path = '../data_processing'
formulas_file = json.load(open(f'{data_path}/cleaned_formulas.json', encoding='utf-8'))
out_path = f'{data_path}/albert_data_latex_FOP_unprocessed'
makedirs(out_path, exist_ok=True)

splits = 1


filtered_formulas = []

def split_at_rel(f):
    splitted = []
    matches = rel_pattern.finditer(f)
    next_start = 0
    for match in matches:
        splitted.append(f[next_start: match.end()].strip())
        next_start = match.end()
    if f[next_start:].strip() != '':
        splitted.append(f[next_start:])
    return splitted

for question in formulas_file:
    for f in question['question_formulas']:
        splitted = split_at_rel(f)
        if len(splitted) > 2:
            filtered_formulas.append(splitted)
    for answer in question['question_formulas']:
        for f in answer:
            splitted = split_at_rel(f)
            if len(splitted) > 2:
                filtered_formulas.append(splitted)

print(f'Total len of formulas: {len(filtered_formulas)}')

written_lines = 0
with open(f'{out_path}/rel_split_formulas.txt', 'w', encoding='utf-8') as out_file:
    for formula in filtered_formulas:
        for part in formula:
            out_file.write(part + '\n')
            written_lines += 1
        out_file.write('\n')
print(f'Total number of non-empty files: {written_lines}')
