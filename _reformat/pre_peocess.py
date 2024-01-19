import json
import tqdm
import multiprocessing as mp
import random
import torch
from collections import Counter
import os
random.seed(13)
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original = json.load(open('./PESConv.json'))


problem_list = ['ongoing depression', 'breakup with partner', 'job crisis', 'problems with friends', 'academic pressure']
texts_usr = {
    'ongoing depression': [],
    'breakup with partner': [],
    'job crisis': [],
    'problems with friends': [],
    'academic pressure': [],
    'others': []
}
texts_sys = {
    'ongoing depression': [],
    'breakup with partner': [],
    'job crisis': [],
    'problems with friends': [],
    'academic pressure': [],
    'others': []
}
texts_sys_strategy = {
    'ongoing depression': [],
    'breakup with partner': [],
    'job crisis': [],
    'problems with friends': [],
    'academic pressure': [],
    'others': []
}
strategy = {
    'ongoing depression': [],
    'breakup with partner': [],
    'job crisis': [],
    'problems with friends': [],
    'academic pressure': [],
    'others': []
}
usr_list = []
sys_list = []
str_list = []
sys_list_str = []
with open('train.txt') as f:
    reader = f.readlines()
for data in reader:
    data = eval(data)
    dialog = data['dialog']
    # persona_list = data['persona_list']
    persona = ''
    user_number = 0
    problem = data['problem_type']
    if problem not in problem_list:
        problem = 'others'

    for i in range(len(dialog)):
        if dialog[i]['speaker'] != 'sys':
            user_number += 1

        if i > 0 and dialog[i]['speaker'] == 'sys':
            # if dialog[i - 1]['speaker'] != 'sys' and user_number > 2:
            #     persona = persona_list[user_number - 3]
            #     # persona = persona.replace('<persona> ', '').replace(' <input>', '')
            texts_usr[problem].append(dialog[i-1]['text'])
            texts_sys[problem].append(dialog[i]['text'])
            texts_sys_strategy[problem].append(persona + '[' + dialog[i]['strategy'] + '] ' + dialog[i]['text'])
            # texts_sys_strategy[problem].append('[' + dialog[i]['strategy'] + '] ' + dialog[i]['text'])
            strategy[problem].append(dialog[i]['strategy'])
            # usr_list.append(dialog[i-1]['text'])
            # sys_list.append(dialog[i]['text'])
            # str_list.append(dialog[i]['strategy'])
            # sys_list_str.append(persona + ' [' + dialog[i]['strategy'] + '] ' + dialog[i]['text'])


def _norm(x):
    return ' '.join(x.strip().split())


def add_dpr(data):
    """
    处理数据集，便于数据获取
    """
    dialog = data['dialog']
    persona_list = data['persona_list']
    persona = ""
    user_number = 0
    problem = data['problem_type']
    if problem not in problem_list:
        problem = 'others'

    for i in range(len(dialog)):
        if dialog[i]['speaker'] != 'supporter':
            user_number += 1

        if i > 0 and dialog[i]['speaker'] == 'supporter':
            last_text = _norm(dialog[i - 1]['content'])

            if dialog[i - 1]['speaker'] != 'supporter' and user_number > 2:
                persona = persona_list[user_number - 3]
                # persona = persona.replace('<persona> ', '').replace(' <input>', '')

            with torch.no_grad():
                # 细节还需把控???
                # Internet search???
                encoded_inputs = DPR_tokenizer(
                    questions=[persona + last_text] * len(strategy[problem]),
                    # questions=[last_text] * len(strategy[problem]),
                    # titles=[dialog[i]['annotation']['strategy']] * len(strategy[problem]),
                    texts=texts_sys_strategy[problem],
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=32
                ).to(device)
                # # 可以用全部的数据, 对比一下效果
                # encoded_inputs = DPR_tokenizer(
                #     questions=[last_text] * len(sys_list),
                #     texts=sys_list_str,
                #     return_tensors="pt",
                #     padding='max_length',
                #     truncation=True,
                #     max_length=32
                # ).to(device)
                outputs = DPR_reader(**encoded_inputs)
                relevance_logits = outputs.relevance_logits
                _, indices = torch.topk(relevance_logits, 10)  # 同步修改get_infer_batch中的context
                dialog[i]['dpr'] = [[texts_usr[problem][index]] + [strategy[problem][index]] + [texts_sys[problem][index]] for index in indices]
                # dialog[i]['dpr'] = [[usr_list[index]] + [str_list[index]] + [sys_list[index]] for index in indices]

    data['dialog'] = dialog

    return data


from transformers import DPRReaderTokenizerFast, DPRReader

DPR_tokenizer = DPRReaderTokenizerFast.from_pretrained('../DPR-reader')
DPR_reader = DPRReader.from_pretrained('../DPR-reader').to(device)

data = []
for d in tqdm.tqdm(original, total=len(original)):
    data.append(add_dpr(d))

# DPRConv_1: problem
# DPRConv_2: problem + persona
# DPRConv_3: all
# DPRConv_4: all + persona
# DPRConv_5: top-k
# DPRConv_6: top-k + persona
with open('DPRConv_6.json', 'w', encoding='utf-8', errors='ignore') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
