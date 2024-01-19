import json
import pickle

import tqdm
import numpy as np
import multiprocessing as mp
import nltk
import random
from collections import Counter
import argparse
from src.utils.constants import WORD_PAIRS as word_pairs
from src.utils.comet import Comet
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

random.seed(13)
relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(x):
    if x == "True":
        return True
    elif x == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("must be True or False")


parser = argparse.ArgumentParser()
parser.add_argument('--add_persona', type=str2bool, required=True,
                    help="True or False, this determine whether to use ESConv with persona")
args = parser.parse_args()


def _norm(x):
    return ' '.join(x.strip().split())


def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


strategies = json.load(open('./_reformat/strategy.json'))
strategies = [e[1:-1] for e in strategies]
strat2id = {strat: i for i, strat in enumerate(strategies)}
print(args.add_persona)
# if args.add_persona:
#     original = json.load(open('./_reformat/DPRConv_5.json'))
# else:
#     original = json.load(open('./_reformat/ESConv.json'))


def get_commonsense(comet, item):
    cs_list = []
    input_event = " ".join(item)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list.append(cs_res)

    return cs_list


# def comet_data(d):
#     emotion = d['emotion_type']
#     problem = d["problem_type"]
#     situation = d['situation']
#     persona = d['persona']
#     persona_list = d['persona_list']
#
#     d = d['dialog']
#     dial = []
#     pre_text = ""
#
#     for uttr in d:
#         text = _norm(uttr['content'])
#         role = uttr['speaker']
#
#         if role == 'seeker':
#             dial.append({
#                 'text': text,
#                 'speaker': 'usr',
#             })
#         else:
#             if len(uttr) > 3:
#                 post = _norm(pre_text)
#                 item = process_sent(post)
#                 cs_list = get_commonsense(comet, item)
#
#                 dial.append({
#                     'text': text,
#                     'speaker': 'sys',
#                     'strategy': uttr['annotation']['strategy'],
#                     'dpr': uttr['dpr'],
#                     'comet': cs_list
#                 })
#             else:
#                 dial.append({
#                     'text': text,
#                     'speaker': 'sys',
#                     'strategy': uttr['annotation']['strategy'],
#                 })
#         pre_text = uttr['content']
#
#     res = {
#         'emotion_type': emotion,
#         'problem_type': problem,
#         'persona': persona,
#         'persona_list': persona_list,
#         'situation': situation,
#         # 'init_intensity': init_intensity,
#         # 'final_intensity': final_intensity,
#         'dialog': dial,
#     }
#
#     return res


def comet_data(d):
    d = eval(d)
    emotion = d['emotion_type']
    problem = d["problem_type"]
    situation = d['situation']
    persona = d['persona']
    persona_list = d['persona_list']

    d = d['dialog']
    dial = []
    pre_text = ""
    context = ""
    user_number = 0

    for uttr in d:
        text = _norm(uttr['text'])
        role = uttr['speaker']

        if role == 'usr':
            dial.append({
                'text': text,
                'speaker': 'usr',
            })
            user_number += 1
        else:
            if len(uttr) > 3:
                persona = persona_list[user_number - 3]

                post = _norm(pre_text + persona)
                # post = _norm(pre_text)
                item = process_sent(post)
                # item = process_sent(context + persona)
                cs_list = get_commonsense(comet, item)

                dial.append({
                    'text': text,
                    'speaker': 'sys',
                    'strategy': uttr['strategy'],
                    'dpr': uttr['dpr'],
                    'comet': cs_list
                })
            else:
                dial.append({
                    'text': text,
                    'speaker': 'sys',
                    'strategy': uttr['strategy'],
                })
        pre_text = uttr['text']
        context += pre_text

    res = {
        'emotion_type': emotion,
        'problem_type': problem,
        'persona': persona,
        'persona_list': persona_list,
        'situation': situation,
        # 'init_intensity': init_intensity,
        # 'final_intensity': final_intensity,
        'dialog': dial,
    }

    return res


"""
1: DPRConv_1
2: DPRConv_2
3: DPRConv_2 + persona (🐶)
4: DPRConv_1 + persona
5: DPRConv_5 + persona
"""
comet = Comet("src/utils/data/ED/Comet", device)
# comet = Comet("src/utils/data/ED/Comet_GPT2", device)
data = []
datasets = ['train.txt', 'test.txt', 'valid.txt']
for dataset in datasets:
    with open('./_reformat/' + dataset) as f:
        original = f.readlines()
    data = []
    for e in map(comet_data, tqdm.tqdm(original, total=len(original))):
        data.append(e)
    with open('./DATA/6_' + dataset, 'w') as f:
        for e in data:
            f.write(json.dumps(e) + '\n')
# with mp.Pool(processes=mp.cpu_count()) as pool:
#     for e in pool.imap(comet_data, tqdm.tqdm(original, total=len(original))):
#         data.append(e)
# for e in map(comet_data, tqdm.tqdm(original, total=len(original))):
#     data.append(e)

emotions = Counter([e['emotion_type'] for e in data])
problems = Counter([e['problem_type'] for e in data])
print('emotion', emotions)
print('problem', problems)

# random.shuffle(data)
# dev_size = int(0.1 * len(data))
# test_size = int(0.1 * len(data))
# valid = data[:dev_size] + data[dev_size + test_size: dev_size + dev_size + test_size]
# test = data[dev_size: dev_size + test_size]
# train = data[dev_size + dev_size + test_size:]
#
# print('train', len(train))
# with open('./train.txt', 'w') as f:
#     for e in train:
#         f.write(json.dumps(e) + '\n')
# with open('./sample.json', 'w') as f:
#     json.dump(train[:10], f, ensure_ascii=False, indent=2)
#
# print('valid', len(valid))
# with open('./valid.txt', 'w') as f:
#     for e in valid:
#         f.write(json.dumps(e) + '\n')
#
# print('test', len(test))
# with open('./test.txt', 'w') as f:
#     for e in test:
#         f.write(json.dumps(e) + '\n')
