import json
import pickle

import tqdm
import numpy as np
import multiprocessing as mp
import nltk
import random
from collections import Counter
import argparse

random.seed(13)


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


strategies = json.load(open('./strategy.json'))
strategies = [e[1:-1] for e in strategies]
strat2id = {strat: i for i, strat in enumerate(strategies)}
print(args.add_persona)
if args.add_persona:
    # original = json.load(open('./PESConv.json'))
    original = json.load(open('./DPRConv_6.json'))
else:
    original = json.load(open('./ESConv.json'))


def process_data(d):
    emotion = d['emotion_type']
    problem = d["problem_type"]
    situation = d['situation']
    persona = d['persona']
    persona_list = d['persona_list']
    # persona_list = []
    # init_intensity = int(d['score']['speaker']['begin_intensity'])
    # final_intensity = int(d['score']['speaker']['end_intensity'])

    d = d['dialog']
    dial = []
    for uttr in d:
        text = _norm(uttr['content'])
        role = uttr['speaker']
        if role == 'seeker':
            dial.append({
                'text': text,
                'speaker': 'usr',
            })
            # persona_list.append(text)
        else:
            if len(uttr) > 3:
                dial.append({
                    'text': text,
                    'speaker': 'sys',
                    'strategy': uttr['annotation']['strategy'],
                    'dpr': uttr['dpr']
                })
            else:
                dial.append({
                    'text': text,
                    'speaker': 'sys',
                    'strategy': uttr['annotation']['strategy']
                })
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


data = []
with mp.Pool(processes=mp.cpu_count()) as pool:
    for e in pool.imap(process_data, tqdm.tqdm(original, total=len(original))):
        data.append(e)

emotions = Counter([e['emotion_type'] for e in data])
problems = Counter([e['problem_type'] for e in data])
print('emotion', emotions)
print('problem', problems)

random.shuffle(data)
dev_size = int(0.1 * len(data))
test_size = int(0.1 * len(data))
valid = data[:dev_size] + data[dev_size + test_size: dev_size + dev_size + test_size]
test = data[dev_size: dev_size + test_size]
train = data[dev_size + dev_size + test_size:]

turns = 0
uttrs = 0
print('train', len(train))
with open('train.txt', 'w') as f:
    for e in train:
        turns += len(e['dialog'])
        for uttr in e['dialog']:
            uttrs += len(uttr['text'].split())
        # f.write(json.dumps(e) + '\n')
# print('Avg. length of dialogues (train): ', turns / len(train))
# print('Avg. length of utterances (train): ', uttrs / turns)
# with open('./sample.json', 'w') as f:
#     json.dump(train[:10], f, ensure_ascii=False, indent=2)

# turns = 0
# uttrs = 0
print('valid', len(valid))
with open('valid.txt', 'w') as f:
    for e in valid:
        turns += len(e['dialog'])
        for uttr in e['dialog']:
            uttrs += len(uttr['text'].split())
#         f.write(json.dumps(e) + '\n')
# print('Avg. length of dialogues (valid): ', turns / len(valid))
# print('Avg. length of utterances (valid): ', uttrs / turns)

# turns = 0
# uttrs = 0
print('test', len(test))
with open('test.txt', 'w') as f:
    for e in test:
        turns += len(e['dialog'])
        for uttr in e['dialog']:
            uttrs += len(uttr['text'].split())
#         f.write(json.dumps(e) + '\n')
# print('Avg. length of dialogues (test): ', turns / len(test))
# print('Avg. length of utterances (test): ', uttrs / turns)

print('Avg. length of dialogues (test): ', turns / (len(train) + len(valid) + len(test)))
print('Avg. length of utterances (test): ', uttrs / turns)


# def save_comet(texts, dev_size, test_size, type, dataset):
#     if type == "context":
#         name = "sys_dialog_texts"
#     elif type == "target":
#         name = "sys_target_texts"
#     elif type == "emotion":
#         name = "sys_emotion_texts"
#     elif type == "situation":
#         name = "sys_situation_texts"
#     else:
#         name = ""
#
#     if dataset == "train":
#         data_dir = f"./comet_data/{name}.train.npy"
#         data = texts[dev_size + dev_size + test_size:]
#     elif dataset == "test":
#         data_dir = f"./comet_data/{name}.test.npy"
#         data = texts[dev_size: dev_size + test_size]
#     elif dataset == "valid":
#         data_dir = f"./comet_data/{name}.dev.npy"
#         data = texts[:dev_size] + texts[dev_size + test_size: dev_size + dev_size + test_size]
#     else:
#         data_dir = f"./comet_data/{name}.npy"
#         data = []
#
#     with open(data_dir, 'wb') as file:
#         pickle.dump(data, file)
#
#
# assert (
#     len(context)
#     == len(target)
#     == len(emotion)
#     == len(situation)
# )
# # random.shuffle(context)
# # random.shuffle(target)
# # random.shuffle(emotion)
# # random.shuffle(situation)
# dev_size = int(0.1 * len(context))
# test_size = int(0.1 * len(context))
# save_comet(context, dev_size, test_size, "context", "train")
# save_comet(context, dev_size, test_size, "context", "test")
# save_comet(context, dev_size, test_size, "context", "valid")
# save_comet(target, dev_size, test_size, "target", "train")
# save_comet(target, dev_size, test_size, "target", "test")
# save_comet(target, dev_size, test_size, "target", "valid")
# save_comet(emotion, dev_size, test_size, "emotion", "train")
# save_comet(emotion, dev_size, test_size, "emotion", "test")
# save_comet(emotion, dev_size, test_size, "emotion", "valid")
# save_comet(situation, dev_size, test_size, "situation", "train")
# save_comet(situation, dev_size, test_size, "situation", "test")
# save_comet(situation, dev_size, test_size, "situation", "valid")
