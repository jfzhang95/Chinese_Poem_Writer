from utils import process_poems, generate_batch
import argparse
import torch.nn as nn
import torch.optim as optim
import sys
from char_cnn.utils import *
from char_cnn.model import TCN
import time
import math
import numpy as np

import warnings

warnings.filterwarnings("ignore")  # Suppress the RunTimeWarning on unicode

start_token = 'B'
end_token = 'E'
corpus_file = './data/poems.txt'


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    if sample < 0:
        sample = 0
    return vocabs[sample]


def write_poem(begin_word=None):
    model = torch.load('model/model_100.pth', map_location=lambda storage, loc: storage)
    model.eval()


    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)

    x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, start_token))])).long())

    y = model(x)

    if begin_word:
        word = begin_word
    else:
        word = to_word(y.detach().numpy(), vocabularies)

    poem = ''
    before_word = None
    flag = False

    ignore_list = ['B', 'E', ' ', '。', '，', '  ']

    i = 0

    while word:
        if word in ignore_list:
            x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, start_token))])).long())

            y = model(x)
            word = to_word(y.detach().numpy(), vocabularies)
            continue

        elif word == before_word and flag == True:
            flag = False
            continue

        else:
            poem += word
            before_word = word
            flag = True
            i += 1

            if i == 7 or i == 21:
                poem += ' '
            if i == 14:
                poem += '\n'
            if i >= 28:
                break
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]

            x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, start_token))])).long())

            y = model(x)
            word = to_word(y.detach().numpy(), vocabularies)

    print(poem)


if __name__ == "__main__":
    write_poem()






