import torch
from torch.autograd import Variable
from utils import process_poems, to_word
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # Suppress the RunTimeWarning on unicode

parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
parser.add_argument("-m", "--mode", choices=["random", "head"], default="random",
                    help="select mode by 'random' or head")

start_token = 'B'
end_token = 'E'
corpus_file = './data/poems.txt'


def write_poem_random(k=4):
    model = torch.load('model/model_100.pth', map_location=lambda storage, loc: storage)
    model.eval()

    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)

    x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, start_token))])).long())
    y = model(x)
    word = to_word(y.detach().numpy(), vocabularies)

    poem = ''

    before_word = None
    flag = True

    ignore_list = ['B', 'E', ' ', '。', '，', '\n']

    i = 0
    if np.random.random() <= 0.5:
        col = 5
    else:
        col = 7

    while word:
        if word in ignore_list:
            x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, word))])).long())
            y = model(x)
            word = to_word(y.detach().numpy(), vocabularies)
            continue

        elif word == before_word and flag == True:
            x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, word))])).long())
            y = model(x)
            word = to_word(y.detach().numpy(), vocabularies)
            flag = False
            continue

        else:
            poem += word
            before_word = word
            flag = True
            i += 1

            if i % col == 0 and i < k*col:
                poem += '\n'
            if i >= k*col:
                break

            x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, word))])).long())
            y = model(x)
            word = to_word(y.detach().numpy(), vocabularies)

    print(poem)


def write_poem_head(begin_words=None):
    model = torch.load('model/model_100.pth', map_location=lambda storage, loc: storage)
    model.eval()

    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)
    poem = ''

    ignore_list = ['B', 'E', ' ', '。', '，', '\n']

    if np.random.random() <= 0.5:
        col = 5
    else:
        col = 7

    for begin_word in begin_words:
        poem += begin_word
        before_word = begin_word
        flag = True
        i = 1

        x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, begin_word))])).long())
        y = model(x)
        word = to_word(y.detach().numpy(), vocabularies)

        while i < col:
            if word in ignore_list:
                x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, word))])).long())
                y = model(x)
                word = to_word(y.detach().numpy(), vocabularies)
                continue

            elif word == before_word and flag == True:
                x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, word))])).long())
                y = model(x)
                word = to_word(y.detach().numpy(), vocabularies)
                flag = False
                continue

            else:
                poem += word
                before_word = word
                flag = True
                i += 1

                x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, word))])).long())
                y = model(x)
                word = to_word(y.detach().numpy(), vocabularies)
        if begin_word != begin_words[-1]:
            poem += '\n'

    print(poem)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == 'random':
        k = int(input("Please input poem's row: "))
        print('Generating...')
        write_poem_random(k)
    elif args.mode == 'head':
        begin_words = input("Please input Chinese character: ")
        print('Generating...')
        write_poem_head(begin_words)

