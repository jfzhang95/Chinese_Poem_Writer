import torch
from torch.autograd import Variable
from utils import process_poems, to_word
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # Suppress the RunTimeWarning on unicode

parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
parser.add_argument("-m", "--mode", choices=["context", "fast", "head"], default="fast",
                    help="select mode by 'context', 'fast' or head")

start_token = 'B'
end_token = 'E'
corpus_file = './data/poems.txt'


def write_poem_context(rows=4, cols=None, begin_word=None, context_info=None):
    """This mode will generate poem by considering whole previous context"""
    model = torch.load('model/model_100.pth', map_location=lambda storage, loc: storage)
    model.eval()

    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)

    if begin_word is not None:
        word = begin_word
    else:
        x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, start_token))])).long())
        y = model(x)
        word = to_word(y.detach().numpy(), vocabularies)

    poem = ''
    if context_info is not None:
        context = context_info
    else:
        context = 'B'

    before_word = None
    flag = True

    ignore_list = ['B', 'E', ' ', '。', '，', '\n']

    i = 0

    if cols is None:
        if np.random.random() <= 0.5:
            cols = 5
        else:
            cols = 7

    while word:
        if word in ignore_list:
            x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, context))])).long())
            y = model(x)
            word = to_word(y.detach().numpy(), vocabularies)
            continue

        elif word == before_word and flag == True:
            x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, context))])).long())
            y = model(x)
            word = to_word(y.detach().numpy(), vocabularies)
            flag = False
            continue

        else:
            poem += word
            context += word
            before_word = word
            flag = True
            i += 1

            if i % cols == 0 and i < rows*cols:
                poem += '\n'
            if i >= rows*cols:
                break

            x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, context))])).long())
            y = model(x)
            word = to_word(y.detach().numpy(), vocabularies)
    return poem

def write_poem_fast(rows=4, cols=None, begin_word=None):
    """This mode just generates each word without context information"""
    model = torch.load('model/model_100.pth', map_location=lambda storage, loc: storage)
    model.eval()

    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)

    if begin_word is not None:
        word = begin_word
    else:
        x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, start_token))])).long())
        y = model(x)
        word = to_word(y.detach().numpy(), vocabularies)

    poem = ''

    before_word = None
    flag = True

    ignore_list = ['B', 'E', ' ', '。', '，', '\n']

    i = 0

    if cols is None:
        if np.random.random() <= 0.5:
            cols = 5
        else:
            cols = 7

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

            if i % cols == 0 and i < rows*cols:
                poem += '\n'
            if i >= rows*cols:
                break

            x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, word))])).long())
            y = model(x)
            word = to_word(y.detach().numpy(), vocabularies)

    return poem


def write_poem_head(begin_words=None):
    rows = len(begin_words)
    model = torch.load('model/model_100.pth', map_location=lambda storage, loc: storage)
    model.eval()

    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)
    poem = ''
    context = ''

    ignore_list = ['B', 'E', ' ', '。', '，', '\n']

    if np.random.random() <= 0.5:
        cols = 5
    else:
        cols = 7

    for begin_word in begin_words:
        poem += begin_word
        context += begin_word
        before_word = begin_word
        flag = True
        i = 1

        x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, begin_word))])).long())
        y = model(x)
        word = to_word(y.detach().numpy(), vocabularies)

        while i < cols:
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
                context += word
                before_word = word
                flag = True
                i += 1

                x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, word))])).long())
                y = model(x)
                word = to_word(y.detach().numpy(), vocabularies)
        if begin_word != begin_words[-1]:
            poem += '\n'
    if rows < 4:
        poem += '\n'
        remaining = 4 - rows
        poem_ = write_poem_context(remaining, cols, word, context)
        poem += poem_

    return poem


if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == 'context':
        rows = int(input("Please input poem's rows: "))
        print('Generating...')
        poem = write_poem_context(rows)
        print(poem)
    elif args.mode == 'head':
        begin_words = input("Please input Chinese characters: ")
        print('Generating...')
        poem = write_poem_head(begin_words)
        print(poem)
    elif args.mode == 'fast':
        rows = int(input("Please input poem's rows: "))
        print('Generating...')
        poem = write_poem_fast(rows)
        print(poem)




