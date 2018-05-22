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
warnings.filterwarnings("ignore")   # Suppress the RunTimeWarning on unicode


parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.1)')
parser.add_argument('--emb_dropout', type=float, default=0.1,
                    help='dropout applied to the embedded layer (0 = no dropout) (default: 0.1)')
parser.add_argument('--clip', type=float, default=0.15,
                    help='gradient clip, -1 means no clip (default: 0.15)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=2,
                    help='# of levels (default: 3)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='initial learning rate (default: 4)')
parser.add_argument('--emsize', type=int, default=20,
                    help='dimension of character embeddings (default: 100)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer (default: 450)')
parser.add_argument('--validseqlen', type=int, default=81,
                    help='valid sequence length (default: 320)')
parser.add_argument('--seq_len', type=int, default=90,
                    help='total sequence length, including effective history (default: 400)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='dataset to use (default: ptb)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

file_path = 'data/poems.txt'

poems_vector, word_to_int, vocabularies = process_poems(file_path)
batches_inputs, batches_outputs = generate_batch(args.batch_size, poems_vector, word_to_int, args.seq_len)

batches_inputs = Variable(torch.from_numpy(np.array(batches_inputs)).long()).cuda()
batches_outputs = Variable(torch.from_numpy(np.array(batches_outputs)).long()).cuda()


n_characters = len(vocabularies)
num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
k_size = args.ksize
dropout = args.dropout
emb_dropout = args.emb_dropout
model = TCN(args.emsize, n_characters, num_chans, kernel_size=k_size, dropout=dropout, emb_dropout=emb_dropout)

if args.cuda:
    model.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    losses = []
    n = 0
    n_chunk = len(poems_vector) // args.batch_size
    for batch_idx in range(n_chunk):
        inp, target = batches_inputs[n], batches_outputs[n]
        # inp, target = get_batch(source, i, args)
        optimizer.zero_grad()
        output = model(inp)
        eff_history = args.seq_len - args.validseqlen
        final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
        final_target = target[:, eff_history:].contiguous().view(-1)
        # final_output = output.contiguous().view(-1, n_characters)
        # final_target = target.contiguous().view(-1)
        loss = criterion(final_output, final_target)
        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.data

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss[0] / args.log_interval
            losses.append(cur_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                  'loss {:5.3f} | bpc {:5.3f}'.format(
                epoch, batch_idx, int(n_chunk), lr,
                              elapsed * 1000 / args.log_interval, cur_loss, cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()


        # if batch % (200 * args.log_interval) == 0 and batch > 0:
        #     vloss = evaluate(val_data)
        #     print('-' * 89)
        #     print('| In epoch {:3d} | valid loss {:5.3f} | '
        #           'valid bpc {:8.3f}'.format(epoch, vloss, vloss / math.log(2)))
        #     model.train()

    if epoch != 0 and epoch % 25 == 0:
        save_filename = 'model/model_{}_20.pth'.format(str(epoch))
        torch.save(model, save_filename)
        print('Saved as %s' % save_filename)

    return sum(losses) * 1.0 / len(losses)


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]

def pretty_print_poem(poem_):
    poem_sentences = poem_.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')

if __name__ == "__main__":

    for epoch in range(1, args.epochs + 1):
        train(epoch)


    # begin_word = None
    # start_token = 'B'
    # end_token = 'E'
    # corpus_file = './data/poems.txt'
    #
    # model = torch.load('model/model100.pth')
    # model.eval()
    #
    # poems_vector, word_int_map, vocabularies = process_poems(corpus_file)
    #
    # batches_inputs = Variable(torch.from_numpy(np.array(batches_inputs)).long()).cuda()
    #
    # x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, start_token))])).long()).cuda()
    #
    # y = model(x)
    #
    # if begin_word:
    #     word = begin_word
    # else:
    #     word = to_word(y.detach().cpu().numpy(), vocabularies)
    #
    # poem_ = ''
    #
    # i = 0
    # while word != end_token:
    #     poem_ += word
    #     i += 1
    #     if i >= 24:
    #         break
    #     x = np.zeros((1, 1))
    #     x[0, 0] = word_int_map[word]
    #
    #     x = Variable(torch.from_numpy(np.array([list(map(word_int_map.get, start_token))])).long()).cuda()
    #
    #     y = model(x)
    #     word = to_word(y.detach().cpu().numpy(), vocabularies)
    #
    #
    # pretty_print_poem(poem_=poem_)
#




