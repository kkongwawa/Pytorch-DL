import numpy as np
from arg import *


def get_worddict(file):
    datas = open(file, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    word2ind = {"UNK": 0, "PAD": 1}
    label2ind = {}
    word_cnt, label_cnt = 2, 0
    for line in datas:
        line = line.split(' ')
        for w in line[0]:
            if w not in word2ind:
                word2ind[w] = word_cnt
                word_cnt += 1
        if line[1] not in label2ind:
            label2ind[line[1]] = label_cnt
            label_cnt += 1
    ind2word = {word2ind[w]: w for w in word2ind}
    ind2label = {label2ind[a]: a for a in label2ind}
    return word2ind, ind2word, label2ind, ind2label, datas


def get_input(word2ind, label2ind, datas):
    datas = [data.split(" ") for data in datas]
    x, y = [], []
    for data in datas:
        xi = []
        for w in data[0]:
            xi.append(word2ind.get(w, word2ind["UNK"]))
        x.append(xi)
        y.append(label2ind[data[1]])
    x, y = padding(x, y, args.sentence_length, word2ind)
    return x, y


def padding(x, y, length, word2ind):
    x_new = []
    for xi in x:
        if len(xi) > length:
            xi = xi[:length]
        else:
            xi += [word2ind["PAD"]]*(length-len(xi))
        x_new.append(xi)
    x = np.array(x_new)
    y = np.array(y)
    return x, y


if __name__ == "__main__":
    word2ind, ind2word, label2ind, ind2label, datas = get_worddict(args.file)

