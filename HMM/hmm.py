# -*- coding: utf-8 -*-

import time

import numpy as np


def preprocessing(ftrain):
    start = 0
    sentences = []
    with open(ftrain, 'r') as train:
        lines = [line for line in train]
    for i, line in enumerate(lines):
        if len(lines[i]) <= 1:
            sentences.append([l.split()[1:4:2] for l in lines[start:i]])
            start = i + 1
            while start < len(lines) and len(lines[start]) <= 1:
                start += 1
    return sentences


class HMM(object):

    def __init__(self, words, tags):
        # 所有不同的单词
        self.words = words
        # 所有不同的词性
        self.tags = tags
        # 单词对应索引的字典
        self.wordict = {w: i for i, w in enumerate(words)}
        # 词性对应索引的字典
        self.tagdict = {t: i for i, t in enumerate(tags)}

        self.M = len(self.words)
        self.N = len(self.tags)

    def train(self, sentences):
        trans_matrix = np.zeros((self.N + 1, self.N + 1))
        emit_matrix = np.zeros((self.M + 1, self.N))

        for sentence in sentences:
            prev = -1
            for word, tag in sentence:
                trans_matrix[self.tagdict[tag]][prev] += 1
                emit_matrix[self.wordict[word]][self.tagdict[tag]] += 1
                prev = self.tagdict[tag]
            trans_matrix[self.N][prev] += 1
        trans_matrix = self.smooth(trans_matrix)

        # 迁移概率
        self.A = np.log(trans_matrix[:-1, :-1])
        # 句首迁移概率
        self.BOS = np.log(trans_matrix[:-1, -1])
        # 句尾迁移概率
        self.EOS = np.log(trans_matrix[-1, :-1])
        # 发射概率
        self.B = np.log(self.smooth(emit_matrix))

    def viterbi(self, wordseq):
        T = len(wordseq)
        delta = np.zeros((T, self.N))
        paths = np.zeros((T, self.N), dtype='int')
        indices = [self.wordict[w]
                   if w in self.wordict else -1
                   for w in wordseq]

        delta[0] = self.BOS + self.B[indices[0]]

        for i in range(1, T):
            for j in range(self.N):
                probs = delta[i - 1] + self.A[j]
                paths[i][j] = np.argmax(probs)
                delta[i][j] = probs[paths[i][j]] + self.B[indices[i]][j]
        prev = np.argmax(delta[-1] + self.EOS)

        predict = [prev]
        for i in range(T - 1, 0, -1):
            prev = paths[i, prev]
            predict.append(prev)
        return [self.tags[i] for i in reversed(predict)]

    def smooth(self, matrix, alpha=0.3):
        sums = np.sum(matrix, axis=0)
        return (matrix + alpha) / (sums + alpha * len(matrix))

    def evaluate(self, sentences):
        tp, total = 0, 0

        for sentence in sentences:
            total += len(sentence)
            wordseq, tagseq = zip(*sentence)
            preseq = self.viterbi(wordseq)
            tp += sum([t == p for t, p in zip(tagseq, preseq)])
        precision = tp / total
        return tp, total, precision


if __name__ == '__main__':
    train = preprocessing('data/train.conll')
    dev = preprocessing('data/dev.conll')

    all_words, all_tags = zip(*np.vstack(train))
    words, tags = list(set(all_words)), list(set(all_tags))

    start = time.time()

    print("Creating HMM with %d words and %d tags" % (len(words), len(tags)))
    hmm = HMM(words, tags)

    print("Using %d sentences to train the HMM" % (len(train)))
    hmm.train(train)

    print("Using Viterbi algorithm to tag the dev data")
    tp, total, precision = hmm.evaluate(dev)

    print("Successfully evaluated dev data using the model")
    print("Precision: %d / %d = %4f" % (tp, total, precision))

    print("%4fs elapsed" % (time.time() - start))
