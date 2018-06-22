# -*- coding: utf-8 -*-

import pickle

import numpy as np


def preprocess(fdata):
    start = 0
    sentences = []
    with open(fdata, 'r') as train:
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
        self.wdict = {w: i for i, w in enumerate(words)}
        # 词性对应索引的字典
        self.tdict = {t: i for i, t in enumerate(tags)}

        self.m = len(self.words)
        self.n = len(self.tags)

    def train(self, sentences, alpha=0.01, file=None):
        trans_matrix = np.zeros((self.n + 1, self.n + 1))
        emit_matrix = np.zeros((self.m + 1, self.n))

        for sentence in sentences:
            prev = -1
            for word, tag in sentence:
                trans_matrix[self.tdict[tag], prev] += 1
                emit_matrix[self.wdict[word], self.tdict[tag]] += 1
                prev = self.tdict[tag]
            trans_matrix[self.n, prev] += 1
        trans_matrix = self.smooth(trans_matrix, alpha)

        # 迁移概率
        self.A = np.log(trans_matrix[:-1, :-1])
        # 句首迁移概率
        self.BOS = np.log(trans_matrix[:-1, -1])
        # 句尾迁移概率
        self.EOS = np.log(trans_matrix[-1, :-1])
        # 发射概率
        self.B = np.log(self.smooth(emit_matrix, alpha))

        # 保存训练好的模型
        if file is not None:
            self.dump(file)

    def smooth(self, matrix, alpha):
        sums = np.sum(matrix, axis=0)
        return (matrix + alpha) / (sums + alpha * len(matrix))

    def predict(self, wordseq):
        T = len(wordseq)
        delta = np.zeros((T, self.n))
        paths = np.zeros((T, self.n), dtype='int')
        indices = [self.wdict[w] if w in self.wdict else -1 for w in wordseq]

        delta[0] = self.BOS + self.B[indices[0]]

        for i in range(1, T):
            for j in range(self.n):
                probs = delta[i - 1] + self.A[j]
                paths[i, j] = np.argmax(probs)
                delta[i, j] = probs[paths[i, j]] + self.B[indices[i], j]
        prev = np.argmax(delta[-1] + self.EOS)

        predict = [prev]
        for i in range(T - 1, 0, -1):
            prev = paths[i, prev]
            predict.append(prev)
        preseq = [self.tags[i] for i in reversed(predict)]
        return preseq

    def evaluate(self, sentences):
        tp, total = 0, 0

        for sentence in sentences:
            total += len(sentence)
            wordseq, tagseq = zip(*sentence)
            preseq = self.predict(wordseq)
            tp += sum([t == p for t, p in zip(tagseq, preseq)])
        precision = tp / total
        return tp, total, precision

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            hmm = pickle.load(f)
        return hmm
