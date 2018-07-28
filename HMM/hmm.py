# -*- coding: utf-8 -*-

import pickle

import numpy as np


class HMM(object):

    def __init__(self, nw, nt):
        # 词汇数量
        self.nw = nw
        # 词性数量
        self.nt = nt

    def train(self, train, file, alpha=0.01):
        trans = np.zeros((self.nt + 1, self.nt + 1))
        emit = np.zeros((self.nw, self.nt))

        for wiseq, tiseq in train:
            prev = -1
            for wi, ti in zip(wiseq, tiseq):
                trans[ti, prev] += 1
                emit[wi, ti] += 1
                prev = ti
            trans[self.nt, prev] += 1
        trans = self.smooth(trans, alpha)

        # 迁移概率
        self.A = np.log(trans[:-1, :-1])
        # 句首迁移概率
        self.SOS = np.log(trans[:-1, -1])
        # 句尾迁移概率
        self.EOS = np.log(trans[-1, :-1])
        # 发射概率
        self.B = np.log(self.smooth(emit, alpha))

        # 保存训练好的模型
        if file is not None:
            self.dump(file)

    def smooth(self, matrix, alpha):
        sums = np.sum(matrix, axis=0)
        return (matrix + alpha) / (sums + alpha * len(matrix))

    def predict(self, wiseq):
        T = len(wiseq)
        delta = np.zeros((T, self.nt))
        paths = np.zeros((T, self.nt), dtype='int')

        delta[0] = self.SOS + self.B[wiseq[0]]

        for i in range(1, T):
            probs = self.A + delta[i - 1]
            paths[i] = np.argmax(probs, axis=1)
            delta[i] = probs[np.arange(self.nt), paths[i]] + self.B[wiseq[i]]
        prev = np.argmax(delta[-1] + self.EOS)

        predict = [prev]
        for i in reversed(range(1, T)):
            prev = paths[i, prev]
            predict.append(prev)
        predict.reverse()
        return predict

    def evaluate(self, data):
        tp, total = 0, 0

        for wiseq, tiseq in data:
            total += len(wiseq)
            predict = np.array(self.predict(wiseq))
            tp += np.sum(tiseq == predict)
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
