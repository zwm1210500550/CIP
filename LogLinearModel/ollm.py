# -*- coding: utf-8 -*-

import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.misc import logsumexp


class LogLinearModel(object):

    def __init__(self, nt):
        # 词性数量
        self.nt = nt

    def create_feature_space(self, data):
        # 特征空间
        self.epsilon = list({
            f for wordseq, tiseq in data
            for i, ti in enumerate(tiseq)
            for f in self.instantiate(wordseq, i)
        })
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros((self.d, self.nt))

    def SGD(self, train, dev, file,
            epochs, batch_size, interval, eta, decay, lmbda,
            anneal, regularize, shuffle):
        # 记录更新次数
        count = 0
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_precision = 0, 0.0

        train_data = [(wordseq, i, ti)
                      for wordseq, tiseq in train
                      for i, ti in enumerate(tiseq)]
        n = len(train_data)
        # 迭代指定次数训练模型
        for epoch in range(epochs):
            start = datetime.now()
            # 随机打乱数据
            if shuffle:
                random.shuffle(train_data)
            # 设置L2正则化系数
            if not regularize:
                lmbda = 0
            # 按照指定大小对数据分割批次
            batches = [train_data[i:i + batch_size]
                       for i in range(0, n, batch_size)]
            nb = len(batches)
            # 根据批次数据更新权重
            for batch in batches:
                if not anneal:
                    self.update(batch, lmbda, n, eta)
                # 设置学习速率的指数衰减
                else:
                    self.update(batch, lmbda, n, eta * decay ** (count / nb))
                count += 1

            print("Epoch %d / %d: " % (epoch, epochs))
            print("\ttrain: %d / %d = %4f" % self.evaluate(train))
            tp, total, precision = self.evaluate(dev)
            print("\tdev: %d / %d = %4f" % (tp, total, precision))
            t = datetime.now() - start
            print("\t%ss elapsed" % t)
            total_time += t

            # 保存效果最好的模型
            if precision > max_precision:
                self.dump(file)
                max_e, max_precision = epoch, precision
            elif epoch - max_e > interval:
                break
        print("max precision of dev is %4f at epoch %d" %
              (max_precision, max_e))
        print("mean time of each epoch is %ss" % (total_time / (epoch + 1)))

    def update(self, batch, lmbda, n, eta):
        gradients = defaultdict(float)

        for wordseq, i, ti in batch:
            fv = self.instantiate(wordseq, i)
            fiseq = (self.fdict[f] for f in fv if f in self.fdict)
            scores = self.score(fv)
            probs = np.exp(scores - logsumexp(scores))

            for fi in fiseq:
                gradients[fi, ti] += 1
                gradients[fi] -= probs
        if lmbda != 0:
            self.W *= (1 - eta * lmbda / n)
        for k, v in gradients.items():
            self.W[k] += eta * v

    def predict(self, wordseq, index):
        fv = self.instantiate(wordseq, index)
        scores = self.score(fv)
        return np.argmax(scores)

    def score(self, fvector):
        scores = np.array([self.W[self.fdict[f]]
                           for f in fvector if f in self.fdict])
        return np.sum(scores, axis=0)

    def instantiate(self, wordseq, index):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else '^^'
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else '$$'
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        fvector = []
        fvector.append(('02', word))
        fvector.append(('03', prev_word))
        fvector.append(('04', next_word))
        fvector.append(('05', word, prev_char))
        fvector.append(('06', word, next_char))
        fvector.append(('07', first_char))
        fvector.append(('08', last_char))

        for char in word[1:-1]:
            fvector.append(('09', char))
            fvector.append(('10', first_char, char))
            fvector.append(('11', last_char, char))
        if len(word) == 1:
            fvector.append(('12', word, prev_char, next_char))
        for i in range(1, len(word)):
            prev_char, char = word[i - 1], word[i]
            if prev_char == char:
                fvector.append(('13', char, 'consecutive'))
            if i <= 4:
                fvector.append(('14', word[:i]))
                fvector.append(('15', word[-i:]))
        if len(word) <= 4:
            fvector.append(('14', word))
            fvector.append(('15', word))
        return fvector

    def evaluate(self, data):
        tp, total = 0, 0

        for wordseq, tiseq in data:
            total += len(wordseq)
            piseq = np.array([self.predict(wordseq, i)
                              for i in range(len(wordseq))])
            tp += np.sum(tiseq == piseq)
        precision = tp / total
        return tp, total, precision

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            llm = pickle.load(f)
        return llm
