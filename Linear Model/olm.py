# -*- coding: utf-8 -*-

import sys
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


class LinearModel(object):

    def __init__(self, tags):
        # 所有不同的词性
        self.tags = tags
        # 词性对应索引的字典
        self.tagdict = {t: i for i, t in enumerate(tags)}

        self.N = len(self.tags)

    def create_feature_space(self, sentences):
        feature_space = set()
        for sentence in sentences:
            wordseq, tagseq = zip(*sentence)
            for i, tag in enumerate(tagseq):
                fvector = self.instantialize(wordseq, i)
                feature_space.update(fvector)

        # 特征空间
        self.epsilon = list(feature_space)
        # 特征对应索引的字典
        self.feadict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.D = len(self.epsilon)

        # 特征权重
        self.W = np.zeros((self.D, self.N), dtype='int')
        # 累加特征权重
        self.V = np.zeros((self.D, self.N), dtype='int')

    def online(self, sentences, epochs=20):
        # 迭代指定次数训练模型
        for epoch in range(epochs):
            for sentence in sentences:
                wordseq, tagseq = zip(*sentence)
                # 根据单词序列的正确词性更新权重
                for i, tag in enumerate(tagseq):
                    self.update(wordseq, i, tag)
            yield epoch

    def update(self, wordseq, index, tag):
        # 根据现有权重向量预测词性
        pre = self.predict(wordseq, index)
        # 如果预测词性与正确词性不同，则更新权重
        if tag != pre:
            for feature in self.instantialize(wordseq, index):
                if feature in self.feadict:
                    f_index = self.feadict[feature]
                    t_index, p_index = self.tagdict[tag], self.tagdict[pre]
                    self.W[f_index][t_index] += 1
                    self.W[f_index][p_index] -= 1
            self.V += self.W

    def predict(self, wordseq, index, average=False):
        fvector = self.instantialize(wordseq, index)
        scores = self.score(fvector, average=average)
        return self.tags[np.argmax(scores)]

    def score(self, fvector, average=False):
        # 计算特征对应累加权重的得分
        if average:
            scores = [self.V[self.feadict[f]]
                      for f in fvector if f in self.feadict]
        # 计算特征对应未累加权重的得分
        else:
            scores = [self.W[self.feadict[f]]
                      for f in fvector if f in self.feadict]
        return np.sum(scores, axis=0)

    def instantialize(self, wordseq, index):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else "^^"
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else "$$"
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

    def evaluate(self, sentences, average=False):
        tp, total = 0, 0

        for sentence in sentences:
            total += len(sentence)
            wordseq, tagseq = zip(*sentence)
            preseq = [self.predict(wordseq, i, average)
                      for i in range(len(wordseq))]
            tp += sum([t == p for t, p in zip(tagseq, preseq)])
        precision = tp / total
        return tp, total, precision


if __name__ == '__main__':
    train = preprocessing('data/train.conll')
    dev = preprocessing('data/dev.conll')

    all_words, all_tags = zip(*np.vstack(train))
    tags = sorted(set(all_tags))

    start = time.time()

    print("Creating Linear Model with %d tags" % (len(tags)))
    lm = LinearModel(tags)

    print("Using %d sentences to create the feature space" % (len(train)))
    lm.create_feature_space(train)
    print("The size of the feature space is %d" % lm.D)

    average = len(sys.argv) > 1 and sys.argv[1] == 'average'
    evaluations = []

    print("Using online-training algorithm to train the model")
    for epoch in lm.online(train):
        print("Epoch %d" % epoch)
        result = lm.evaluate(train, average=average)
        print("\ttrain: %d / %d = %4f" % result)
        result = lm.evaluate(dev, average=average)
        print("\tdev: %d / %d = %4f" % result)
        evaluations.append(result)

    print("Successfully evaluated dev data using the model")
    print("Precision: %d / %d = %4f" % max(evaluations, key=lambda x: x[2]))
    print("%4fs elapsed" % (time.time() - start))
