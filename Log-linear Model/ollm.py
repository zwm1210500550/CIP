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


class LogLinearModel(object):

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
                features = self.instantialize(wordseq, i)
                feature_space.update(features)

        # 特征空间
        self.epsilon = list(feature_space)
        # 特征对应索引的字典
        self.feadict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.D = len(self.epsilon)

        # 特征权重
        self.W = np.zeros((self.D, self.N))
        # 累加特征权重
        self.V = np.zeros((self.D, self.N))

    def SGD(self, sentences, batch_size=50, iter=40):
        b = 0
        gradients = np.zeros((self.D, self.N), dtype='float128')
        for it in range(iter):
            for sentence in sentences:
                wordseq, tagseq = zip(*sentence)
                # 根据单词序列的正确词性更新权重
                for i, tag in enumerate(tagseq):
                    b += 1
                    tag_index = self.tagdict[tag]
                    features = self.instantialize(wordseq, i)
                    scores = np.array(self.score(features))
                    probs = np.array([1 / sum(np.exp(scores - s))
                                      for s in scores])

                    for f in features:
                        if f in self.feadict:
                            gradients[self.feadict[f]][tag_index] += 1
                            gradients[self.feadict[f]] *= (1 - probs)

                    if b == batch_size:
                        self.W += gradients
                        gradients = np.zeros((self.D, self.N))
                        b = 0
            yield it

    def predict(self, wordseq, index):
        features = self.instantialize(wordseq, index)
        scores = self.score(features)
        return self.tags[np.argmax(scores)]

    def score(self, features):
        scores = [self.W[self.feadict[f]]
                  for f in features if f in self.feadict]
        return np.sum(scores, axis=0)

    def instantialize(self, wordseq, index):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else "$$"
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else "##"
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        features = []
        features.append(('02', word))
        features.append(('03', prev_word))
        features.append(('04', next_word))
        features.append(('05', word, prev_char))
        features.append(('06', word, next_char))
        features.append(('07', first_char))
        features.append(('08', last_char))

        for char in word[1:-1]:
            features.append(('09', char))
            features.append(('10', first_char, char))
            features.append(('11', last_char, char))
        if len(word) == 1:
            features.append(('12', word, prev_char, next_char))
        for i in range(1, len(word)):
            prev_char, char = word[i - 1], word[i]
            if prev_char == char:
                features.append(('13', char, 'consecutive'))
            if i <= 4:
                features.append(('14', word[:i]))
                features.append(('15', word[-i:]))
        if len(word) <= 4:
            features.append(('14', word))
            features.append(('15', word))
        return features

    def evaluate(self, sentences):
        tp, total = 0, 0

        for sentence in sentences:
            total += len(sentence)
            wordseq, tagseq = zip(*sentence)
            preseq = [self.predict(wordseq, i)
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
    lm = LogLinearModel(tags)

    print("Using %d sentences to create the feature space" % (len(train)))
    lm.create_feature_space(train)
    print("The size of the feature space is %d" % lm.D)

    evaluations = []

    print("Using SGD-training algorithm to train the model")
    for i in lm.SGD(train):
        print("iteration %d" % i)
        result = lm.evaluate(train)
        print("\ttrain: %d / %d = %4f" % result)
        result = lm.evaluate(dev)
        print("\tdev: %d / %d = %4f" % result)
        evaluations.append(result)

    print("Successfully evaluated dev data using the model")
    print("Precision: %d / %d = %4f" % max(evaluations, key=lambda x: x[2]))
    print("%4fs elapsed" % (time.time() - start))
