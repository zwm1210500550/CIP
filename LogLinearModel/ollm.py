# -*- coding: utf-8 -*-

import pickle
import random
import time

import numpy as np
from scipy.misc import logsumexp


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


class LogLinearModel(object):

    def __init__(self, tags):
        # 所有不同的词性
        self.tags = tags
        # 词性对应索引的字典
        self.tdict = {t: i for i, t in enumerate(tags)}

        self.N = len(self.tags)

    def create_feature_space(self, sentences):
        feature_space = set()
        for sentence in sentences:
            wordseq, tagseq = zip(*sentence)
            for i, tag in enumerate(tagseq):
                fvector = self.instantiate(wordseq, i)
                feature_space.update(fvector)

        # 特征空间
        self.epsilon = list(feature_space)
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.D = len(self.epsilon)

        # 特征权重
        self.W = np.zeros((self.D, self.N))

    def SGD(self, sentences, batch_size=50, c=0.0001, eta=0.5, epochs=20):
        training_data = []
        for sentence in sentences:
            wordseq, tagseq = zip(*sentence)
            for i, tag in enumerate(tagseq):
                training_data.append((wordseq, i, tag))
        for epoch in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[i:i + batch_size]
                       for i in range(0, len(training_data), batch_size)]
            for batch in batches:
                # 根据批次数据更新权重
                self.update(batch, c, max(eta, 0.00001))
                eta *= 0.999
            yield epoch

    def update(self, batch, c, eta):
        gradients = np.zeros((self.D, self.N))
        for wordseq, i, tag in batch:
            ti = self.tdict[tag]

            fvector = self.instantiate(wordseq, i)
            scores = self.score(fvector)
            probs = np.exp(scores - logsumexp(scores))

            for f in fvector:
                if f in self.fdict:
                    fi = self.fdict[f]
                    gradients[fi][ti] += 1
                    gradients[fi] -= probs

        self.W -= eta * c * self.W
        self.W += eta * gradients

    def predict(self, wordseq, index):
        fvector = self.instantiate(wordseq, index)
        scores = self.score(fvector)
        return self.tags[np.argmax(scores)]

    def score(self, fvector):
        scores = [self.W[self.fdict[f]]
                  for f in fvector if f in self.fdict]
        return np.sum(scores, axis=0)

    def instantiate(self, wordseq, index):
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

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            hmm = pickle.load(f)
        return hmm


if __name__ == '__main__':
    train = preprocess('data/train.conll')
    dev = preprocess('data/dev.conll')

    all_words, all_tags = zip(*np.vstack(train))
    tags = sorted(set(all_tags))

    start = time.time()

    print("Creating Log-Linear Model with %d tags" % (len(tags)))
    lm = LogLinearModel(tags)

    print("Using %d sentences to create the feature space" % (len(train)))
    lm.create_feature_space(train)
    print("The size of the feature space is %d" % lm.D)

    evaluations = []

    print("Using SGD algorithm to train the model")
    for epoch in lm.SGD(train):
        print("Epoch %d" % epoch)
        result = lm.evaluate(train)
        print("\ttrain: %d / %d = %4f" % result)
        result = lm.evaluate(dev)
        print("\tdev: %d / %d = %4f" % result)
        evaluations.append(result)

    print("Successfully evaluated dev data using the model")
    print("Precision: %d / %d = %4f" % max(evaluations, key=lambda x: x[2]))
    print("%4fs elapsed" % (time.time() - start))
