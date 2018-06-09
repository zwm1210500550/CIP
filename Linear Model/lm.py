# -*- coding: utf-8 -*-

import time

import numpy as np


def preprocessing(ftrain):
    with open(ftrain, 'r') as train:
        it = iter(train)
        sentences = []
        for line in it:
            sentence = []
            while len(line) > 1:
                splits = line.split()
                sentence.append((splits[1], splits[3]))
                line = next(it)
            sentences.append(sentence)
        return sentences


class LinearModel(object):

    def __init__(self, words, tags):
        # 所有不同的单词
        self.words = words
        # 所有不同的词性
        self.tags = tags

    def create_feature_space(self, sentences):
        feature_space = set()
        for sentence in sentences:
            wordseq, tagseq = zip(*sentence)
            for i, tag in enumerate(tagseq):
                features = self.instantialize(wordseq, i, tag)
                feature_space.update(features)

        # 特征空间
        self.epsilon = list(feature_space)
        # 特征对应索引的字典
        self.feadict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征权重
        self.weights = np.zeros(len(self.epsilon), dtype='int')
        # 特征空间维度
        self.D = len(self.epsilon)

    def online_train(self, sentences, iter=20):
        for it in range(iter):
            for sentence in sentences:
                wordseq, tagseq = zip(*sentence)
                predicts = self.predict(wordseq)
                for i, (tag, pos) in enumerate(zip(tagseq, predicts)):
                    if tag != pos:
                        features = self.instantialize(wordseq, i, tag)
                        pos_features = self.instantialize(wordseq, i, pos)
                        for cf, pf in zip(features, pos_features):
                            if pf in self.feadict:
                                self.weights[self.feadict[pf]] -= 1
                            if cf in self.feadict:
                                self.weights[self.feadict[cf]] += 1

            tagseqs, preseqs = [], []
            for sentence in sentences:
                ws, ts = zip(*sentence)
                tagseqs.append(ts)
                preseqs.append(lm.predict(ws))
            tp, total, precision = lm.evaluate(tagseqs, preseqs)
            print('iteration %d: %d / %d = %4f' % (it, tp, total, precision))

    def instantialize(self, wordseq, index, tag):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else "$$"
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else "##"
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        features = []
        features.append(('02', tag, word))
        features.append(('03', tag, prev_word))
        features.append(('04', tag, next_word))
        features.append(('05', tag, word, prev_char))
        features.append(('06', tag, word, next_char))
        features.append(('07', tag, first_char))
        features.append(('08', tag, last_char))

        for char in word[1:-1]:
            features.append(('09', tag, char))
            features.append(('10', tag, first_char, char))
            features.append(('11', tag, last_char, char))
        if len(word) == 1:
            features.append(('12', tag, word, prev_char, next_char))
        for i in range(1, len(word) - 1):
            char, next_char = word[i], word[i + 1]
            if char == next_char:
                features.append(('13', tag, char, 'consecutive'))
            if i <= 4:
                features.append(('14', tag, word[:i]))
                features.append(('15', tag, word[-i:]))
        return features

    def predict(self, wordseq):
        args = [
            np.argmax([self.score(self.instantialize(wordseq, i, tag))
                       for tag in self.tags])
            for i in range(len(wordseq))
        ]
        return [self.tags[i] for i in args]

    def score(self, features):
        return sum(self.weights[self.feadict[f]]
                   for f in features
                   if f in self.feadict)

    def evaluate(self, tagseqs, predicts):
        tp, total = 0, 0
        for tagseq, predict in zip(tagseqs, predicts):
            total += len(tagseq)
            tp += sum(t == p for t, p in zip(tagseq, predict))
        precision = tp / total
        return tp, total, precision


if __name__ == '__main__':
    sentences = preprocessing('data/train.conll')

    all_words, all_tags = zip(*np.vstack(sentences))
    words, tags = list(set(all_words)), list(set(all_tags))

    start = time.time()

    print("Creating Linear Model with %d words and %d tags"
          % (len(words), len(tags)))
    lm = LinearModel(words, tags)

    print("Using %d sentences to create the feature space"
          % (len(sentences)))
    lm.create_feature_space(sentences)
    print("The size of the feature space is %d" % lm.D)

    print("Using online-training algorithm to train the model")
    lm.online_train(sentences)

    sentences = preprocessing('data/dev.conll')
    tagseqs, preseqs = [], []

    print("Using the trained model to tag %d sentences"
          % len(sentences))
    for sentence in sentences:
        ws, ts = zip(*sentence)
        tagseqs.append(ts)
        preseqs.append(lm.predict(ws))

    print("Evaluating the result")
    tp, total, precision = lm.evaluate(tagseqs, preseqs)
    print("precision: %d / %d = %4f" % (tp, total, precision))
    print("%4fs elapsed" % (time.time() - start))
