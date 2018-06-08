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

    print("Creating HMM with %d words and %d tags"
          % (len(words), len(tags)))
    hmm = HMM(words, tags)

    print("Using %d sentences to train the HMM"
          % (len(sentences)))
    hmm.train(sentences)

    sentences = preprocessing('data/dev.conll')
    tagseqs, preseqs = [], []

    print("Using Viterbi algorithm to tag %d sentences"
          % len(sentences))
    start = time.time()
    for sentence in sentences:
        ws, ts = zip(*sentence)
        tagseqs.append(ts)
        preseqs.append(hmm.viterbi(ws))
    print("Successfully tagged all the sentences. %4fs elapsed"
          % (time.time() - start))

    print("Evaluating the result")
    tp, total, precision = hmm.evaluate(tagseqs, preseqs)
    print("tp: %d\ntotal: %d\nprecision: %4f"
          % (tp, total, precision))
