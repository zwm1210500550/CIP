import datetime
import numpy as np
import random
from config import config


class dataset(object):
    def __init__(self, filename):
        self.filename = filename
        self.sentences = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        f = open(filename, encoding='utf-8')
        while (True):
            line = f.readline()
            if not line:
                break
            if line == '\n':
                self.sentences.append(sentence)
                self.tags.append(tag)
                sentence = []
                tag = []
            else:
                sentence.append(line.split()[1])
                tag.append(line.split()[3])
                word_num += 1
        self.sentences_num = len(self.sentences)
        self.word_num = word_num

        print('%s:共%d个句子,共%d个词。' % (filename, self.sentences_num, self.word_num))
        f.close()

    def shuffle(self):
        temp = [(s, t) for s, t in zip(self.sentences, self.tags)]
        random.shuffle(temp)
        self.sentences = []
        self.tags = []
        for s, t in temp:
            self.sentences.append(s)
            self.tags.append(t)


class CRF(object):
    def __init__(self):
        self.train_data = dataset(train_data_file)
        self.dev_data = dataset(dev_data_file)
        self.test_data = dataset(test_data_file)
        self.features = {}
        self.weights = []
        self.v = []
        self.tag2id = {}
        self.id2tag = {}
        self.tags = []
        self.EOS = 'EOS'
        self.BOS = 'BOS'

    def create_feature_template(self, sentence, position, pre_tag):
        template = []
        cur_word = sentence[position]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if position == 0:
            last_word = '##'
            last_word_last_char = '#'
        else:
            last_word = sentence[position - 1]
            last_word_last_char = sentence[position - 1][-1]

        if position == len(sentence) - 1:
            next_word = '$$'
            next_word_first_char = '$'
        else:
            next_word = sentence[position + 1]
            next_word_first_char = sentence[position + 1][0]

        template.append('01:' + pre_tag)
        template.append('02:' + cur_word)
        template.append('03:' + last_word)
        template.append('04:' + next_word)
        template.append('05:' + cur_word + '*' + last_word_last_char)
        template.append('06:' + cur_word + '*' + next_word_first_char)
        template.append('07:' + cur_word_first_char)
        template.append('08:' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + sentence[position][i])
            template.append('10:' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + sentence[position][-1] + '*' + sentence[position][i])
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + sentence[position][i] + '*' + 'consecutive')

        if len(sentence[position]) > 1 and sentence[position][0] == sentence[position][1]:
            template.append('13:' + sentence[position][0] + '*' + 'consecutive')

        if len(sentence[position]) == 1:
            template.append('12:' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + sentence[position][0:i + 1])
            template.append('15:' + sentence[position][-(i + 1)::])
        return template

    def create_feature_space(self):
        for i in range(len(self.train_data.sentences)):
            sentence = self.train_data.sentences[i]
            tags = self.train_data.tags[i]
            for j in range(len(sentence)):
                if j == 0:
                    pre_tag = self.BOS
                else:
                    pre_tag = tags[j - 1]
                template = self.create_feature_template(sentence, j, pre_tag)
                for f in template:
                    if f not in self.features:
                        self.features[f] = len(self.features)
                for tag in tags:
                    if tag not in self.tags:
                        self.tags.append(tag)
        self.tags = sorted(self.tags)
        self.tag2id = {t: i for i, t in enumerate(self.tags)}
        self.id2tag = {i: t for i, t in enumerate(self.tags)}
        self.weights = np.zeros((len(self.features), len(self.tag2id)))
        self.g = np.zeros((len(self.features), len(self.tag2id)))
        print("the total number of features is %d" % (len(self.features)))

    def score(self, feature):
        scores = [self.weights[self.features[f]]
                  for f in feature if f in self.features]
        return np.sum(scores, axis=0)

    def predict(self, sentence):
        states = len(sentence)
        type = len(self.tag2id)

        max_score = np.zeros((states, type))
        paths = np.zeros((states, type), dtype='int')

        feature = self.create_feature_template(sentence, 0, self.BOS)
        max_score[0] = self.score(feature)

        for i in range(1, states):
            tag_features = [
                self.create_feature_template(sentence, i, prev_tag)
                for prev_tag in self.tags
            ]
            scores = [max_score[i - 1][j] + self.score(fs)
                      for j, fs in enumerate(tag_features)]
            paths[i] = np.argmax(scores, axis=0)
            max_score[i] = np.max(scores, axis=0)
        prev = np.argmax(max_score[-1])

        predict = [prev]
        for i in range(len(sentence) - 1, 0, -1):
            prev = paths[i, prev]
            predict.append(prev)
        return [self.tags[i] for i in reversed(predict)]

    def evaluate(self, data):
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total_num += len(tags)
            predict = self.predict(sentence)
            for j in range(len(tags)):
                if tags[j] == predict[j]:
                    correct_num += 1

        return (correct_num, total_num, correct_num / total_num)

    def logsumexp(self, scores):
        s_max = np.max(scores)
        return s_max + np.log(np.exp(scores - s_max).sum())

    def forward(self, sentence):
        scores = np.zeros((len(sentence), len(self.tags)))
        feature = self.create_feature_template(sentence, 0, self.BOS)
        scores[0] = (self.score(feature))

        for i in range(1, len(sentence)):
            for j in range(len(self.tags)):
                features = [self.create_feature_template(sentence, i, pre_tag) for pre_tag in self.tags]
                score = [self.score(feature)[j] for feature in features]
                scores[i][j] = self.logsumexp(score + scores[i - 1])
        return scores

    def backward(self, sentence):
        states = len(sentence)
        scores = np.zeros((states, len(self.tags)))

        for i in range(states - 2, -1, -1):
            features = [self.create_feature_template(sentence, i + 1, pre_tag) for pre_tag in self.tags]
            for j in range(len(self.tags)):
                score = scores[i + 1] + self.score(features[j])
                scores[i][j] = self.logsumexp(score)
        return scores

    def update_gradient(self, sentence, tags):
        for i in range(len(sentence)):
            if i == 0:
                pre_tag = self.BOS
            else:
                pre_tag = tags[i - 1]
            cur_tag = tags[i]
            feature = self.create_feature_template(sentence, i, pre_tag)
            for f in feature:
                if f in self.features:
                    self.g[self.features[f]][self.tag2id[cur_tag]] += 1

        forward_scores = self.forward(sentence)
        backward_scores = self.backward(sentence)
        log_dinominator = self.logsumexp(forward_scores[-1])  # 得到分母log(Z(S))
        for i in range(len(sentence)):
            if i == 0:
                pre_tag = self.BOS
                template = self.create_feature_template(sentence, i, pre_tag)
                score = self.score(template)
                for cur_tag in self.tags:
                    forward = 0
                    backward = backward_scores[i][self.tag2id[cur_tag]]
                    p = np.exp(forward + score[self.tag2id[cur_tag]] + backward - log_dinominator)

                    for f in template:
                        if f in self.features:
                            self.g[self.features[f]][self.tag2id[cur_tag]] -= p
            else:
                for pre_tag in self.tags:
                    template = self.create_feature_template(sentence, i, pre_tag)
                    score = (self.score(template))
                    for cur_tag in self.tags:
                        forward = forward_scores[i - 1][self.tag2id[pre_tag]]
                        backward = backward_scores[i][self.tag2id[cur_tag]]
                        p = np.exp(forward + score[self.tag2id[cur_tag]] + backward - log_dinominator)

                        for f in template:
                            if f in self.features:
                                self.g[self.features[f]][self.tag2id[cur_tag]] -= p

    def SGD_train(self, iteration=20, batchsize=1, shuffle=False):
        max_dev_precision = 0
        for iter in range(iteration):
            b = 0
            starttime = datetime.datetime.now()
            print('iterator: %d' % (iter), flush=True)
            if shuffle:
                print('shuffle the train data...', flush=True)
                self.train_data.shuffle()

            for i in range(len(self.train_data.sentences)):
                # print('sentence' + str(i))
                b += 1
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                self.update_gradient(sentence, tags)
                if b == batchsize:
                    self.weights += self.g
                    self.g = np.zeros((len(self.features), len(self.tag2id)))
                    b = 0

            if b > 0:
                self.weights += self.g
                self.g = np.zeros((len(self.features), len(self.tag2id)))
                b = 0

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data)
            print('\t' + 'train准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision), flush=True)
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data)
            print('\t' + 'dev准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision), flush=True)

            if 'test.conll' in self.test_data.filename:
                test_correct_num, test_num, test_precision = self.evaluate(self.test_data)
                print('\t' + 'test准确率：%d / %d = %f' % (test_correct_num, test_num, test_precision), flush=True)

            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iter

            endtime = datetime.datetime.now()
            print("\titeration executing time is " + str((endtime - starttime)) + " s", flush=True)
        print('iterator = %d , max_dev_precision = %f' % (max_iterator, max_dev_precision), flush=True)


if __name__ == '__main__':
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    iterator = config['iterator']
    batchsize = config['batchsize']
    shuffle = config['shuffle']

    starttime = datetime.datetime.now()
    crf = CRF()
    crf.create_feature_space()
    print(crf.tag2id)
    crf.SGD_train(iterator, batchsize, shuffle)

    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")
