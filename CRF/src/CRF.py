import datetime
import numpy as np
import random
from scipy.misc import logsumexp
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

    def create_feature_template(self, sentence, position, pre_tag, cur_tag):
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

        template.append('01:' + cur_tag + '*' + pre_tag)
        template.append('02:' + cur_tag + '*' + cur_word)
        template.append('03:' + cur_tag + '*' + last_word)
        template.append('04:' + cur_tag + '*' + next_word)
        template.append('05:' + cur_tag + '*' + cur_word + '*' + last_word_last_char)
        template.append('06:' + cur_tag + '*' + cur_word + '*' + next_word_first_char)
        template.append('07:' + cur_tag + '*' + cur_word_first_char)
        template.append('08:' + cur_tag + '*' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + cur_tag + '*' + sentence[position][i])
            template.append('10:' + cur_tag + '*' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + cur_tag + '*' + sentence[position][-1] + '*' + sentence[position][i])
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + cur_tag + '*' + sentence[position][i] + '*' + 'consecutive')

        if len(sentence[position]) > 1 and sentence[position][0] == sentence[position][1]:
            template.append('13:' + cur_tag + '*' + sentence[position][0] + '*' + 'consecutive')

        if len(sentence[position]) == 1:
            template.append('12:' + cur_tag + '*' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + cur_tag + '*' + sentence[position][0:i + 1])
            template.append('15:' + cur_tag + '*' + sentence[position][-(i + 1)::])
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
                template = self.create_feature_template(sentence, j, pre_tag, tags[j])
                for f in template:
                    if f not in self.features:
                        self.features[f] = len(self.features)
                for tag in tags:
                    if tag not in self.tags:
                        self.tags.append(tag)
        self.tags = sorted(self.tags)
        self.tag2id = {t: i for i, t in enumerate(self.tags)}
        self.id2tag = {i: t for i, t in enumerate(self.tags)}
        self.weights = np.zeros(len(self.features))
        self.g = np.zeros(len(self.features))
        print("the total number of features is %d" % (len(self.features)))

    def score(self, feature):
        score = 0
        for f in feature:
            if f in self.features:
                score += self.weights[self.features[f]]
        return score

    def predict(self, sentence):
        states = len(sentence)
        type = len(self.tag2id)

        max_score = np.zeros((states, type))
        paths = np.zeros((states, type))

        for j in range(type):
            feature = self.create_feature_template(sentence, 0, self.BOS, self.tags[j])
            max_score[0][j] = self.score(feature)
            paths[0][j] = -1

        # 动态规划
        for i in range(1, states):
            for j in range(type):
                features = [self.create_feature_template(sentence, i, tag, self.tags[j]) for tag in self.tags]
                cur_score = [self.score(feature) for feature in features]
                max_score[i][j] = max(cur_score + max_score[i - 1])
                paths[i][j] = np.argmax(cur_score + max_score[i - 1])

        gold_path = []
        cur_state = states - 1
        step = np.argmax(max_score[cur_state])
        gold_path.insert(0, self.id2tag[step])
        while True:
            step = int(paths[cur_state][step])
            if step == -1:
                break
            gold_path.insert(0, self.id2tag[step])
            cur_state -= 1
        return gold_path

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


    def forward(self, sentence):
        path_scores = np.zeros((len(sentence), len(self.tags)))

        features = [self.create_feature_template(sentence, 0, self.BOS, tag) for tag in self.tags]
        path_scores[0] = [self.score(feature) for feature in features]

        for i in range(1, len(sentence)):
            for j in range(len(self.tags)):
                features = [self.create_feature_template(sentence, i, pre_tag, self.tags[j]) for pre_tag in self.tags]
                scores = [self.score(feature) for feature in features]
                path_scores[i][j] = logsumexp(path_scores[i - 1] + scores)
        return path_scores

    def backward(self, sentence):
        path_scores = np.zeros((len(sentence), len(self.tags)))

        for i in reversed(range(len(sentence) - 1)):
            for j in range(len(self.tags)):
                features = [self.create_feature_template(sentence, i + 1, self.tags[j], tag) for tag in self.tags]
                scores = [self.score(feature) for feature in features]
                path_scores[i][j] = logsumexp(path_scores[i + 1] + scores)
        return path_scores

    def update_gradient(self, sentence, tags):
        for i in range(len(sentence)):
            if i == 0:
                pre_tag = self.BOS
            else:
                pre_tag = tags[i - 1]
            cur_tag = tags[i]
            feature = self.create_feature_template(sentence, i, pre_tag, cur_tag)
            for f in feature:
                if f in self.features:
                    self.g[self.features[f]] += 1

        forward_scores = self.forward(sentence)
        backward_scores = self.backward(sentence)
        dinominator = logsumexp(forward_scores[-1])

        for i in range(len(sentence)):
            if i == 0:
                pre_tag = self.BOS
                for cur_tag in self.tags:
                    template = self.create_feature_template(sentence, i, pre_tag, cur_tag)
                    score = self.score(template)
                    forward = 0
                    backward = backward_scores[i][self.tag2id[cur_tag]]
                    p = np.exp(forward + score + backward - dinominator)

                    for f in template:
                        if f in self.features:
                            self.g[self.features[f]] -= p
            else:
                for pre_tag in self.tags:
                    for cur_tag in self.tags:
                        template = self.create_feature_template(sentence, i, pre_tag, cur_tag)
                        score = self.score(template)
                        forward = forward_scores[i - 1][self.tag2id[pre_tag]]
                        backward = backward_scores[i][self.tag2id[cur_tag]]
                        p = np.exp(forward + score + backward - dinominator)

                        for f in template:
                            if f in self.features:
                                self.g[self.features[f]] -= p

    def SGD_train(self, iteration=20, batchsize=1, shuffle=False, regulization=False, step_opt=False, eta=0.5,
                  C=0.0001):
        max_dev_precision = 0
        if regulization:
            print('add regulization...C=%f' % (C), flush=True)
        if step_opt:
            print('add step optimization...eta=%f' % (eta), flush=True)
        for iter in range(iteration):
            b = 0
            starttime = datetime.datetime.now()
            print('iterator: %d' % (iter), flush=True)
            if shuffle:
                print('shuffle the train data...', flush=True)
                self.train_data.shuffle()
            for i in range(len(self.train_data.sentences)):
                b += 1
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                self.update_gradient(sentence, tags)
                if b == batchsize:
                    if step_opt:
                        self.weights += eta * self.g
                    else:
                        self.weights += self.g
                    if regulization:
                        self.weights -= C * eta * self.weights
                    eta = max(eta * 0.999, 0.00001)
                    self.g = np.zeros(len(self.features))
                    b = 0

            if b > 0:
                if step_opt:
                    self.weights += eta * self.g
                else:
                    self.weights += self.g
                if regulization:
                    self.weights -= C * eta * self.weights
                eta = max(eta * 0.999, 0.00001)
                self.g = np.zeros(len(self.features))
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
    regulization = config['regulization']
    step_opt = config['step_opt']
    C = config['C']
    eta = config['eta']

    starttime = datetime.datetime.now()
    crf = CRF()
    crf.create_feature_space()
    print(crf.tag2id)
    crf.SGD_train(iterator, batchsize, shuffle, regulization, step_opt, eta, C)
    # print(crf.forward(['你', '好', '啊']))
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")
