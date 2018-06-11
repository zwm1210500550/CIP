import datetime
import numpy as np


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


class liner_model(object):
    def __init__(self):
        self.train_data = dataset('./data/train.conll')
        self.dev_data = dataset('./data/dev.conll')
        self.features = {}
        self.tag2id = {}
        self.id2tag = {}

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
                    pre_tag = '^'
                else:
                    pre_tag = tags[j - 1]
                template = self.create_feature_template(sentence, j, pre_tag, tags[j])
                for f in template:
                    if f not in self.features.keys():
                        self.features[f] = 0
                for tag in tags:
                    if tag not in self.tag2id:
                        self.tag2id[tag] = len(self.tag2id)
                        self.id2tag[len(self.id2tag)] = tag

        print("the total number of features is %d" % (len(self.features)))

    def dot(self, feature):
        score = 0
        for f in feature:
            if f in self.features.keys():
                score += self.features[f]
        return score

    def score(self, sentence, position, pre_tag, cur_tag):
        feature = self.create_feature_template(sentence, position, pre_tag, cur_tag)
        return self.dot(feature)

    def predict(self, sentence):
        states = len(sentence)
        type = len(self.tag2id)

        max_score = np.zeros((states, type))
        paths = np.zeros((states, type))
        max_score = np.log(max_score)

        for j in range(type):
            max_score[0][j] = self.score(sentence, 0, '^', self.id2tag[j])
            paths[0][j] = -1

        for i in range(1, states):
            for j in range(type):
                last_path = -1
                for k in range(type):
                    cur_score = self.score(sentence, i, self.id2tag[k], self.id2tag[j])
                    score = max_score[i - 1][k] + cur_score
                    if score > max_score[i][j]:
                        max_score[i][j] = score
                        last_path = k
                paths[i][j] = last_path

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

    def train(self, iter):
        max_dev_precision = 0
        for iteration in range(iter):
            print('iterator: %d' % (iteration), flush=True)
            for i in range(len(self.train_data.sentences)):
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                predict = self.predict(sentence)
                if predict != tags:
                    for j in range(len(tags)):
                        if j == 0:
                            gold_pre_tag = '^'
                            predict_pre_tag = '^'
                        else:
                            gold_pre_tag = tags[j - 1]
                            predict_pre_tag = predict[j - 1]
                        gold_feature = self.create_feature_template(sentence, j, gold_pre_tag, tags[j])
                        predict_feature = self.create_feature_template(sentence, j, predict_pre_tag, predict[j])
                        for f in gold_feature:
                            if f in self.features:
                                self.features[f] += 1
                        for f in predict_feature:
                            if f in self.features:
                                self.features[f] -= 1

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data)
            print('\t' + 'train准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision), flush=True)
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data)
            print('\t' + 'dev准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision), flush=True)
            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iteration
                self.save('./result.txt')
        print('iterator = %d , max_dev_precision = %f' % (max_iterator, max_dev_precision), flush=True)


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    lm = liner_model()
    lm.create_feature_space()
    lm.train(20)
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")
