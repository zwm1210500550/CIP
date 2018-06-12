import datetime
import numpy as np
import sys


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


class loglinear_model(object):
    def __init__(self):
        self.train_data = dataset('./data/train.conll')
        self.dev_data = dataset('./data/dev.conll')
        self.weights = []
        self.features = {}
        self.g = []
        self.tag_list = []
        self.tag_dict = {}
        return

    def create_feature_template(self, sentence, position):
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

        template.append('02:' + '*' + cur_word)
        template.append('03:' + '*' + last_word)
        template.append('04:' + '*' + next_word)
        template.append('05:' + '*' + cur_word + '*' + last_word_last_char)
        template.append('06:' + '*' + cur_word + '*' + next_word_first_char)
        template.append('07:' + '*' + cur_word_first_char)
        template.append('08:' + '*' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + '*' + sentence[position][i])
            template.append('10:' + '*' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + '*' + sentence[position][-1] + '*' + sentence[position][i])
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + '*' + sentence[position][i] + '*' + 'consecutive')

        if len(sentence[position]) > 1 and sentence[position][0] == sentence[position][1]:
            template.append('13:' + '*' + sentence[position][0] + '*' + 'consecutive')

        if len(sentence[position]) == 1:
            template.append('12:' + '*' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + '*' + sentence[position][0:i + 1])
            template.append('15:' + '*' + sentence[position][-(i + 1)::])

        return template

    def create_feature_space(self):
        for i in range(len(self.train_data.sentences)):
            sentence = self.train_data.sentences[i]
            tags = self.train_data.tags[i]
            for j in range(len(sentence)):
                template = self.create_feature_template(sentence, j)
                for f in template:
                    if f not in self.features:
                        self.features[f] = len(self.features)
                for tag in tags:
                    if tag not in self.tag_list:
                        self.tag_list.append(tag)

        self.tag_dict = {t: i for i, t in enumerate(self.tag_list)}
        self.weights = np.zeros(len(self.features) * len(self.tag_dict))
        self.g = np.zeros(len(self.features) * len(self.tag_dict))
        print("the total number of features is %d" % (len(self.features)))

    def dot(self, feature, tag):
        score = 0.0
        offset = self.tag_dict[tag] * len(self.features)
        for f in feature:
            if f in self.features:
                score += self.weights[self.features[f] + offset]
        return score

    def predict(self, sentence, position):
        probilitys = []
        for tag in self.tag_list:
            template = self.create_feature_template(sentence, position)
            cur_score = np.exp(self.dot(template, tag))
            probilitys.append(cur_score)

        s = sum(probilitys)
        for i in range(len(probilitys)):
            probilitys[i] /= s

        return self.tag_list[np.argmax(probilitys)]

    def evaluate(self, data):
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total_num += len(tags)
            for j in range(len(sentence)):
                predict_tag = self.predict(sentence, j)
                if predict_tag == tags[j]:
                    correct_num += 1

        return (correct_num, total_num, correct_num / total_num)

    def basic_train(self, iteration, batch_size=50):
        b = 0
        max_dev_precision = 0
        for iter in range(iteration):
            print('iterator: %d' % (iter))
            for i in range(len(self.train_data.sentences)):
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                for j in range(len(sentence)):
                    b += 1
                    gold_tag = tags[j]
                    gold_feature = self.create_feature_template(sentence, j)
                    gold_offset = self.tag_dict[gold_tag] * len(self.features)
                    for f in gold_feature:
                        if f in self.features:
                            self.g[self.features[f] + gold_offset] += 1

                    feature_list = []
                    prob_list = []
                    for tag in self.tag_dict:
                        feature = self.create_feature_template(sentence, j)
                        feature_list.append((feature, tag))
                        prob_list.append(np.exp(self.dot(feature, tag)))
                    s = sum(prob_list)
                    for k in range(len(feature_list)):
                        for f in feature_list[k][0]:
                            offset = self.tag_dict[feature_list[k][1]] * len(self.features)
                            if f in self.features:
                                self.g[self.features[f] + offset] -= prob_list[k] / s

                    if b == batch_size:
                        self.weights += self.g
                        b = 0
                        self.g = np.zeros(len(self.features) * len(self.tag_dict))
            if b > 0:
                self.weights += self.g
                b = 0
                self.g = np.zeros(len(self.features) * len(self.tag_dict))

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data)
            print('\t' + 'train准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data)
            print('\t' + 'dev准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision), flush=True)
            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iter
        print('iterator = %d , max_dev_precision = %f' % (max_iterator, max_dev_precision), flush=True)

    def optimize_train(self, iteration, batch_size=50):
        b = 0
        max_dev_precision = 0
        eta = 0.5
        for iter in range(iteration):
            print('iterator: %d' % (iter))
            for i in range(len(self.train_data.sentences)):
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                for j in range(len(sentence)):
                    b += 1
                    gold_tag = tags[j]
                    gold_feature = self.create_feature_template(sentence, gold_tag, j)
                    for f in gold_feature:
                        if f in self.features:
                            self.g[self.features[f]] += 1

                    feature_list = []
                    prob_list = []
                    for tag_id in range(len(self.tag_list)):
                        feature = self.create_feature_template(sentence, self.tag_list[tag_id], j)
                        feature_list.append(feature)
                        prob_list.append(np.exp(self.dot(feature)))
                    s = sum(prob_list)
                    for k in range(len(feature_list)):
                        for f in feature_list[k]:
                            if f in self.features:
                                self.g[self.features[f]] -= prob_list[k] / s

                    if b == batch_size:
                        # self.weights -= 0.000001 * self.weights
                        self.weights += self.g
                        b = 0
                        eta = max(eta * 0.999, 0.00001)
                        # print(eta)
                        self.g = np.zeros(len(self.features))
            if b > 0:
                # self.weights -= 0.000001 * self.weights
                self.weights += self.g
                b = 0
                eta = max(eta * 0.999, 0.00001)
                # print(eta)
                self.g = np.zeros(len(self.features) * len(self.tag_dict))

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data)
            print('\t' + 'train准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data)
            print('\t' + 'dev准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision), flush=True)
            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iter
        print('iterator = %d , max_dev_precision = %f' % (max_iterator, max_dev_precision), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        optimized = sys.argv[1]
    else:
        optimized = False
    starttime = datetime.datetime.now()
    print('start...')
    model = loglinear_model()
    model.create_feature_space()
    if optimized == 'optimize':
        model.optimize_train(40, 50)
    else:
        model.basic_train(40, 50)
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")
