import datetime


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
        self.weights = []
        self.tag_list = []

    def create_feature_template(self, sentence, tag, position):
        template = []
        cur_word = sentence[position]
        cur_tag = tag
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
                template = self.create_feature_template(sentence, tags[j], j)
                for f in template:
                    if f not in self.features.keys():
                        self.features[f] = 0
                for tag in tags:
                    if tag not in self.tag_list:
                        self.tag_list.append(tag)

        print("the total number of features is %d" % (len(self.features)))

    def dot(self, feature):
        score = 0
        for f in feature:
            if f in self.features.keys():
                score += self.features[f]
        return score

    def predict(self, sentence, position):
        score = 0
        predict_tag = 'null'
        for tag in self.tag_list:
            template = self.create_feature_template(sentence, tag, position)
            cur_score = self.dot(template)
            if cur_score > score:
                score = cur_score
                predict_tag = tag
        return predict_tag

    def save(self, path):
        f = open(path, 'w', encoding='utf-8')
        for key, value in self.features.items():
            f.write(key + '\t' + str(value) + '\n')
        f.close()

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

    def train(self):
        max_dev_precision = 0
        max_iterator = -1
        for iterator in range(20):
            for i in range(len(self.train_data.sentences)):
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                for j in range(len(sentence)):
                    predict_tag = self.predict(sentence, j)
                    gold_tag = tags[j]
                    if predict_tag != gold_tag:
                        feature_max = self.create_feature_template(sentence, predict_tag, j)
                        feature_gold = self.create_feature_template(sentence, gold_tag, j)
                        for f in feature_max:
                            if f in self.features.keys():
                                self.features[f] -= 1
                        for f in feature_gold:
                            if f in self.features.keys():
                                self.features[f] += 1
            print('iterator: %d' % (iterator))
            # train_correct_num, total_num, train_precision = self.evaluate(self.train_data)
            # print('\t' + 'train准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data)
            print('\t' + 'dev准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision))
            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iterator
                self.save('./result.txt')
        print('iterator = %d , max_dev_precision = %f' % (max_iterator, max_dev_precision))


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    lm = liner_model()
    lm.create_feature_space()
    lm.train()
    # lm.evaluate(lm.dev_data)
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")
