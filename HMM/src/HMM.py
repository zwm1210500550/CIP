import numpy as np
import datetime

alpha = 0.3


def read_data(filename):
    f = open(filename, encoding='utf-8')
    data = []
    sentence = []
    while (True):
        line = f.readline()
        if not line:
            break
        if line != '\n':
            word = line.split()[1]
            tag = line.split()[3]
            sentence.append((word, tag))
        else:
            data.append(sentence)
            sentence = []
    f.close()
    return data


class Binary_HMM(object):
    # 属性1：train_data存放所有的训练句子，[[(戴相龙,NR),(,),(,)....],[],[]....]
    # 属性2：tag_dict存放训练集中所有的tag，及其编号,考虑了起始和终止词性
    # 属性3：word_dict存放训练集中所有的word，及其编号,加入了未知词
    # 属性4：transition_matrix转移概率矩阵,第(i,j)个元素表示词性j在词性i后面的概率,最后一行是start，最后一列是stop
    # 属性5: launch_matrix发射概率矩阵,第(i,j)个元素表示词性i发射到词j的概率,最后一列是未知词
    def __init__(self, train_data):
        self.train_data = train_data
        self.tag_dict = {}
        self.word_dict = {}
        for sentence in train_data:
            for word, tag in sentence:
                if word not in self.word_dict.keys():
                    self.word_dict[word] = len(self.word_dict)
                if tag not in self.tag_dict.keys():
                    self.tag_dict[tag] = len(self.tag_dict)

        self.tag_dict['BOS'] = len(self.tag_dict)
        self.tag_dict['EOS'] = len(self.tag_dict)
        self.word_dict['???'] = len(self.word_dict)

        self.transition_matrix = np.zeros(
            [len(self.tag_dict) - 1, len(self.tag_dict) - 1])  # 第(i,j)个元素表示词性j在词性i后面的概率,最后一行是start，最后一列是stop
        self.launch_matrix = np.zeros([len(self.tag_dict) - 2, len(self.word_dict)])  # 第(i,j)个元素表示词性i发射到词j的概率

    def launch_params(self):
        for sentence in self.train_data:
            for word, tag in sentence:
                self.launch_matrix[self.tag_dict[tag]][self.word_dict[word]] += 1
        for i in range(len(self.launch_matrix)):
            s = sum(self.launch_matrix[i])
            for j in range(len(self.launch_matrix[i])):
                self.launch_matrix[i][j] = (self.launch_matrix[i][j] + alpha) / (s + alpha * (len(self.word_dict)))

    def transition_params(self):
        for i in range(len(self.train_data)):
            for j in range(len(self.train_data[i]) + 1):
                if j == 0:
                    self.transition_matrix[-1][self.tag_dict[train_data[i][j][1]]] += 1
                elif j == len(self.train_data[i]):
                    self.transition_matrix[self.tag_dict[train_data[i][j - 1][1]]][-1] += 1
                else:
                    self.transition_matrix[self.tag_dict[train_data[i][j - 1][1]]][
                        self.tag_dict[train_data[i][j][1]]] += 1

        for i in range(len(self.transition_matrix)):
            s = sum(self.transition_matrix[i])
            for j in range(len(self.transition_matrix[i])):
                self.transition_matrix[i][j] = (self.transition_matrix[i][j] + alpha) / (
                        s + alpha * (len(self.tag_dict) - 1))
                # self.transition_matrix[i][j]/=s

    def write_matrix(self, matrix, path):
        f = open(path, 'w', encoding='utf-8')
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                f.write('%-10.8f ' % (matrix[i][j]))
            f.write('\n')
        f.close()

    def viterbi(self, word_list):
        word_index = []
        for word in word_list:
            if word in self.word_dict.keys():
                word_index.append(self.word_dict[word])
            else:
                word_index.append(self.word_dict['???'])

        states = len(word_list) + 1
        type = len(self.tag_dict) - 2
        max_p = np.zeros((states, type))
        path = np.zeros((states, type))

        # 遇到0会自动变成-inf，但会有警告
        # max_p = np.log(max_p)
        launch_matrix = np.log(self.launch_matrix)
        transition_matrix = np.log(self.transition_matrix)

        # max_p = max_p
        # launch_matrix = self.launch_matrix
        # transition_matrix = self.transition_matrix
        # 初始化起始状态
        for i in range(type):
            path[0][i] = -1
            max_p[0][i] = transition_matrix[-1][i] + launch_matrix[i][word_index[0]]

        # 动态规划
        for i in range(1, states):
            # 到达end状态有点区别
            if i == states - 1:
                for k in range(type):
                    max_p[i][k] = max_p[i - 1][k] + transition_matrix[k][-1]
                    last_path = k
                    path[i][k] = last_path
            else:
                for j in range(type):
                    # 最原始的viterbi算法,需要三层循环，执行速度大约1分多钟
                    # last_path = -1
                    # for k in range(type):
                    #     score = max_p[i - 1][k] + transition_matrix[k][j] + launch_matrix[j][
                    #         word_index[i]]
                    #     if score > max_p[i][j]:
                    #         max_p[i][j] = score
                    #         last_path = k
                    # path[i][j] = last_path

                    # 利用numpy特性可以减少一层循环，加快速度
                    prob = max_p[i - 1] + transition_matrix[:-1, j] + launch_matrix[j][word_index[i]]
                    path[i][j] = np.argmax(prob)
                    max_p[i][j] = max(prob)

        gold_path = []
        cur_state = states - 1
        step = np.argmax(max_p[cur_state])
        while (True):
            step = int(path[cur_state][step])
            if step == -1:
                break
            gold_path.insert(0, step)
            cur_state -= 1
        return gold_path

    def evaluate(self, test_data):
        total_words = 0
        correct_words = 0
        sentence_num = 0
        print('正在评估测试集...')
        f = open('./data/predict.txt', 'w', encoding='utf-8')
        for sentence in test_data:
            sentence_num += 1
            # print('正在预测第%d个句子' % (sentence_num))
            word_list = []
            tag_list = []
            for word, tag in sentence:
                word_list.append(word)
                tag_list.append(tag)
            predict = self.viterbi(word_list)
            total_words += len(sentence)
            for i in range(len(predict)):
                f.write(word_list[i] + '	_	' + list(self.tag_dict.keys())[predict[i]] + '\n')
                if predict[i] == self.tag_dict[tag_list[i]]:
                    correct_words += 1
            f.write('\n')
        f.close()
        print('共%d个句子' % (sentence_num))
        print('共%d个单词，预测正确%d个单词' % (total_words, correct_words))
        print('准确率：%f' % (correct_words / total_words))


if __name__ == '__main__':
    train_data = read_data('./data/train.conll')
    HMM = Binary_HMM(train_data)
    HMM.launch_params()
    HMM.transition_params()
    test_data = read_data('./data/dev.conll')
    starttime = datetime.datetime.now()
    HMM.evaluate(test_data)
    endtime = datetime.datetime.now()
    print('共耗时' + str(endtime - starttime))
