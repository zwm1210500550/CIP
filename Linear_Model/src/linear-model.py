import numpy as np
import datetime
import random
from config import config

train_data_file = config['train_data_file']
dev_data_file = config['dev_data_file']
test_data_file = config['test_data_file']
averaged = config['averaged']
iterator = config['iterator']
stop_iterator = config['stop_iterator']


def data_handle(filename):
    sentences = list()
    sentence = list()
    sentence_num = 0
    word_num = 0
    with open(filename,"r") as dataTxt:
        for line in dataTxt:
            if len(line) == 1:
                sentences.append(sentence)
                sentence = []
                sentence_num += 1
            else:
                word = line.split()[1]
                tag = line.split()[3]
                sentence.append((word,tag))
                word_num += 1
    print("文件%s:共%d个句子，%d个词" % (filename,sentence_num,word_num))
    return sentences    #sentences格式：[[(戴相龙,NR),(,),(,)....],[],[]....]


class linear_model(object):
    def __init__(self):
        self.train_data = data_handle(train_data_file)
        self.dev_data = data_handle(dev_data_file)
        self.test_data = data_handle(test_data_file)
        tags = set()
        for sentence in self.train_data:
            for word,tag in sentence:
                tags.add(tag)
        self.tags = list(tags)      #tag的列表
        self.N = len(self.tags)     #tag的数目
    
    def create_feature_template(self,sentence,tag,index):    #创建某个tag相对于sentence句子中第index个词的特征
        words = [_word for _word,_tag in sentence]
        words.insert(0,"^")     #插入句首开始符
        words.append("$")       #插入句尾结束符
        feature_template = set()
        feature_template.add("02:" + tag + "*" + words[index+1])
        feature_template.add("03:" + tag + "*" + words[index])
        feature_template.add("04:" + tag + "*" + words[index+2])
        feature_template.add("05:" + tag + "*" + words[index+1] + "*" + words[index][-1])
        feature_template.add("06:" + tag + "*" + words[index+1] + "*" + words[index+2][0])
        feature_template.add("07:" + tag + "*" + words[index+1][0])
        feature_template.add("08:" + tag + "*" + words[index+1][-1])
        for j in range(1,len(words[index+1])-1):
            feature_template.add("09:" + tag + "*" + words[index+1][j])
            feature_template.add("10:" + tag + "*" + words[index+1][0] + "*" + words[index+1][j])
            feature_template.add("11:" + tag + "*" + words[index+1][-1] + "*" + words[index+1][j])
        if len(words[index+1]) == 1:
            feature_template.add("12:" + tag + "*" + words[index+1] + "*" + words[index][-1] + "*" + words[index+2][0])
        for j in range(len(words[index+1])-1):
            if words[index+1][j] == words[index+1][j+1]:
                feature_template.add("13:" + tag + "*" + words[index+1][j] + "*" + "consecutive")
        for j in range(1,min(len(words[index+1]),4)+1):
            feature_template.add("14:" + tag + "*" + words[index+1][0:j])
            feature_template.add("15:" + tag + "*" + words[index+1][-j:])
        return feature_template

    def create_feature_space(self):
        feature_space = set()
        for sentence in self.train_data:
            tags = [tag for word,tag in sentence]
            for i in range(len(sentence)):
                feature_space |= self.create_feature_template(sentence,tags[i],i)
        self.feature_space_list = list(feature_space)
        self.feature_space = {feature:index for index,feature in enumerate(self.feature_space_list)}    #特征空间是一个字典，格式：{“NR*戴相龙:0”,"",....}
        self.E = len(self.feature_space)    #特征空间的数目


    def online_training(self):
        self.w = np.zeros(self.E,dtype=int)         
        self.v = np.zeros(self.E,dtype=int)         
        score_matrix = np.zeros(self.N,dtype=int)
        if averaged == "TRUE":
            print("使用累加特征权重：")
        else:
            print("不使用累加特征权重：")
        max_dev_data_precision = 0
        max_dev_data_precision_index = 0
        for iter in range(iterator):
            print("第%d次迭代：" % (iter+1))
            startime = datetime.datetime.now()
            print("正在打乱训练数据...")
            random.shuffle(self.train_data)
            print("数据已打乱")
            for sentence in self.train_data:
                for i in range(len(sentence)):
                    right_tag = sentence[i][1]
                    for tag_index in range(self.N):
                        features = self.create_feature_template(sentence,self.tags[tag_index],i)
                        score = 0
                        for feature in features :
                            if feature in self.feature_space:
                                score += self.w[self.feature_space.get(feature)]
                        score_matrix[tag_index] = score 
                    max_tag = self.tags[np.argmax(score_matrix)]
                    if right_tag != max_tag:
                        right_tag_feature = self.create_feature_template(sentence,right_tag,i)
                        for feature in right_tag_feature:
                            if feature in self.feature_space:
                                self.w[self.feature_space.get(feature)] += 1
                        max_tag_feature = self.create_feature_template(sentence,max_tag,i)
                        for feature in max_tag_feature:
                            if feature in self.feature_space:
                                self.w[self.feature_space.get(feature)] -= 1
                        self.v += self.w
            print("训练集：",end="")
            self.evaluate(self.train_data)
            print("开发集：",end="")
            dev_data_precision = self.evaluate(self.dev_data)
            if dev_data_precision > max_dev_data_precision:
                max_dev_data_precision = dev_data_precision
                max_dev_data_precision_index = iter + 1
                test_data_precision = self.evaluate(self.test_data,"FALSE")
            stoptime = datetime.datetime.now()
            time = stoptime - startime 
            print("本轮用时：%s" % str(time))
            if ((iter+1)-max_dev_data_precision_index) > stop_iterator:     #stop_iterator轮性能没有提升
                break
        print("\n共迭代%d轮" % (iter+1))
        print("开发集第%d轮准确率最高，为：%f" % (max_dev_data_precision_index , max_dev_data_precision))
        print("此时测试集准确率为：%f" % test_data_precision)


    def evaluate(self,sentences,out = 'TRUE'):
        count_right = 0
        count_all = 0
        score_matrix = np.zeros(self.N,dtype="i4")
        for sentence in sentences:
            for i in range(len(sentence)):
                count_all += 1
                right_tag = sentence[i][1]
                for tag_num in range(len(self.tags)):
                    features = self.create_feature_template(sentence,self.tags[tag_num],i)
                    score = 0
                    for feature in features :
                        if feature in self.feature_space:
                            if averaged == "TRUE":
                                score += self.v[self.feature_space.get(feature)]
                            else:
                                score += self.w[self.feature_space.get(feature)]
                    score_matrix[tag_num] = score 
                max_tag = self.tags[np.argmax(score_matrix)]
                if right_tag == max_tag:
                    count_right += 1
        precision = count_right/count_all
        if out == "TRUE":
            print("正确词数：%d\t总词数：%d\t正确率%f" % (count_right,count_all,precision))
        return precision 



if __name__ == "__main__":
    startime = datetime.datetime.now()
    lm = linear_model()
    lm.create_feature_space()
    lm.online_training()
    stoptime = datetime.datetime.now()
    time = stoptime - startime
    print("耗时：" +  str(time))

