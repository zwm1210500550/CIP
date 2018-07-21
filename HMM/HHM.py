#!\usr\bin\python
# _*_ coding=UTF-8_*_
import codecs
import math

def get_source_and_test(filename, list_term):
    # list_term 是存储每句话中词的列表
    alpha = 0.01
    fs = codecs.open(filename, 'r', 'UTF-8')
    str_line = fs.readline()
    vocabulary = []  # 词汇表
    vocabulary_dic = {'*': 1}  # 词性表
    vocabulary_dic['EOS'] = 0
    list_data = []  # 二元字符串(词-词)的列表
    list_count = []  # 二元字符串频数的列表
    part_of_speech_term_list = []  # 词性-词的列表
    part_of_speech_term_count = []  # 词性-词列表对应的频数
    part_of_speech_dic = {}  # 词性的数量的字典
    # temp_list = ['<BOS>'] 词汇转移
    temp_list = ['*']
    while str_line:
        if str_line == '\n':
            temp_list.append('<EOS>')
            list_term.append(temp_list)
            vocabulary_dic['*'] += 1
            temp_list = ['*']
        else:
            cur_term_list = str_line.split('\t')
            temp_term_ = cur_term_list[1]
              # 词性转移
            temp_part_of_speech = cur_term_list[3]
            part_of_speech_term = [temp_part_of_speech, temp_term_]
            if temp_term_ not in vocabulary:
                vocabulary.append(temp_term_)
            if temp_part_of_speech in vocabulary_dic:
                vocabulary_dic[temp_part_of_speech] += 1
            else:
                vocabulary_dic[temp_part_of_speech] = 1
            if part_of_speech_term not in part_of_speech_term_list:
                part_of_speech_term_list.append(part_of_speech_term)
                part_of_speech_term_count.append(1)
            else:
                ind = part_of_speech_term_list.index(part_of_speech_term)
                part_of_speech_term_count[ind] += 1
            if temp_part_of_speech in part_of_speech_dic:
                part_of_speech_dic[temp_part_of_speech] += 1
            else:
                part_of_speech_dic[temp_part_of_speech] = 1
            temp_list.append(temp_part_of_speech)
        str_line = fs.readline()

    if len(temp_list) > 1:
        temp_list.append('<EOS>')
        list_term.append(temp_list)

    for list_line in list_term:
        for i in range(0, len(list_line) - 1):
            temp_data = list_line[i:i + 2]
            # print temp_data[0].encode('utf-8'),temp_data[1].encode('utf-8'),temp_data[2].encode('utf-8')
            if temp_data in list_data:
                index = list_data.index(temp_data)
                list_count[index] += 1
            else:
                list_data.append(temp_data)
                list_count.append(1)

    count_vocabulary = len(vocabulary)  # |V|语料的词汇量
    print count_vocabulary, len(part_of_speech_dic)

    list_probability = []
    tagging('dev.conll', list_data, list_count, vocabulary_dic, alpha, len(vocabulary),
            part_of_speech_term_list ,part_of_speech_term_count)

def smooth(data, count, dic, alpha, V, elem):
    '数据加alpha平滑'
    # data :词性-词性（词汇）转移表 count：转移表中各项的频数 dic： 词汇（词性）-频数的字典 elem: 转移的双方[前者，后者]
    a = 0.0
    b = 0.0
    if elem in data:
        a = count[data.index(elem)]
    if elem[0] in dic:
        b = dic[elem[0]]
    return float((a + alpha)) / (b + alpha * V)

def trans_pos(data, count, dic, elem):
    a = 0.0
    b = 0.0
    if elem in data:
        a = count[data.index(elem)]
    if elem[0] in dic:
        b = dic[elem[0]]
        return a*1.0/b
    return 0.0


def viterbi(sentence, list_data, list_count, vocabulary_dic, alpha, V,part_of_speech_term_list,part_of_speech_term_count):
    prob = [{'*': 0.0}]  # 概率序列
    pos_list = []  # 路径序列
    cur_prob = 0
    cur_prob_dic = {}
    pof_sentence=[]  # 词性标注结果
    for i in range(0, len(sentence)):
        pos_list.append({})
        for v in vocabulary_dic.keys():
            for w in prob[i].keys():
                q=smooth(list_data, list_count, vocabulary_dic,alpha, V, [w, v])
                e = smooth(part_of_speech_term_list, part_of_speech_term_count, vocabulary_dic, alpha, V,[v, sentence[i]])
                temp_prob = prob[i][w] - math.log10(q) - math.log(e)
                if temp_prob < cur_prob or cur_prob==0:
                    cur_prob = temp_prob
                    pos_list[i][v] = w
                    cur_prob_dic[v] = cur_prob
            cur_prob = 0
        prob.append(cur_prob_dic)
        cur_prob_dic={}

    cur_w = ''
    n=len(sentence)
    for w in prob[n].keys():
        q = smooth(list_data, list_count, vocabulary_dic,alpha, V, [w, '<EOS>'])
        temp_prob = prob[n][w] - math.log10(q)
        if temp_prob < cur_prob or cur_prob==0:
            cur_w=w
            cur_prob=temp_prob
    # 从cur_w开始回溯
    pof_sentence.insert(0, cur_w)
    for i in range(len(sentence)-1,0,-1):
        cur_w2=pos_list[i][cur_w]
        pof_sentence.insert(0,cur_w2)
        cur_w=cur_w2
    return pof_sentence

def tagging(filename, list_data, list_count, vocabulary_dic, alpha, V,part_of_speech_term_list\
            ,part_of_speech_term_count):
    fh=codecs.open(filename,'r','utf-8')
    sentence_list=[]  # 词组列表
    sentence=[]  # 临时词组
    pos_list=[]  # 词性列表
    right=0
    total=0
    for str_line in fh:
        if len(str_line)<=1:
            r_pos = viterbi(sentence, list_data, list_count, vocabulary_dic, alpha, V,
                                   part_of_speech_term_list \
                                   , part_of_speech_term_count)
            l=len(r_pos)
            total+=l
            for i in range(0,l):
                if r_pos[i]==pos_list[i]:
                    right+=1
            sentence_list.append(sentence)
            print right, total, float(right) / total
            sentence=[]
            pos_list=[]
        else:
            str_str=str_line.split('\t')[1]
            str_pos=str_line.split('\t')[3]
            try:
                pos_list.append(str_pos)
            except BaseException:
                print pos_list
            sentence.append(str_str)

    with open('result.txt','w+') as ff:
        ff.write('词性标注准确率%d/%d=%f'%(right,total,float(right)/total))
    print right, total, float(right)/total

list_term = []
get_source_and_test('train.conll', list_term)
