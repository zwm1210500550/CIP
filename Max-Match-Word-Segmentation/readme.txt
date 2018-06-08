运行环境：python 2.7,第三方模块xpinyin（为了把词典写到文件里按拼音顺序，可有可无)

运行方法：
  python src/CreateDataTxt.py
  python src/CreateWordDict.py
  python src/WordSplit.py
  python src/Evaluate.py

运行结果:
  P=0.99343
  R=0.99066
  F=0.99204

src文件夹:
  1、CreateWordDict.py
  用到了第三方库xpinyin,构建词典dict，以首汉字为key，词列表为value
  按拼音顺序写入word.dict.txt,
  格式为 [首汉字1]:[以该汉字为首的词语1] [词语2]......
         [首汉字2]:[以该汉字为首的词语1] [词语2]......

  2、CreateDataTxt.py
  创建毛文本,将data.conll文件中的格式修改为：每行一句话，词语之间无空格，
  起名为data.txt

  3、WordSplit.py
  读取文件word.dict.txt
  构建词典dict，以首汉字为key，词列表为value
  用正向最大匹配算法分词 结果输出到out.txt
  每行一句话，词之间用空格隔开
  
  4、Evaluate.py
  评价算法的正确率

data文件夹:
  1、data.conll
  初试数据文件

  2、data.txt
  毛文本

  3、word_dict.txt
  词典文件

  4、out.txt
  分词的结果

