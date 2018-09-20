# -*- coding: UTF-8 -*-   
# 使用朴素贝叶斯过滤垃圾邮件  (使用朴素贝叶斯对文档进行分类)
import bayes
import re
import jieba
#数据集。有两个文件夹，一个叫“ham”，包含了25封平常的邮件；另一个叫“spam”，包含了各种25封垃圾邮件。

#拿到文本后，首先进行分词（切分文本），形成词向量---切分文本
mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
regEx = re.compile('\W*')
listOfTokens = regEx.split(mySent)
print (listOfTokens)
#没有效果
#[tok.lower() for tok in listOfTokens if len(tok)>0]
#print listOfTokens
# emailText = open('email/ham/6.txt').read()
# listOfTokens = regEx.split(emailText)

bayes.spamTest()
bayes.spamTest()
# 首先，读取一共50封邮件，每一封拆成一个个单词放入一个wordList，再把wordList放入docList。前25封标为垃圾邮件，后25封标为正常邮件。
# 然后，创建词汇表（用的是createVocabList函数）。再把50条数据随机选10条当做测试集，40条当做训练集。
# 然后，拿训练集进行训练。训练函数是trainNB0。
# 要注意的是转换成词向量的时候，用的是bagOfWords2VecMN函数，即词袋模型。这个模型统计的是单词的出现次数。
# 最后用测试集进行测试，输出错判的数据，以及最终的错误率。