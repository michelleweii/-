# -*- coding: UTF-8 -*-     
import bayes
import random
import numpy as np
import feedparser

# 先把文本拆成词，然后构建训练数据集、测试数据及，最后训练、测试，看错误率。
def calcMostFreq(vocabList, fullText):  #从fullText中找出最高频的前30个单词
    import operator
    freqDict = {}
    for token in vocabList:  #统计词汇表里所有单词的出现次数
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]  #返回字典


def localWords(feed1, feed0):  # 两份RSS文件分别经feedparser解析，得到2个字典
    docList = []  # 一条条帖子组成的List, 帖子拆成了单词
    classList = []  # 标签列表
    fullText = []  # 所有帖子的所有单词组成的List
    # entries条目包含多个帖子，miNLen记录帖子数少的数目，怕越界
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = bayes.textParse(feed1['entries'][i]['summary'])  # 取出帖子内容，并拆成词
        docList.append(wordList)  # ['12','34'].append(['56','78']) ==> [ ['12','34'], ['56','78'] ]
        fullText.extend(wordList)  # ['12','34'].extend(['56','78']) ==> ['12','34','56','78']
        classList.append(1)  # 纽约的标签是1
        wordList = bayes.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)  # 旧金山的标签是0

    vocabList = bayes.createVocabList(docList)  # 创建词汇表
    # 从fulltext中找出最高频的30个单词，并从vocabList中去除它们
    top30Words = calcMostFreq(vocabList, fullText)
    for (word, count) in top30Words:
        if word in vocabList:
            vocabList.remove(word)

    trainingSet = range(2 * minLen);
    testSet = []  # 创建训练集、测试集
    for i in range(minLen / 10):  # 随机选取10%的数据，建立测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))  # 将训练集中的每一条数据，转化为词向量
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNB0(np.array(trainMat), np.array(trainClasses))  # 开始训练

    # 用测试数据，测试分类器的准确性
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if bayes.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)
    return vocabList, p0V, p1V
#把纽约和旧金山的RSS文件解析以后，调用locaWords()函数里面，构建分类器，获得每个词出现的概率，p1V和p0V。
# 然后只留下概率（的对数）>-6.0的单词，排序后从高到低输出。
def getTopWords(feeds_ny, feeds_sf):
    vocabList,p0V,p1V=localWords(feeds_ny, feeds_sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for (word, prob) in sortedNY:
        print word
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for (word, prob) in sortedSF:
        print word

feeds_ny = feedparser.parse('https://newyork.craigslist.org/search/stp?format=rss')  #纽约
feeds_sf = feedparser.parse('https://sfbay.craigslist.org/search/stp?format=rss')  #旧金山
print len(feeds_ny['entries']), len(feeds_sf['entries'])
localWords(feeds_ny, feeds_sf)
getTopWords(feeds_ny, feeds_sf)
# feedparser.parse返回的是个字典。rss_doc['entries']是所有帖子，它是个List，里面每一条entry是一条帖子。
# 每个entry又是个字典，entry['summary_detail']是帖子详情， 它也是个字典。就是帖子的内容了。
# 可以看到里面有25条帖子，然后把帖子的内容打印出来了...
print 'small test:'
rss_doc = feedparser.parse('https://newyork.craigslist.org/search/w4m?format=rss') #
print len(rss_doc['entries'])
for entry in rss_doc['entries']:
    print entry['summary_detail']['value']