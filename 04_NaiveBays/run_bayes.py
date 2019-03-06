# -*- coding: UTF-8 -*-     
import bayes
#从文本中构建词向量
listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print myVocabList
print bayes.setOfWords2Vec(myVocabList,listOPosts[0])
#为什么会输出这个么多的数据，listOPosts[0]只有7个词？？
#A:是按照词汇表的数据标注的，生成的不重复的词汇表有32个词，在listOPosts[0]文档中里有的词，在相应的词汇表位置标注1.
#myVocabList中索引为2的元素是help，help在第一篇文档中出现(索引2是1)，在第4篇文档中没有出现(索引2是0)。
#myVocablist里包含的是postingList所有的词汇（所有文档），函数setOfWords2Vec（词汇表，某一个文档）。
print bayes.setOfWords2Vec(myVocabList,listOPosts[3])

#从词向量计算概率
trainMat =[]
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))
print "trainMat:"
#trainMat中存放的是：postingList中所有的文档，每一个文档有一个词向量。此mar是6*32的矩阵。
#trainMat是文档矩阵
print trainMat
#listClasses是每一篇文档的类别标签。有6个数据77777777777777777777777777777777777777777777777777
p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)
print "文档属于侮辱类的概率为："
#pAB含义是任意文档属于侮辱性的概率
print pAb
#p0V:词汇表中有32个词，第一个词cute出现在正常文档中的概率，第二个词love出现在正常文档中的概率...
print "p0V:词汇表中有32个词，第一个词cute出现在正常文档中的概率，第二个词love出现在正常文档中的概率..."
print p0V
#p1V:词汇表中有32个词，第一个词cute出现在垃圾文档中的概率，第二个词love出现在垃圾文档中的概率...
print p1V

print "******************************* Step 2 ***************************************************"
#将所有词的出现数初始化为1，将分母初始化为2——避免概率为0
#为了避免下溢，对乘积取自然对数
bayes.testingNB()
