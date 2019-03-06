import KMeans
from numpy import *

dataMat = KMeans.loadDataSet('testSet.txt')
# print(dataMat)
print(min(dataMat[:,0]))
print(KMeans.randCent(dataMat,2)) # 80*2

myCent, clustAssing = KMeans.kmeans(dataMat,4)

print("myCent:{}".format(myCent))