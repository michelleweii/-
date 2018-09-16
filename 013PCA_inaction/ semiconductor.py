# -*- coding: UTF-8 -*-  
from numpy import *
import matplotlib.pyplot as plt
import pca
#半导体
def replaceNanWithMean():
    datMat = pca.loadDataSet('/Users/weiwenjing/Desktop/M.L./machinelearninginaction/Ch13/secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

dataMat = replaceNanWithMean()
meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals  # remove mean
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
#print eigVals
print sum(eigVals)*0.9
print sum(eigVals[:6])
plt.plot(eigVals[:20])#对前20个画图观察
plt.title('In Action:')
plt.show()


from sklearn import decomposition
pca_sklean = decomposition.PCA()
pca_sklean.fit(replaceNanWithMean())
main_var = pca_sklean.explained_variance_
print sum(main_var)*0.9
print sum(main_var[:6])
plt.plot(main_var[:20])
plt.title('SKlearn：')
plt.show()


'''
前两行数字与后两行数字
显示我们编写的程序得到主成分的90%为81131452.777与sklearn计算得出的81079677.7592相差不多。
前6个也依然差不多。
'''