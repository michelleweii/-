# -*- coding: UTF-8 -*-     
'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
import numpy as np

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # 对每行数据的两个数字都变为float型
    datArr = [map(float,line) for line in stringArr]
    return np.mat(datArr)

def pca(dataMat, topNfeat=9999999):  #dataMat为1000×2的矩阵
    # numpy中的mat类型有算均值mean的方法,并且对每列进行计算，得到2*1的矩阵
    meanVals = np.mean(dataMat, axis=0)
    # 平移到以0为中心的地方，避免均值影响。1000*2
    meanRemoved = dataMat - meanVals #remove mean
    # 计算协方差，维度为2,得到2×2的协方差矩阵
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算特征值及特征向量，2×1,2×2,得到两个特征值，两列特征向量，向量长度为2
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    print(u'eigVals=',eigVals,u'eigVects=',eigVects)

    eigValInd = np.argsort(eigVals)            #sort, sort goes smallest to largest
    print('eigValInd after argsort:',eigValInd)

    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    print('eigValInd after [:-(topNfeat+1):-1]',eigValInd)

    #
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    print('redEigVects',redEigVects)

    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    # print('lowDDataMat',lowDDataMat)

    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # print('reconMat',reconMat)
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    datMat = loadDataSet('/Users/weiwenjing/Desktop/M.L./machinelearninginaction/Ch13/secom.data', ' ')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat


