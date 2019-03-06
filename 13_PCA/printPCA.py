#!/usr/bin/env python
# coding=utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim="\t"):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]  # 对每行数据的两个数字都变为float型
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):  # dataMat为1000×2的矩阵，
    meanVals = mean(dataMat, axis=0)  # numpy中的mat类型有算均值mean的方法,并且对每列进行计算，得到2*1的矩阵
    meanRemoved = dataMat - meanVals  # 平移到以0为中心的地方，避免均值影响。1000*2
    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差，维度为2,得到2×2的协方差矩阵
    eigVals, eigVects = linalg.eig(mat(covMat))  # 计算特征值及特征向量，2×1,2×2,得到两个特征值，两列特征向量，向量长度为2
    eigValInd = argsort(eigVals)  # 默认快排，从小到大进行排序
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 得到topNfeat个特征值，大的特征值排在前面，若topNfeat为1。，则取1个特征值
    redEigVects = eigVects[:, eigValInd]  # 将topNfeat个最大的特征值对应的那列向量取出来，若topNfeat为1,
    lowDDataMat = meanRemoved * redEigVects  # 1000×2 * 2×topNfeat得到1000×topNfeat的低维空间的数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 1000×topNfeat * topNfeat*2 +1000×2重构数据
    return lowDDataMat, reconMat


dataMat = loadDataSet("testSet.txt")
lowDMat, reconMat = pca(dataMat, 1)
print shape(dataMat)
print shape(lowDMat)
print shape(reconMat)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker="^", s=90)
ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker="o", s=50, c="red")
plt.show()

'''
lowDMat矩阵为降维后的矩阵，大小为1000*topNfeat。
reconMat矩阵为重构数据的矩阵，也即将低维的lowDMat矩阵转到原来数据上，从上图看，lowDMat为一维坐标上的点（没画出来），其重构到数据上便是红色圆圈的点，即在二维空间里的点（在一条直线上），及reconMat矩阵。
若是topNfeat为2,即没有剔除任何特征，重构之后的数据会和原始的数据重合。
'''