# -*- coding: UTF-8 -*-  
import regression
from numpy import *
import matplotlib.pyplot as plt
# xArr,yArr = regression.loadDataSet('ex0.txt')
# # print xArr
# # print yArr
# #首先看前两条数据
# # print xArr[0:2]
# #第一个总是X0=1.0，第二是x1
#
# # ws = regression.standRegres(xArr,yArr)
# # print('ws:');print(ws)
#
# #
# xMat = mat(xArr) # # print xMat
# yMat = mat(yArr) # print yMat
# # yHat = xMat * ws
# fig = plt.figure()  # 创建绘图对象
# ax = fig.add_subplot(111)  # 111表示将画布划分为1行2列选择使用从上到下第一块
# # scatter绘制散点图
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
# # 复制，排序
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = xCopy * ws
# # plot画线
# ax.plot(xCopy[:,1], yHat)
# # plt.show()
#
# #相关系数
# #print('corrcoef:');print(corrcoef(yHat.T,yMat))   #预测值和真实值得匹配程度
# #corrcoef:
# # [[ 1.          0.13653777]
# # [ 0.13653777  1.        ]]
# #end：Liner Regression

xArr,yArr = regression.loadDataSet('ex0.txt')
#对单点进行估计,输出预测值
print('xArr[0]:');print(xArr[0])
#  xArr[0]:
# [1.0, 0.067732]
print(yArr[0])  #output: 3.176513
print(regression.lwlr(xArr[0],xArr,yArr,1.0))  #output:  martix([[ 3.12204471]])
print(regression.lwlr(xArr[0],xArr,yArr,0.001))     #output:  martix([[ 3.20175729]])

#为了得到数据集里所有点的估计，可以调用LwlrTest()函数：
yHat = regression.lwlrTest(xArr,xArr,yArr,0.003)
print('all points about yHat:');    print(yHat)

#查看拟合效果
xMat = mat(xArr)   #xArr是什么？
srtInd = xMat[:,1].argsort(0) #对xArr排序
xSort = xMat[srtInd][:,0,:]
#绘图：
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
plt.show()