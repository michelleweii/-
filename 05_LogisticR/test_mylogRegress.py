# -*- coding: UTF-8 -*-   
# Logistic回归的目的是寻找一个非线性函数sigmoid的最佳拟合参数，求解过程可以由最优化算法来完成。
# 在最优化算法中，最常用的就是梯度上升算法，而梯度上升算法又可以简化为随机梯度上升算法。

# 随机梯度上升算法和梯度上升算法的效果相当，但占用更少的计算资源。随机梯度上升算法是一种在线算法，
# 可以在数据到来时就完成参数的更新，而不需要重新读取整个数据集来进行批处理运算。
import logRegres
from numpy import *
import math
dataArr,labelMat = logRegres.loadDataSet()
# weights = logRegres.gradAscent(dataArr,labelMat) #梯度上升——计算回归系数
# print(logRegres.gradAscent(dataArr,labelMat))
#output:
# [[ 4.12414349]
#  [ 0.48007329]
#  [-0.6168482 ]]
# logRegres.plotBestFit(weights.getA()) #梯度上升,计算量大  getA()s是什么？？？
#
#
#
# weights = logRegres.stocGradAscent0(array(dataArr),labelMat)  #为什么这里用array()？？？？？？？
# logRegres.plotBestFit(weights)   #随机梯度上升0,效果差
#
#
#

# weights = logRegres.stocGradAscent1(array(dataArr),labelMat)
# logRegres.plotBestFit(weights)
#
#
# 以下代码把测试集上的每个特征向量乘以最优化算法得来的回归系数w，再将该乘积结果求和，最后输入到Sigmoid函数，
# 如果对应的Sigmoid函数值大于0.5，则将该样本的类别判定为1，否则判定为0；最后，统计判定结果与实际结果的误差，
# 由于误差具有不确定性，程序最后使用了10次分类结果求平均的方法，得出算法的平均分类错误率。
logRegres.multiTest()