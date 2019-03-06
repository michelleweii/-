#岭回归
#岭回归最先用来处理特征数多于样本数的情况，现在也用于在估计中加入偏差，从而得到更好的估计。
#通过引入参数'入'来限制了所有w之和，用过引入该惩罚项，能够减少不重要的参数，for：更好的理解数据
import regression
from numpy import *
import matplotlib.pyplot as plt


def rssError(yArr, yHatArr):  # yArr and yHatArr both need to be arrays
    return ((yArr - yHatArr) ** 2).sum()

#计算回归系数
#实现了给定lambda下的岭回归求解
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam #用lam乘以单位矩阵
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)  #如果矩阵非奇异
    return ws  #计算回归系数

#用于在一组'入'上测试结果
def ridgeTest(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    #数据标准化的过程（为了使用岭回归和缩减技术，首先要对特征做标准化处理）
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar   #（所有特征都减去各自的均值并处以方差）
    numTestPts = 30  #在30个不同的'入'下调用ridgeRegression()函数
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):    #进行numTestPts次计算岭回归，每次的系数向量都放到wMat的一行中。
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T #将所有的回归系数输出到一个矩阵并返回
    return wMat   #这样就得到了30个不同'入'所对应的回归系数


# abX,abY = regression.loadDataSet('abalone.txt')
# ridgeWeights = ridgeTest(abX,abY)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ridgeWeights)   #ridgeWeights,为30*8的矩阵，对矩阵画图，则以每列为一个根线，为纵坐标，
#                           横坐标为range(shape(ridgeWeights)[0])也即从0到29,第一行的横坐标为0,最后一行的行坐标为29
# plt.show()


#前向逐步线性回归
#一开始，所有的权重都社设为1，然后每一步所做的决策时对某个权重增加或减少一个很小的值。
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0) #按照行取平均，每行平均
    yMat = yMat - yMean     #对y进行标准化   #can also regularize ys but will get smaller coef
    xMat = regression.regularize(xMat)
    m,n=shape(xMat)
    #returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:    #分别计算增加或减少该特征对误差的影响
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:   #如果误差error小于当前最小误差lowesterror：设置wbest等于当前的w
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()

    #     returnMat[i,:]=ws.T
    # return returnMat

# xArr,yArr = regression.loadDataSet('abalone.txt')
# # print('逐步向前回归结果：');print(regression.stageWise(xArr,yArr,0.001,5000))

# xMat = mat(xArr)
# yMat = mat(yArr).T
# xMat = regression.regularize(xMat)
# yM = mean(yMat,0)
# yMat = yMat - yM
# weights = regression.standRegres(xMat,yMat.T)
# print('最小二乘法：'); print(weights.T)
# #OUTPUT:
# #[[ 0.0430442  -0.02274163  0.13214087  0.02075182  2.22403814 -0.99895312
# #  -0.11725427  0.16622915]]