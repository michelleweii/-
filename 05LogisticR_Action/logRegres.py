# -*- coding: UTF-8 -*-     
from numpy import *

#打开文本，每行前两个值分别是x1和x2（特征），第三个值是数据对应的类别标签
#为了计算方便，将x0的值设为1.0
def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

#梯度上升 #定义求解最佳回归系数
#dataMathIn是一个2维Numpy数组，每列分别代表每个不同的特征，每行则代表每个训练样本
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix #将数组转为矩阵
    #第二个参数是类别标签，它是一个1*100的行向量。
    #为了便于矩阵运算，需要将该行向量转换为列向量，做法是将原向量转置，如下：
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    m, n = shape(dataMatrix) #返回矩阵的行和列
    alpha = 0.001  #向目标移动的步长
    maxCycles = 500  #迭代次数
    weights = ones((n, 1)) #初始化最佳回归系数
    for k in range(maxCycles):  # heavy on matrix operations
        #h是一个列向量，列向量的元素个数等于样本个数，这里是100
        # 引用原书的代码，求梯度
        h = sigmoid(dataMatrix * weights)  # matrix mult
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights #返回训练好的回归系数

#画出数据集和画出Logisitic回归最佳拟合直线的函数
#分析数据，画出决策边界
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat) #将矩阵转化为数组
    n = shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()



def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#随机梯度上升
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(0,numIter):
        dataIndex = list(range(m)) # 新添加，因为 'range' object doesn't support item deletion
        for i in range(0,m):
            # alpha值每次迭代时都进行调整
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            randIndex = int(random.uniform(0, len(dataIndex)))  # #随机选取更新  go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#二分类问题进行分类
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

#本实例采用的处理缺失值得办法是，过滤掉label为空的数据，同时将数据中除了类别标签之外的所有空值填充0，且所用数据为处理后的数据.
#训练和测试
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # 训练回归模型
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    #测试回归模型
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate
# colicTest()：打开测试集和训练集，并对数据进行格式化处理，并调用stocGradAscent1()函数进行回归系数的计算，
# 设置迭代次数为500，这里可以自行设定，*在系数计算完之后，导入测试集并计算分类错误率。*
# 整体看来，colictest()函数具有完全独立的功能，多次运行的结果可能稍有不同，这是因为其中包含有随机的成分在里边，
# 如果计算的回归系数是完全收敛，那么结果才是确定的


# multiTest()：调用 colicTest() 10次并求结果的平均值
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))

