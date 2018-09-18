# -*- coding: UTF-8 -*-     
from numpy import *
import logRegres
def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    print('dataMat:');print(dataMat)
    print('labelMat:');print(labelMat)
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

#梯度上升
#dataMathIn是一个2维Numpy数组，每列分别代表每个不同的特征，每行则代表每个训练样本
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    print('dataMatrix:');print(dataMatrix)
    #第二个参数是类别标签，它是一个1*100的行向量。
    #为了便于矩阵运算，需要将该行向量转换为列向量，做法是将原向量转置，如下：
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    print('labelMat:');print(labelMat)
    m, n = shape(dataMatrix)
    alpha = 0.001  #向目标移动的步长
    maxCycles = 500  #迭代次数
    weights = ones((n, 1))
    for k in range(maxCycles):  # heavy on matrix operations
        #h是一个列向量，列向量的元素个数等于样本个数，这里是100
        h = sigmoid(dataMatrix * weights)  # matrix mult
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights #返回训练好的回归系数（最佳参数）


# dataArr,labelMat = loadDataSet()
# weights = gradAscent(dataArr,labelMat)

def colicTest():
    frTrain = open('horseColicTraining.txt')

    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # 训练回归模型
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')  #每行按\t分割
        print('currLine:');print(currLine) #每一行进行读取数据
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
            print('lineArr:');print(lineArr)
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = logRegres.stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    #测试回归模型
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(logRegres.classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

error = colicTest()