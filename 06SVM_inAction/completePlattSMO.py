# -*- coding: UTF-8 -*-  
import svmMLiA
from numpy import *
import matplotlib.pyplot as plt
'''#######********************************
Non-Kernel VErsions below
'''#######********************************
#启发式SMO算法的支持函数
#新建一个类的数据结构，保存当前重要的值
class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag

#格式化计算误差的函数，方便多次调用
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#修改选择第二个变量alphaj的方法
def selectJK(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    # 将误差矩阵每一行第一列置1，以此确定出误差不为0的样本
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    # 获取缓存中Ei不为0的样本对应的alpha列表
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    # 在误差不为0的列表中找出使abs(Ei-Ej)最大的alphaj
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEkK(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        # 否则，就从样本集中随机选取alphaj
        j = svmMLiA.selectJrand(i, oS.m)
        Ej = calcEkK(oS, j)
    return j, Ej

#更新误差矩阵
def updateEkK(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEkK(oS, k)
    oS.eCache[k] = [1, Ek]


#内循环寻找alphaj
def innerLK(i, oS):
    # 计算误差
    Ei = calcEkK(oS, i)
    # 违背kkt条件
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJK(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        # 计算上下界
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        # 计算两个alpha值
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = svmMLiA.clipAlpha(oS.alphas[j],H,L)
        updateEkK(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEkK(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        #
        # 在这两个alpha值情况下，计算对应的b值
        # 注，非线性可分情况，将所有内积项替换为核函数K[i,j]
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        # 如果有alpha对更新
        return 1
    # 否则返回0
    else: return 0

#SMO外循环代码
def smoPK(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin',0)):    #full Platt SMO
    # 保存关键数据
    oS = optStructK(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    # 选取第一个变量alpha的三种情况，从间隔边界上选取或者整个数据集
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        # 没有alpha更新对
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerLK(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            # 统计alphas向量中满足0<alpha<C的alpha列表
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLK(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        # 如果本次循环没有改变的alpha对，将entireSet置为true，
        # 下个循环仍遍历数据集
        elif (alphaPairsChanged == 0): entireSet = True
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas
dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')
b,alphas = smoPK(dataArr,labelArr,0.6,0.001,40)
ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
print('ws:',ws)
#对数据进行分类
datMat = mat(dataArr)
print('计算值:',datMat[0]*mat(ws)+b)
#确认分类结果
print('实际标签值：',labelArr[0])

#绘图
x0 = []; y0 = []
x1 = []; y1 = []
x_sv = []; y_sv = []
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(100):
    if alphas[i] > 0.0:
        x_sv.append(dataArr[i][0])
        y_sv.append(dataArr[i][1])
    elif labelArr[i] > 0.0:
        x0.append(dataArr[i][0])
        y0.append(dataArr[i][1])
    else:
        x1.append(dataArr[i][0])
        y1.append(dataArr[i][1])
ax.scatter(x0,y0,color='r')
ax.scatter(x1,y1,color='g')
ax.scatter(x_sv,y_sv,color='b')
X = linspace(2,6,1000)
Y = [(float(ws[0])*x + float(b))/-float(ws[1]) for x in X]
ax.plot(X, Y)
plt.show()