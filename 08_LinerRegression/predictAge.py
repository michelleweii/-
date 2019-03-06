#预测鲍鱼年龄
import regression
from numpy import *
import matplotlib.pyplot as plt
#============预测鲍鱼的年龄================
#预测误差的大小
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

#========训练集上的误差
abX, abY = regression.loadDataSet("abalone.txt")
yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

print("k=0.1,训练集上的误差：",rssError(abY[0:99], yHat01.T))
print("k=1,  训练集上的误差：",rssError(abY[0:99], yHat1.T))
print("k=10, 训练集上的误差：",rssError(abY[0:99], yHat10.T))

#========测试集上的误差
yHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
yHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
yHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)

print("k=0.1,测试集上的误差：",rssError(abY[100:199], yHat01.T))
print("k=1,  测试集上的误差：",rssError(abY[100:199], yHat1.T))
print("k=10, 测试集上的误差：",rssError(abY[100:199], yHat10.T))

ws = regression.standRegres(abX[0:99], abY[0:99])
yHat = mat(abX[100:199])*ws
print("简单线性回归上的误差和：",rssError(abY[100:199],yHat.T.A))