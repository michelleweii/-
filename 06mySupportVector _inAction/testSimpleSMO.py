import svmMLiA
from numpy import *
import matplotlib.pyplot as plt
dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')
# print(labelArr) #[-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0,...]
b,alphas = svmMLiA.smoSimple(dataArr,labelArr,0.6,0.001,40)#数据集，类别标签，常数C，容错率，退出前最大的循环次数
print('martix b:',b)
am,an = shape(alphas)
print('am:',am,',an:',an)
print('martix alphas:',alphas[alphas>0])
print('支持向量的个数：',shape(alphas[alphas>0]))
for i in range(100):
    if alphas[i]>0.0: print('这些点是支持向量：',dataArr[i],labelArr[i])
#output:
# martix b: [[-3.79474649]]
# am: 100 ,an: 1
# martix alphas: [[ 0.11482438  0.1309357   0.11348042  0.08434201  0.27489848]]
# 支持向量的个数： (1, 5)
# 这些点是支持向量： [4.658191, 3.507396] -1.0
# 这些点是支持向量： [3.457096, -0.082216] -1.0
# 这些点是支持向量： [2.893743, -1.643468] -1.0
# 这些点是支持向量： [5.286862, -2.358286] 1.0
# 这些点是支持向量： [6.080573, 0.418886] 1.0

ws = svmMLiA.calcWs(alphas, dataArr, labelArr)

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