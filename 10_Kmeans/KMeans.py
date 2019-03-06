import numpy as np

# 读取文件内容
def loadDataSet(filename):
    # 构建一个包含许多其他列表的列表
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # fltLine = map(float,curLine)
        dataMat.append(curLine)
    return np.array(dataMat,dtype=np.float64)

# 计算两个向量的欧氏距离
def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.square(vecA-vecB)))

# 一开始应该随机初始化centroids,随机选择k个样本点作为质心
def randCent(dataSet,k):
    m,n = dataSet.shape # m行n列
    centroids = np.zeros((k,n))
    randIndex = np.random.choice(m,k)
    print(randIndex)
    centroids = dataSet[randIndex]
    return centroids

#计算所有样本的均值，改变centroids
def ChangeCent(dataSet,idx,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = np.where(idx[:,0].ravel()==i)
        # axis=0表示沿矩阵的列方向进行均值计算
        centroids[i] = np.mean(dataSet[index],axis=0)
    return centroids


# 为数据集找到最近的质心，直到质心不变为止

def kmeans(x, k):
    # 1.初始化质心
    centroids = randCent(x, k)
    m = x.shape[0]
    # 创建一个矩阵存储每个点的簇分配结果
    idx = np.zeros((x.shape[0], 2))  # 记录x对应的质心的下标 + 平方误差
    clusterchanged = True  # 记录分类变没变化
    while clusterchanged:
        clusterchanged = False
        # 2.为每个样本点分配质心
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distance = distEclud(x[i, :], centroids[j, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 只要有数据发生更新
            if idx[i,0] != minIndex:
                clusterchanged = True
            idx[i,:] = minIndex, minDist ** 2
        # 3.更新质心的位置
        centroids = ChangeCent(x, idx, k)
    return centroids, idx

# idx 是簇分配结果矩阵，包含两列：一列记录簇索引值，第二列存储误差
# 误差指的是：当前点到簇质心的距离

# 计算质心-分配-重新计算，反复迭代，知道所有数据点的簇分配结果不在改变







