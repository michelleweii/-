import regTrees
from numpy import *


testMat = mat(eye(4))
print('testMat:',testMat)
#按照指定的列的某个值来切分该矩阵
mat0,mat1 = regTrees.binSplitDataSet(testMat,1,0.5)
print('mat0:',mat0)
print('mat1:',mat1)
