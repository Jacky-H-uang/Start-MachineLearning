from numpy import *
import operator         # operator 运算模块
from os import listdir
import matplotlib
import matplotlib.pyplot as plt



def createDataSet():
    group = np.array([1.0 , 1.1] , [1.0 , 1.0] , [0 , 0] , [0 , 0.1])
    labels = ['A' , 'A' , 'B' , 'B']

    return group , labels


# inX      --(输入向量)
# dataSet  --(训练样本集)
# labels   --(标签向量)
# k        --(选择最近邻居的数目)
def classify0(inX , dataSet , labels , k):
    """ Three Steps """

    """
    1. 距离计算         (采用两点间的距离公式)
    """
    dataSize = dataSet.shape[0]                                 # 记录 dataSet 的行数
    diffMat = tile(inX , (dataSize , 1)) - dataSet              # inX 重复 dataSize 行 ， 1 列
    sqDifMat = diffMat ** 2
    sqDistances = sqDifMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()                    # argsort() 排列索引从小到大

    """
    2. 选择最小的 k 个点 (即选择距离最近的 k 个点)
    """
    classCount = { }
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel , 0) + 1

    """
    3. 排序   --(itemgetter(1) 为排序目标为类型出现的次数)
    """
    sortedClassCount = sorted(classCount.iteritem() , key = operator.itemgetter(1) , reverse = true)

    # 返回出现次数最多的 label
    return sortedClassCount[0][0]



# 将文本记录转换为 NumPy 的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()                # get lines of file
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))

    # 解析文件到列表
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index , :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat , classLabelVector



# 归一化特征值 : 公式 : newValues = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals , (m,1))           # 减去最小特性值
    normDataSet = normDataSet / tile(ranges , (m,1))        # 特征值相除

    return normDataSet , ranges , minVals