import numpy as np
import operator         # operator 运算模块



def createDataSet():
    group = np.array([1.0 , 1.1] , [1.0 , 1.0] , [0 , 0] , [0 , 0.1])
    labels = ['A' , 'A' , 'B' , 'B']

    return group , labels


"""
对未知类别属性的数据集中的每个点依次执行以下操作： 
(1) 计算已知类别数据集中的点与当前点之间的距离； 
(2) 按照距离递增次序排序； 
(3) 选取与当前点距离最小的k个点； 
(4) 确定前k个点所在类别的出现频率； 
(5) 返回前k个点出现频率最高的类别作为当前点的预测分类。
"""

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
    diffMat = np.tile(inX , (dataSize , 1)) - dataSet              # inX 重复 dataSize 行 ， 1 列
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