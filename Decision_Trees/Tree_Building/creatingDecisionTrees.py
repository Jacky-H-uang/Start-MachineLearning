# -*- coding: UTF-8 -*-

from math import log
import operator


"""
创建测试的数据集合:
    - 是否能离开 surface 生活 ：    1 代表能 ， 0 代表不能
    - 是否有 flippers        ：    1 代表有 ， 0 代表没有
    - 类别(是否为 fish)：           no 代表 no surfacing，yes 代表 flippers。
"""
def createDataSet():
    dataSet = [
               [1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']
              ]
    labels = ['no surfacing', 'flippers']   # 分类属性
    return dataSet, labels         # 返回数据集和分类属性




# 香农熵的计算
# the higher th entropy the more mixed up the data is.
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]                  # 标签类型
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob*log(prob,2)

    return shannonEnt           # 返回经验熵(香农熵)



# 按给定特征划分数据集
# dataSet : 划分的数据集
# axis :    划分的特征
# value :   返回的特征值
def splitDataSet(dataSet , axis , value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1 : ])
            retDataSet.append(reducedFeatVec)

    return retDataSet


# 利用信息增益的结果来计算最好的决策选择
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)               # 计算香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 得到 dataSet 的第 i 个特征
        featList = [example[i] for example in dataSet]

        # set 去重得到所有的特征
        uniqueVals = set(featList)

        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet , i , value)
            prob = len(subDataSet) / float(len(dataSet))            # 计算该特征所占的比列
            newEntropy += prob * calcShannonEnt(subDataSet)         # 计算条件熵
        infoGain = baseEntropy - newEntropy                         # 算出信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature




# 统计 classList 中出现此处最多的元素 (类标签)
def majorityCnt(classList):
    classCount = { }
    for vote in classList:
        if vote not in classCount.keys() :
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.items() , key = operator.itemgetter(1) , reverse = True)

    return sortedClassCount[0][0]



# 创建决策树         (递归地创建决策树)
def createTree(dataSet , labels , featLabels):
    classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList) :     # 如果类别完全相同就停止
        return classList[0]
    if len(dataSet[0]) == 1 :                                # 遍历完所有的特征之后返回出现最多的
        return majorityCnt(classList)

    # 选择最优的 split 的标签
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)

    myTree = {bestFeatLabel : { }}
    del(labels[bestFeat])                                   # 删去最优的标签
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet , bestFeat , value) , subLabels , featLabels)

    return myTree




if __name__ == '__main__':
    dataSet , labels = createDataSet()
    featLables = []
    myTree = createTree(dataSet , labels , featLables)
    print(myTree)