"""
This is Bayes of ML in Action
This bayes classifier project to filtering the appropriate and in inappropriate words.
"""

import pandas
import numpy as np
from math import *





# 创建实验的 DataSet
def loadDataSet():
    # 切分的词条
    postingList=[
                    ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                ]


    # 1 means Abusive and 0 means Not Abusive
    # 这个 classVec 说明  : 第 0 篇，第 2 篇，第 4 篇文章是 NotAbusive
    #                   : 第 1 篇，第 3 篇，第 5 篇文章是 Aubsive
    classVec = [0,1,0,1,0,1]

    return postingList , classVec






"""
整理词条成为词汇表.

Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(dataSet):
    vocabSet = set([])

    for document in dataSet:
        # 取并集
        vocabSet = vocabSet | set(document)

    return list(vocabSet)






"""
将 inputSet 向量化，向量的每个元素为 0(出现) 或者 1(未出现).

Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型
"""
def setOfWords2Vec(vocabList,inputSet):
    # 创建一个全 0 的向量
    returnVec = [0] * len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else : print("this word : %s isn't in vocabList yet!" %word)

    return returnVec



"""
Improving of function : serOfWords2Vec()
The bag-of-words document model.
"""
def bagOfWords2VecMN(vocabList , inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1

    return returnVec





"""
Calculating probabilities from word vectors.

Parameters:
    trainMatrix - 训练文档矩阵，即 setOfWords2Vec 返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即 loadDataSet 返回的classVec
"""
def trainNB0(trainMatrix , trainCategory):
    numTrainDocs = len(trainMatrix)                         # 多少篇文章
    numWords = len(trainMatrix[0])                          # 每篇文章多少个字
    pAbusive = sum(trainCategory) / float(numTrainDocs)     # 文章属于 Abusive 的概率

    # Numerator         分子
    # Denominator       分母
    p0Num = np.zeros(numWords)              # NotAbusive 文中中 词汇的矩阵
    p1Num = np.zeros(numWords)              # Abusive 文章中 词汇的矩阵
    p0Denom = 2.0                           # Abusive 文章中的 word 的总数量
    p1Denom = 2.0                           # NotAbusive 文章中的 word 的总数量

    for i in range(numTrainDocs):
        # 如果扫描的这篇文章是 Abusive
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]                         # 当这篇文章是 Abusive 的时候就统计过他出现过的词汇数量矩阵
            p1Denom += sum(trainMatrix[i])                  # 统计这篇文章里面的词汇有多少个

        # 如果扫描的这篇文章是 NotAbusive
        else:
            p0Num += trainMatrix[i]                         # 当这篇文章是 NotAbusive 的时候就统计过他出现过的词汇数量矩阵
            p0Denom += sum(trainMatrix[i])                  # 统计这篇文章里面的词汇有多少个

    # 防止 underflow
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)

    # 返回属于 "非侮辱类" 的条件概率数组
    # 属于 "侮辱类" 的条件概率数组
    # 文档属于 "侮辱类" 的概率
    return p0Vect , p1Vect , pAbusive





"""
Classify function.

Parameters:
    vec2Classify    -  待分类的词条数组              (要求被向量化)
    p0Vec           -  非侮辱类的条件概率数组
    p1Vec           -  侮辱类的条件概率数组
    pClass1         -  文档属于侮辱类的概率
"""
def classifyNB(vec2Classify , p0Vec , p1Vec , pClass1):
    # 将其 logarithm 避免 underflow 成 0.0
    # 判断一篇文章是否是 Abusive 的指标即比较每个单词的 abusive 和 not abusive 的概率
    # 这就是朴素的贝叶斯公式的推断    P(X|a) * P(a) = { P(x1|a) * P(x2|a) * ... * P(xn|a) } * P(a)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)

    print("p0",p0)
    print("p1",p1)

    if p1 > p0:
        return 1
    else:
        return 0





"""
Testing the Navie-Bayes
"""
def testingNB():
    listOPosts , listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []

    # 将输入 vector 化并加入数组
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList , postinDoc))

    p0V , p1V , pAb = trainNB0(np.array(trainMat),np.array(listClasses))

    # 测试数据 1
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList , testEntry))
    if classifyNB(thisDoc , p0V , p1V , pAb):
        print(testEntry , 'classified as : Abusive')
    else:
        print(testEntry , 'classified as : Not Abusive')

    # 测试数据 2
    testEntry = ['stupid' , 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList , testEntry))
    if classifyNB(thisDoc , p0V , p1V ,pAb):
        print(testEntry , 'classified as : Abusive')
    else:
        print(testEntry , 'classified as : Not Abusive')






# 测试
if __name__ == '__main__':
    testingNB()