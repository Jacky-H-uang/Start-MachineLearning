"""
This is Bayes of ML in Action
This bayes classifier project to filtering the appropriate and in inappropriate words.
"""

import pandas
import numpy as np



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
将 inputSet 向量化，向量的每个元素为 0 或者 1.

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
Calculating probabilities from word vectors.

Parameters:
    trainMatrix - 训练文档矩阵，即 setOfWords2Vec 返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即 loadDataSet 返回的classVec
"""
def trainNB0(trainMatrix , trainCategory):
    numTrainDocs = len(trainMatrix)                         # 多少篇文章
    numWords = len(trainMatrix[0])                          # 每篇文章多少个字
    pAbusive = sum(trainCategory) / float(numTrainDocs)     # 文章属于 Abusive 的概率
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0

    #
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom

    # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
    return p0Vect , p1Vect , pAbusive