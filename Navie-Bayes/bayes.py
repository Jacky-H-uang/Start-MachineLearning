"""
This is Bayes of ML in Action
This bayes classifier project to filtering the appropriate and in inappropriate words.
"""



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


    # 1 means Abusive and 0 means Not
    classVec = [0,1,0,1,0,1]

    return postingList , classVec



# 整理词条成为词汇表
def createVocabList(dataSet):
    vocabSet = set([])

    for document in dataSet:
        # 取并集
        vocabSet = vocabSet | set(document)

    return list(vocabSet)



# 将 inpuSet 向量化，向量的每个元素为 0 或者 1
def setIfWords2Vec(vocabList,inputSet):
    # 创建一个全 0 的向量
    returnVec = [0 for i in len(inputSet)]

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else : print("this word : %s isn't in vocabList yet!" %word)

    # 返回输入的 inputSet 的向量化列表
    return returnVec



