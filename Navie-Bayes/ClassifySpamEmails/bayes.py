
"""
This is the famous use of navie bayes : email-spam filtering
"""

import re
import random
import numpy as np




# 解析文本数据
def textParser(bigString):
    # 头部添加一个 r 表示取原生字符 , 不用再添加任何的转义 , 即避免了转义的时候的冲突
    listOfTokens = re.split(r'\W+' , bigString)

    # 除了单个字母 , 其他字母全部变成小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]




# 将所有邮件中的单词整理成不重复的词汇表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    return list(vocabSet)



# 根据词汇表将输入 inputSet 向量化 , 向量化元素 0 或 1
def setOfWords2Vec(vocabList , inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else : print("the word ： %s is not in my Vocabulary!" %word)

    return returnVec




# 根据词汇表 , 构建词袋模型
def bagOfWords2VecMN(vocabList , inputSet):
    returnVec = [0]*len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1

    return returnVec


# navie-bayes 分类器的训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive





# navie-bayes 分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0




# 测试 navie-bayes 分类器
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        # 读取 spam 里面的 25 篇邮件然后全部解析之后放入列表中
        wordList = textParser(open('spam/%d.txt' %i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)           # 拓展列表
        classList.append(1)

        wordList = textParser(open('ham/%d.txt' %i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10) :
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V , p1V , pSpam = trainNB0(np.array(trainMat) , np.array(trainClasses))

    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList , docList[docIndex])
        if classifyNB(np.array(wordVector) , p0V , p1V , pSpam) != classList[docIndex]:
            errorCount += 1
            print("Error test set : " , docList[docIndex])
    print('the error rate is : ' , float(errorCount)/len(testSet) * 100)




if __name__ == '__main__':
    spamTest()