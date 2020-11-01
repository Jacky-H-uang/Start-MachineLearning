from numpy import *
import operator
from os import listdir          # get the contents of the directory



# 使用同样的分类器
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
    sortedClassCount = sorted(classCount.items() , key = operator.itemgetter(1) , reverse = True)

    # 返回出现次数最多的 label
    return sortedClassCount[0][0]



# 将图片转化为二进制文本
# 将 32*32 的图片像素矩阵转换为 1 * 1024 的 vector
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])

    return returnVect


# 测试手写学习算法
# Q : Why not autoNum?
# A : because all the number is between 0 and 1 , so autoNum is not neccessary.
def handwritingClassTest():
    hwLables = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)


    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])            # get the class number
        hwLables.append(classNumStr)
        trainingMat[i , :] = img2vector('trainingDigits/%s' %fileNameStr)


    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])            # get the class number
        vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
        classfierResult = classify0(vectorUnderTest , trainingMat , hwLables , 3)

        print("The Classifier came back with : %d , the real answer is : %d" %(classfierResult , classNumStr))
        if classfierResult != classNumStr:
            errorCount += 1.0

    print("\nThe total number of errors is : %d" %errorCount)
    print("\nThe total error rate is : %f" %(errorCount / float(mTest)))




if __name__ == '__main__':
    handwritingClassTest()