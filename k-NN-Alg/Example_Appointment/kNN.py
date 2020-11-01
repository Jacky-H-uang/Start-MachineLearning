from numpy import *
import operator         # operator 运算模块
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties



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
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1

    return returnMat , classLabelVector



# 数据可视化
def showDatas(datingDataMat , datingLabels):
    # 设置汉字格式
    font = FontProperties(fname=r"C:\Windows\Fonts\STFANGSO.TTF" , size = 14)

    #将 fig 画布分隔成 1 行 1 列,不共享 x 轴和 y 轴,fig 画布的大小为 (13,8)
    #当 nrow = 2 , nclos = 2时,代表 fig 画布被分为四个区域 , axs[0][0] 表示第一行第一个区域
    fig , axs = plt.subplots(nrows = 2 , ncols = 2 , sharex = False , sharey = False , figsize = (13 , 8))

    numberOfLables = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')

    # 画出散点图 , 以 datingDataMat 矩阵第一列(飞行常客例程)，第二列(玩游戏) 数据画散点图，散点大小为 15，透明度为 0.5
    # 设置 x y 轴的标签
    axs[0][0].scatter(x = datingDataMat[: , 0] , y = datingDataMat[: , 1] , color = LabelsColors , s = 15, alpha = .5)
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比' , FontProperties = font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数' , FontProperties = font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占' , FontProperties = font)
    plt.setp(axs0_title_text , size = 9 , weight = 'bold' , color = 'red')
    plt.setp(axs0_xlabel_text , size = 7 , weight = 'bold' , color = 'black')
    plt.setp(axs0_ylabel_text , size = 7 , weight = 'bold' , color = 'black')

    # 画出散点图,以 datingDataMat 矩阵的第一列(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为 15 ,透明度为 0.5
    # 设置 x y 轴的标签
    axs[0][1].scatter(x = datingDataMat[: , 0] , y = datingDataMat[: , 2] , color = LabelsColors , s = 15 , alpha = .5)
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size = 9, weight = 'bold', color = 'red')
    plt.setp(axs1_xlabel_text, size = 7, weight = 'bold', color = 'black')
    plt.setp(axs1_ylabel_text, size = 7, weight = 'bold', color = 'black')

    # 画出散点图,以 datingDataMat 矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为 15,透明度为 0.5
    # 设置标题,x 轴 label, y 轴 label
    axs[1][0].scatter(x=datingDataMat[:, 1], y = datingDataMat[:, 2], color = LabelsColors, s = 15, alpha = .5)
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties = font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties = font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties = font)
    plt.setp(axs2_title_text, size = 9, weight = 'bold', color = 'red')
    plt.setp(axs2_xlabel_text, size = 7, weight = 'bold', color = 'black')
    plt.setp(axs2_ylabel_text, size = 7, weight = 'bold', color = 'black')

    # 设置图例
    didntLike = mlines.Line2D([] , [] , color = 'black' , marker = '.' , markersize = 6 , label = 'didntLike')
    smallDoses = mlines.Line2D([] , [] , color = 'orange' , marker = '.' , markersize = 6 , label = "smallDoses")
    largeDoses = mlines.Line2D([] , [] , color = 'red' , marker = '.' , markersize = 6 , label = "largeDoses")

    # 添加图例
    axs[0][0].legend(handles = [didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles = [didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles = [didntLike,smallDoses,largeDoses])

    plt.show()



# 归一化特征值
# Formula : newValues = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals , (m,1))           # 减去最小特性值
    normDataSet = normDataSet / tile(ranges , (m,1))        # 特征值相除

    return normDataSet , ranges , minVals


# 测试数据  (This function is self-contained)
def datingClassTest():
    hoRatio = 0.10
    datingDataMat , datingLabels = file2matrix('datingTestSet.txt')
    normMat , ranges ,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    erroCount = 0.0
    for i in range(numTestVecs):
        classiferResult = classify0(normMat[i , :] , normMat[numTestVecs : m , :] , datingLabels[numTestVecs:m] , 3)
        print("The classifier came back with : % d , the real answer is : %d" %(classiferResult , datingLabels[i]))

        if classiferResult != datingLabels[i] : erroCount += 1.0

    print("The total error rate is : %f" %(erroCount / float(numTestVecs)))


# 构建一个完整有用的系统
def classifyPerson():
    # 分类结果
    resultList = ['not at all' , 'in small does' , 'in larger does']

    # 输入 Hellen 要判断的数据
    percentTats = float(input("percetage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    # 数据导入并归一化
    datingDataMat , datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minValues = autoNorm(datingDataMat)
    inArr = array([ffMiles , percentTats , iceCream])

    # 测试结果
    classfierResult = classify0((inArr - minValues) / ranges , normMat , datingLabels , 3)
    print("You will probably like this person : " , resultList[classfierResult-1])



if __name__ == '__main__':
    # filename = 'datingTestSet.txt'
    # datingDataMat , datingLabels = file2matrix(filename)
    # showDatas(datingDataMat , datingLabels)
    classifyPerson()