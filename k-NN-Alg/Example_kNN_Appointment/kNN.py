from numpy import *
import operator         # operator 运算模块
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties



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



if __name__ == '__main__':
    filename = 'datingTestSet.txt'
    datingDataMat , datingLabels = file2matrix(filename)
    showDatas(datingDataMat , datingLabels)