from Decision_Trees.Shannon_entropy.shannon_entropy import *
from math import log
import operator
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth" , fc = "0.8")
leafNode = dict(boxstyle = "round4" , fc = "0.8")
arrow_args = dict(arrowstyle = "<-")



"""
Two Functions get ： 
            how many leaf nodes you have
            how many levels you have
"""
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs += 1

    return numLeafs
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth : maxDepth = thisDepth

    return maxDepth




"""
Funtion : 返回一个测试数据
"""
def retrieveTree(i):
    listOfTrees = [
                   {'no surfacing' : {0 : 'no' , 1 : {'flippers' : {0 : ' no' , 1 : 'yes'}}}},
                   {'no surfacing' : {0 : 'no' , 1 : {'flippers' : {0 : {'head' : {0 : 'no' , 1 : 'yes'}} , 1 : 'no'}}}}
                  ]

    return listOfTrees[i]





"""
Function : 画结点
    nodeTxT :  结点名
    centerPt : 文本位置
    parentPt : 标注的箭头位置
    nodeType : 结点格式
"""
def plotNode(nodeTxt , centerPt , parentPt , nodeType):
    createPlot.ax1.annotate(nodeTxt , xy = parentPt , xycoords = 'axes fraction' , xytext = centerPt , textcoords = 'axes fraction',
                            va = "center" , ha = "center" , bbox = nodeType , arrowprops = arrow_args)





"""
Function : 创建绘画板
"""
def createPlot():
    fig = plt.figure(1 , facecolor = 'white')
    fig.clf()               # 将画图清空
    createPlot().ax1 = plt.subplot(111,frameon = False)
    plotNode('a decision node' , (0.5 , 0.1) , (0.1 , 0.5) , decisionNode)
    plotNode('a leaf node' , (0,8 , 0.1) , (0.3 , 0.8) , leafNode)
    plt.show()


"""
Function  : 标注有向边属性值
    
    cntrPt、parentPt  - 计算标注的位置
    txtString         - 标注的内容
"""
def plotMidText(cntrPt , parentPt , txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot().ax1.text(xMid , yMid , txtString)






"""
Function : Visualize the Tree
"""
def plotTree(myTree , parentPt , nodeTxt):
    numLeafs = getNumLeafs(myTree)
    getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW , plotTree.yOff)

    plotMidText(cntrPt , parentPt , nodeTxt)
    plotNode(firstStr , cntrPt , parentPt , decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key] , cntrPt , str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key] , (plotTree.xOff , plotTree.yOff) , cntrPt , leafNode)
            plotMidText((plotTree.xOff , plotTree.yOff) , cntrPt , str(key) ,)
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
def createPlot(inTree):
    fig = plt.figure(1,facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [] , yticks = [])
    createPlot().ax1 = plt.subplot(111,frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree , (0.5,1.0),' ')
    plt.show()





