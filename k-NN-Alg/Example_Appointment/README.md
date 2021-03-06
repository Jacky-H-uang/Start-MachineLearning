# 约会匹配学习(kNN 实现约会匹配)：
## 主要流程：
1. 收集数据：提供文本文件。
2. 准备数据：使用Python解析文本文件。
3. 分析数据：使用Matplotlib画二维扩散图。
4. 训练算法：此步骤不适用于k-近邻算法。
5. 测试算法：使用海伦提供的部分数据作为测试样本。测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
6. 使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否 为自己喜欢的类型。




### 1：Parse data from text
>文本特征：
>
> - 每年获得的飞行常客里程数 
>
> - 玩视频游戏所耗时间百分比 
>
> - 每周消费的冰淇淋公升数


### 2：Analyze Data
> 使用 Matplotlib 创建可视化的散点图


### 3：Normalizing Data
> - 在处理这种不同取值范围的特征值时，我们通常采用的方法是将数值归一化，如将取值范围
> 处理为 0 到 1 或者 -1 到 1 之间。处理归一化的时候有公式：
> Formula : newValue = (oldValue - min) / (max - min)
> 其中 min 和 max 分别表示最小特征值和最大特征值
> 


### 4：Testing the classifier as a whole program
> - The error rate is the number of misclassified pieces of data
> divided by the total number of data points tested.
>
> - An error rate of 0 means you have a perfect classifier,
> and an error rate of 1.0 means yhe classifier is always wrong.


### 5：Constructing an Useful System
> 创建一个输入输出数据的 function ，然后输出学习的结果。
> 至此 KNN 实现约会匹配的学习算法完成。