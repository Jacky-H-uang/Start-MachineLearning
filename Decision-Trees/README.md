# This is Introduction of the Decision Trees,and the contents are following:

- Introducing decision trees.
- Measuring consistency(一致性) in a dataset.
- Using recursion to construct a decision tree.
- Plotting trees in Matplotlib.

### General approach to decision trees
1. Collect : Any method.
2. Prepare : This tree-building algorithm works only on nomial values , so any continuous values will need to be quantized.
3. Analyze : Any merthod . You should visually inspect the tree after it is built.
4. Train : Construct a tree data structure.
5. Test : Calculate the error rate with the learned tree.
6. Use : This can be used in ant supervised learning task.Often , trees are used to better understand the data.




### Prerequisite knowledge: Information Theory.
>Shannon entropy (香农熵):
>
>- Entropy is defined as the expected values of the infotmation.
>
>- The information of xi is defined as : l(xi) = log2P(xi)
>
>- You need to calculate all the expected value of all the information of all posible values of our class.
>
>- the Python codes to calculate is in file "Shannon-entropy"


