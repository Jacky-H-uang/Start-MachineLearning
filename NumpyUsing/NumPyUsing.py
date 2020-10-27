import numpy as np

randA = np.random.rand(4 , 4)

randMat = np.mat(randA)

# .I 的操作是对举证求逆
randMatReverse = randMat.I

randMul = randMat * randMatReverse

# eye() 创建 4 * 4 的单位矩阵
randEye = randMul - np.eye(4)

print(randA)
print(randMat)
print(randMatReverse)
print(randMul)
print(randEye)
