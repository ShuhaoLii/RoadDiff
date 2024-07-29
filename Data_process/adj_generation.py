import numpy as np
import pandas as pd

# 初始化一个 8x8 的零矩阵
adjacency_matrix = np.zeros((18, 18))

# 遍历矩阵设置相邻节点的关系
for i in range(18):
    adjacency_matrix[i,i] = 1
    if i > 0:  # 如果不是第一个节点，设置与前一个节点的连接
        adjacency_matrix[i, i-1] = 1
    if i < 17:  # 如果不是最后一个节点，设置与后一个节点的连接
        adjacency_matrix[i, i+1] = 1

adjacency_matrix = pd.DataFrame(adjacency_matrix,dtype=int)
# 输出邻接矩阵
print(adjacency_matrix)
adjacency_matrix.to_csv('../Datasets/HuaNan_road_speed_adj.csv')
