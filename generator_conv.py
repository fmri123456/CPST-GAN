import numpy as np
import torch

# 互信度矩阵T
def set_T(A):
    A = A.detach().numpy()
    row = A.shape[0]
    col = A.shape[1]
    T = np.zeros((row, col, col))
    # 脑区与脑区，基因与基因之间不存在互信度
    for k in range(row):
        h = A[k].reshape(col, col).astype(np.int32)
        for i in range(116):
            for j in range(116, col):
                T[k, i, j] = np.sum(np.bitwise_and(h[i], h[j])) / min(sum(h[i]), sum(h[j]))
                T[k, j, i] = T[k, i, j]
    T = torch.FloatTensor(T)
    return T

# 互信度对角矩阵Ts
def set_Ts(T):
    T = T.detach().numpy()
    row = T.shape[0]
    col = T.shape[1]
    Ts = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            Ts[k, i, i] = sum(T[k, i])
    Ts = torch.FloatTensor(Ts)
    return Ts

# 节点调控量接收比例矩阵R,prop用于调参实验
def set_R(prop):
    R = prop * np.ones(161)
    R = np.diag(R)
    R = torch.FloatTensor(R)
    return R

# 条件矩阵C,α用于调参实验
def set_C(T, alpha):
    C = T
    C[T < alpha] = 0
    return C

# 条件基因对相应脑区的调控概率P,β用于调参实验
def set_P(P, beta):
    P.data[P.data >= beta] = 1
    P.data[P.data < beta] = 0
    return P

# 条件基因对相应脑区的调控量Hr
# 分三步：1.计算条件基因调控脑区的条件概率矩阵CP 2.计算条件基因对脑区的调控比例矩阵RP 3.计算调控量矩阵Hr
def set_Hr(C, P, Ts, x):
    C = C.detach().numpy()
    P = P.detach().numpy()
    Ts = Ts.detach().numpy()
    x = x.detach().numpy()
    col = C.shape[1]
    CP = C * P  # 哈达玛积
    # 全1矩阵F
    F = np.ones((col, col))
    RP = np.transpose(CP, [0, 2, 1]) * (np.matmul(F, np.transpose(Ts, [0, 2, 1])))
    Hr = np.matmul(RP, x)
    Hr = torch.FloatTensor(Hr)
    return Hr

# 权重矩阵(皮尔逊相关系数)
def set_weight(X):
    X = X.detach().numpy()
    # 节点个数
    row = X.shape[0]
    # 特征个数
    col = X.shape[1]
    # 初始化权重矩阵
    W = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            for j in range(col):
                # 主对角线元素为0
                if j != i:
                    # Pearson
                    W[k][i][j] = np.min(np.corrcoef(X[k][i], X[k][j]))
                    # Spearman
                    # W[k][i][j] = stats.spearmanr(X[k][i], X[k][j])[0]  # 返回两个值：correlation,pvalue。
                    # Kendall
                    # W[k][i][j] = stats.kendalltau(X[k][i], X[k][j])[0]  # 返回两个值：correlation,pvalue。
    W = weight_threshold(W)
    W = torch.FloatTensor(W)
    return W


# 权重矩阵阈值化(关联系数最小的20%元素置0)
def weight_threshold(W):
    row = W.shape[0]
    col = W.shape[1]
    result = np.zeros((row, col, col))
    for i in range(row):
        threshold = np.sort(np.abs(W[i].flatten()))[int(col * col * 0.2)]  # 阈值
        result[i] = W[i] * (np.abs(W[i]) >= threshold)
    return result

def set_A(W):
    W = W.detach().numpy()
    row = W.shape[0]
    col = W.shape[1]
    A = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            for j in range(col):
                if W[k, i, j] != 0:
                    A[k, i, j] = 1
    A = torch.FloatTensor(A)
    return A

