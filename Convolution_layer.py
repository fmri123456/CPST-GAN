import math
import numpy as np
import torch
from torch import nn
from numpy import linalg as la
import generator_conv
# 卷积层的设计
class Convolution_layer(nn.Module):
    def __init__(self, size, B, use_bias=False):
        super(Convolution_layer, self).__init__()
        self.use_bias = use_bias
        self.size = size
        # 卷积核
        self.B = B
    def save_weight(self):
        pass
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.kernel.size(1))
        self.kernel.data.uniform_(-stdv, stdv)

    def forward(self, input_feature):
        # kernel = self.kernel
        H = input_feature.detach().numpy()
        w = generator_conv.set_weight(input_feature)
        A = generator_conv.set_A(w)
        num = np.size(A, 0)
        row = np.size(A, 1)
        col = np.size(A, 2)
        A_1 = np.zeros(shape=(num, row, col))
        D = np.zeros(shape=(num, row, row))
        V = np.zeros((num, row, row))
        for i in range(num):
            E = np.eye(row, col)
            A_1[i] = A[i] + E
        for n in range(num):
            for i in range(row):
                D[n][i][i] = np.sum(A_1[n][i])
        v, Q = la.eig(D)  # 求D的特征值和特征向量
        for p in range(num):  # 求特征值的对角矩阵
            for q in range(row):
                V[p, q, q] = v[p, q]
        V = np.sqrt(V)
        D_12 = np.linalg.inv((Q * V * Q))  # D的负1/2次幂
        # H_1 = D_12 * A_1 * D_12 * H * self.B
        D_12 = torch.FloatTensor(D_12)
        A_1 = torch.FloatTensor(A_1)
        H = torch.FloatTensor(H)
        H_1 = torch.matmul(torch.bmm(torch.bmm(torch.bmm(D_12, A_1), D_12), H), self.B)

        return H_1
