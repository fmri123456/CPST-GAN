import numpy as np
import torch
import torch.nn as nn
import generator_conv
from torch.nn.parameter import Parameter


# 生成器卷积设计
class generator_Conv(nn.Module):
    def __init__(self):
        super(generator_Conv, self).__init__()  # 初始化
        P = np.zeros((161, 161))
        # 只有基因可以调控脑区，脑区不可以调控基因
        P[116:161, 0:116] = np.random.random(size=(45, 116))
        # 可学习参数P
        self.P = Parameter(torch.FloatTensor(P), requires_grad=True)
        self.linear = nn.Sequential(nn.Linear(45*45, 116*45),
                                    nn.Linear(116*45, 116*116))

    # 这里的x表示的是特征矩阵，w表示权重矩阵，A表示邻接矩阵
    def forward(self, x, A):
        # 互信度矩阵
        T = generator_conv.set_T(A)
        # 互信度对角矩阵
        Ts = generator_conv.set_Ts(T)
        # 条件矩阵
        C = generator_conv.set_C(T, 0.3)
        # 调控概率矩阵
        P = generator_conv.set_P(self.P, 0.5)
        # 节点调控量接收比例矩阵
        R = generator_conv.set_R(0.5)
        # 条件基因对相应脑区的调控量Hr
        Hr = generator_conv.set_Hr(C, P, Ts, x)
        # 更新节点的特征信息
        delta_H = torch.matmul(R, Hr)
        new_H = x + delta_H
        new_w = generator_conv.set_weight(new_H)
        return new_H
