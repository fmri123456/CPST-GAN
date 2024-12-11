import torch
from Convolution_layer import Convolution_layer
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

# 判定器模型设计
class discriminator_model(nn.Module):
    def __init__(self, num, size):
        super(discriminator_model, self).__init__()  # 初始化
        self.num = num
        self.size = size
        # 参数矩阵B
        self.B = Parameter(torch.FloatTensor(size, size))
        self.reset_parameters()
        # 卷积层
        self.conv1 = Convolution_layer(self.size, self.B, False)
        self.conv2 = Convolution_layer(self.size, self.B, False)
        # 连接层设计
        self.fc1 = nn.Linear(self.num * self.size, self.size // 2)
        self.fc2 = nn.Linear(self.size // 2, 2)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.B.size(1))
        self.B.data.uniform_(-stdv, stdv)

    def forward(self, input_feature):
        # 卷积层后的结果
        cn1 = F.relu(self.conv1.forward(input_feature))
        cn2 = torch.sigmoid(self.conv1.forward(cn1))
        # 拉直运算
        cn2_rl = torch.flatten(cn2, 1, -1)
        # 全连接层运算
        fc1 = F.relu(self.fc1(cn2_rl))
        # judge_label = F.softmax(self.fc2(fc1), dim=0)  # 判断是否为患者
        judge_net = torch.sigmoid(self.fc2(fc1))  # 判断网络是否为生成的
        return judge_net
