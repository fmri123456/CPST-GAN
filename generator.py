import torch.nn as nn
from generator_Convolution import generator_Conv

# 生成器模型设计
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        # 卷积层
        self.conv1 = generator_Conv()

    def forward(self, x, A):
        cn1 = self.conv1.forward(x, A)
        return cn1
