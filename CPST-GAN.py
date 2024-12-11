import os
import scipy.io as sio
import torch.nn as nn
import utils
import torch
from generator import generator
from discriminator import discriminator_model
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import generator_conv

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 不用gpu，cuda有点问题

# --------------------------------------------
# 加载数据，包括AD、LMCI
dataFile = 'AD risk prediction task based LMCI.mat'
data = sio.loadmat(dataFile)
AD = data['feat_AD']  # 107*161*70
LMCI = data['feat_LMCI']  # 107*161*70
feat = np.concatenate((LMCI, AD), axis=0)
print("数据加载完成...")
AD = torch.FloatTensor(AD)
LMCI = torch.FloatTensor(LMCI)
LMCI_W = generator_conv.set_weight(LMCI)  # 107*161*161
AD_W = generator_conv.set_weight(AD)  # 107*161*161
LMCI_A = generator_conv.set_A(LMCI_W)
AD_A = generator_conv.set_A(AD_W)
LMCI_test = data['AD_true']
AD_test = data['AD_fake']
test = np.concatenate((AD_test, LMCI_test), axis=0)
test = torch.FloatTensor(test)
W_test = generator_conv.set_weight(test)
A_test = generator_conv.set_A(W_test)
# -------------------------------------------
# 加载数据，包括EMCI、LMCI
# dataFile = 'LMCI risk prediction task based EMCI.mat'
# data = sio.loadmat(dataFile)
# EMCI = data['feat_EMCI']  # 131*164*70
# LMCI = data['feat_LMCI']  # 131*164*70
# feat = np.concatenate((EMCI, LMCI), axis=0)
# print("数据加载完成...")
# EMCI = torch.FloatTensor(EMCI)
# LMCI = torch.FloatTensor(LMCI)
# EMCI_W = generator_conv.set_weight(EMCI)  # 131*164*164
# LMCI_W = generator_conv.set_weight(LMCI)  # 131*164*164
# EMCI_A = generator_conv.set_A(EMCI_W)
# LMCI_A = generator_conv.set_A(LMCI_W)
# EMCI_test = data['LMCI_true']
# LMCI_test = data['LMCI_fake']
# test = np.concatenate((LMCI_test, EMCI_test), axis=0)
# test = torch.FloatTensor(test)
# W_test = generator_conv.set_weight(test)
# A_test = generator_conv.set_A(W_test)
# ------------------------------------------
# 加载数据，包括EMCI、AD
# dataFile = 'AD risk prediction task based EMCI.mat'
# data = sio.loadmat(dataFile)
# EMCI = data['feat_EMCI']  # 145*161*70
# AD = data['feat_AD']  # 145*161*70
# feat = np.concatenate((EMCI, AD), axis=0)
# print("数据加载完成...")
# EMCI = torch.FloatTensor(EMCI)
# AD = torch.FloatTensor(AD)
# EMCI_W = generator_conv.set_weight(EMCI)  # 145*161*161
# AD_W = generator_conv.set_weight(AD)  # 145*161*161
# EMCI_A = generator_conv.set_A(EMCI_W)
# AD_A = generator_conv.set_A(AD_W)
# EMCI_test = data['AD_true']
# AD_test = data['AD_fake']
# test = np.concatenate((AD_test, EMCI_test), axis=0)
# test = torch.FloatTensor(test)
# W_test = generator_conv.set_weight(test)
# A_test = generator_conv.set_A(W_test)


feat_num = feat.shape[1]
feat_seq = feat.shape[2]


# 其他参数定义
LR = 0.00000001
EPOCH = 1
batch_size = 32

# 构建数据集
dataset = TensorDataset(LMCI, AD, LMCI_A, AD_A)
# 加载数据集
dataload = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
print("数据导入成功！")

G = generator()
D = discriminator_model(feat_num, feat_seq)

# 损失函数：二进制交叉熵损失(既可用于二分类，又可用于多分类)
criterion = nn.CrossEntropyLoss()
# 生成器的优化器
g_optimizer = torch.optim.Adam(G.parameters(), lr=LR)
# 判定器的优化器
d_optimizer = torch.optim.Adam(D.parameters(), lr=LR)

# 开始训练
print("开始训练...")
max_acc = 0
D_loss = []
G_loss = []
acc_list = []
for epoch in range(EPOCH):
    for step, (LMCI_train, AD_train, LMCI_A_train, AD_A_train) in enumerate(dataload):
        print('第{}次训练第{}批数据'.format(epoch + 1, step + 1))
        num_img = LMCI_train.size(0)
        G_LMCI = G(LMCI_train, LMCI_A_train)
        # 定义标签
        label = np.concatenate((np.zeros(num_img), np.ones(num_img)))
        label = utils.onehot_encode(label)
        label = torch.LongTensor(np.where(label)[1])
        fake_label = label[0:num_img]  # 定义假label为0
        real_label = label[num_img:2 * num_img]  # 定义真实label为1
        # 计算损失
        real_out = D(AD_train)
        d_loss_real = criterion(real_out, real_label)
        fake_out = D(G_LMCI)
        d_loss_fake = criterion(fake_out, fake_label)

        # 反向传播和优化
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        # 训练生成器
        gen = G(LMCI_train, LMCI_A_train)
        fake_output = D(gen)
        g_loss = criterion(fake_output, real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        # 保存每批数据的损失
        D_loss.append(d_loss)
        G_loss.append(g_loss)
        print('D_loss={:.4f}, G_loss={:.4f}'.format(sum(D_loss)/len(D_loss), sum(G_loss)/len(G_loss)))
    # 每10个为一个epoch
    if epoch % 10 == 0:
        num_test = test.shape[0]
        G.eval()
        # 定义标签
        label = np.concatenate((np.zeros(num_test / 2), np.ones(num_test / 2)))
        label = utils.onehot_encode(label)
        label = torch.LongTensor(np.where(label)[1])
        # fake_label = label[:num_test]  # 定义假label为0
        # real_label = label[num_test:]  # 定义真实label为1
        gen_test = G(test, A_test)
        output = D(gen_test)
        acc_val = utils.accuracy(output, label)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Test set results:",
              "accuracy= {:.4f}".format(acc_val.item()))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        acc_list.append(float(acc_val.item()))

print("best accuracy={:.4f}".format(max_acc))
# 保存
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')

# ==============================================================

num_test = test.shape[0]
G.eval()
# 定义标签
label = np.concatenate((np.zeros(num_test / 2), np.ones(num_test / 2)))
label = utils.onehot_encode(label)
label = torch.LongTensor(np.where(label)[1])
gen_test = G(test, A_test)
output = D(gen_test)

from sklearn.metrics import roc_curve, auc

fpr, tpr, thr = roc_curve(label, gen_test)
roc_auc = auc(fpr, tpr)
ACC, SEN, SPE, MCC = utils.predict_indicators(gen_test, label)
print("ACC=", ACC)
print("SEN=", SEN)
print("SPE=", SPE)
print("MCC=", MCC)
print("tpr=", tpr)
print("fpr=", fpr)


