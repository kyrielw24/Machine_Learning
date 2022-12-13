# 导入实验所需的相关库函数
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

# 下载实验所需的MNIST数据集
train_dataset = MNIST(
    root='MNIST',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

# 查看mnist数据
# print(train_dataset)

# 查看是否支持cuda GPU训练
# print(torch.cuda.is_available())

# 数据集切分
batch_size = 128
dataIter = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# for step, (batch_x, batch_y) in enumerate(dataIter):
#     print('| Step: ', step, '| batch x: ', batch_x.shape, '| batch y: ', batch_y.shape)
# dataIter中存有各个小批次的 X Y 数据
# batch x:  torch.Size([128, 1, 28, 28]) | batch y:  torch.Size([128])

# 进行VAE操作
# 展平
def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)  # torch.Size([128, 784])


class VariationalAutoEncoder_Conv(nn.Module):
    def __init__(self,
                 image_shape=(1, 28, 28),
                 conv_size=(1, 6, 16, 4),
                 dense_size=(784, 128),
                 kernel=3,
                 z_dim=20):
        super(VariationalAutoEncoder_Conv, self).__init__()

        self.relu = nn.ReLU()  # 默认激活函数采用 Relu 激活
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # encoder卷积层
        # 输入 (128,1,28,28) 输出 (128,6,28,28)
        self.enConv0 = nn.Conv2d(conv_size[0], conv_size[1], kernel, padding=1, padding_mode='replicate')
        # 输入 (128,6,28,28) 输出 (128,16,14,14)
        self.enConv1 = nn.Conv2d(conv_size[1], conv_size[2], kernel, stride=2, padding=1, padding_mode='replicate')
        # 输入 (128,16,14,14) 输出 (128,4,14,14)
        self.enConv2 = nn.Conv2d(conv_size[2], conv_size[3], kernel, padding=1, padding_mode='replicate')

        # encoder全连接层
        self.enDense0 = nn.Linear(dense_size[0], dense_size[1])

        # 隐变量层
        self.latent_mean = nn.Linear(dense_size[-1], z_dim)
        self.latent_log_var = nn.Linear(dense_size[-1], z_dim)

        # decoder全连接层
        self.deDense0 = nn.Linear(z_dim, dense_size[-1])
        self.deDense1 = nn.Linear(dense_size[-1], dense_size[-2])
        self.deDense2 = nn.Linear(dense_size[-2], int(np.prod(image_shape)))

        # 需要对 全连接层输出的一维结果数据 进行 reshape
        self.dec_reshape = transforms.Lambda(
            lambda x: torch.reshape(x, (x.shape[0], 1, int(image_shape[1]), int(image_shape[2]))))

    # 编码
    def encode(self, x):
        # 卷积层
        out = self.relu(self.enConv0(x))
        out = self.relu(self.enConv1(out))
        out = self.relu(self.enConv2(out))

        # 展平
        out = flatten(out)

        # 全连接层
        out = self.relu(self.enDense0(out))

        # 均值 mean
        latent_mean = self.latent_mean(out)
        # log方差 log_var
        latent_log_var = self.latent_log_var(out)

        return latent_mean, latent_log_var

    # 重参数化生成隐变量
    def re_parameterize(self, mu, log_var):
        var = torch.exp(log_var)
        epsilon = torch.randn_like(var)
        return mu + torch.mul(var, epsilon)

    # 解码
    def decode(self, z):
        # 全连接层
        out = self.relu(self.deDense0(z))
        out = self.relu(self.deDense1(out))
        out = self.sigmoid(self.deDense2(out))

        # 重构shape
        out = self.dec_reshape(out)

        return out

    # 整个前向传播过程：编码 --> 解码
    def forward(self, x):
        latent_mean, latent_log_var = self.encode(x)
        sampled_Z = self.re_parameterize(latent_mean, latent_log_var)
        Gen_X = self.decode(sampled_Z)
        return latent_mean, latent_log_var, Gen_X


class VariationalAutoEncoder_Dense(nn.Module):
    def __init__(self,
                 image_shape=(1, 28, 28),
                 dense_size=(784, 256, 128),
                 z_dim=20):
        super(VariationalAutoEncoder_Dense, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # encoder全连接层
        self.enDense0 = nn.Linear(dense_size[0], dense_size[1])
        self.enDense1 = nn.Linear(dense_size[1], dense_size[2])

        # 隐变量层
        self.latent_mean = nn.Linear(dense_size[-1], z_dim)
        self.latent_log_var = nn.Linear(dense_size[-1], z_dim)

        # decoder全连接层
        self.deDense0 = nn.Linear(z_dim, dense_size[-1])
        self.deDense1 = nn.Linear(dense_size[-1], dense_size[-2])
        self.deDense2 = nn.Linear(dense_size[-2], dense_size[-3])

        # decoder reshape
        self.dec_reshape = transforms.Lambda(
            lambda x: torch.reshape(x, (x.shape[0], 1, int(image_shape[1]), int(image_shape[2]))))

    # 编码
    def encode(self, x):
        # 展平
        out = flatten(x)
        # 全连接层
        out = self.relu(self.enDense0(out))
        out = self.relu(self.enDense1(out))

        # 均值 mean
        latent_mean = self.latent_mean(out)
        # log方差 log_var
        latent_log_var = self.latent_log_var(out)

        return latent_mean, latent_log_var

    # 重参数化生成隐变量
    def re_parameterize(self, mu, log_var):
        var = torch.exp(log_var)
        epsilon = torch.randn_like(var)
        return mu + torch.mul(var, epsilon)

    # 解码
    def decode(self, z):
        # 全连接层
        out = self.relu(self.deDense0(z))
        out = self.relu(self.deDense1(out))
        out = self.sigmoid(self.deDense2(out))

        # 重构shape
        out = self.dec_reshape(out)

        return out

    # 整个前向传播过程：编码 --> 解码
    def forward(self, x):
        latent_mean, latent_log_var = self.encode(x)
        sampled_Z = self.re_parameterize(latent_mean, latent_log_var)
        Gen_X = self.decode(sampled_Z)
        return latent_mean, latent_log_var, Gen_X


def vae_loss(x, gen_x, mean, log_var):
    # 重构项损失
    mse_loss = torch.nn.MSELoss(reduction='sum')
    loss = mse_loss(x, gen_x)
    # loss = F.binary_cross_entropy(gen_x, x, reduction='sum')
    # 最小化 q(z|x)  和 p(z) 的距离
    KL_loss = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mean, 2) - log_var - 1)
    # print("loss_1:", loss.item(), "loss_2:", KL_loss.item())
    return loss + KL_loss


# 训练模型参数
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
learning_rate = 1e-3  # 学习率
epoches = 40  # 迭代40次

# 实例化模型
modelD = VariationalAutoEncoder_Dense().to(device)
modelC = VariationalAutoEncoder_Conv().to(device)

# 创建优化器
optimizerD = torch.optim.Adam(modelD.parameters(), lr=learning_rate)
optimizerC = torch.optim.Adam(modelC.parameters(), lr=learning_rate)

# 模型训练
number = len(dataIter.dataset)
train_lossD = []  # 保存每个epoch的训练误差
train_lossC = []  # 保存每个epoch的训练误差
result_dir_Dense = './VAEResult/Dense'
result_dir_Conv = './VAEResult/Conv'

for epoch in range(epoches):
    batch_lossD = []
    batch_lossC = []
    for i, (x, y) in enumerate(dataIter):
        x = x.to(device)  # gpu训练

        # 前向传播
        mean_d, log_var_d, gen_x_d = modelD(x)
        mean_c, log_var_c, gen_x_c = modelC(x)

        # 计算损失函数
        lossD = vae_loss(x, gen_x_d, mean_d, log_var_d)
        batch_lossD.append(lossD.item())
        lossC = vae_loss(x, gen_x_c, mean_c, log_var_c)
        batch_lossC.append(lossC.item())

        # 反向传播和优化
        optimizerD.zero_grad()  # 每一次循环之前，将梯度清零
        lossD.backward()  # 反向传播
        optimizerD.step()  # 梯度下降
        optimizerC.zero_grad()  # 每一次循环之前，将梯度清零
        lossC.backward()  # 反向传播
        optimizerC.step()  # 梯度下降

        # 输出batch信息
        if i % 100 == 0 and i > 0:
            print("epoch : {0} | batch : {1} | batch average(Dense) loss: {2}| batch average(Conv) loss: {3}"
                  .format(epoch + 1, i, lossD.item() / x.shape[0], lossC.item() / x.shape[0]))

        # if i == 0:
        #     x_concatD = torch.cat([x.view(-1, 1, 28, 28), gen_x_d.view(-1, 1, 28, 28)], dim=3)
        #     save_image(x_concatD, './%s/reconstructed-%d.png' % (result_dir_Dense, epoch + 1))
        #     x_concatC = torch.cat([x.view(-1, 1, 28, 28), gen_x_c.view(-1, 1, 28, 28)], dim=3)
        #     save_image(x_concatC, './%s/reconstructed-%d.png' % (result_dir_Conv, epoch + 1))

    # 输出epoch信息
    train_lossD.append(np.sum(batch_lossD) / number)
    train_lossC.append(np.sum(batch_lossC) / number)
    print("epoch[{}/{}] | loss(Dense):{}| loss(Conv):{}"
          .format(epoch + 1, epoches, train_lossD[epoch], train_lossC[epoch]))

# evaluation 测试生成效果，从正态分布随机采样z
z = torch.randn((batch_size, 20)).to(device)
logitsD = modelD.decode(z)  # 仅通过解码器生成图片
x_hat = torch.sigmoid(logitsD)  # 转换为像素范围
x_hat = x_hat.view(128, 28, 28).detach().cpu().numpy() * 255.
# 展示图片
_, axes = plt.subplots(6, 6)
for i in range(6):
    for j in range(6):
        axes[i][j].axis('off')
        axes[i][j].imshow(x_hat[i * 3 + j], cmap='gray')
plt.show()

logitsC = modelC.decode(z)  # 仅通过解码器生成图片
x_hat = torch.sigmoid(logitsC)  # 转换为像素范围
x_hat = x_hat.view(128, 28, 28).detach().cpu().numpy() * 255.
# 展示图片
_, axes = plt.subplots(6, 6)
for i in range(6):
    for j in range(6):
        axes[i][j].axis('off')
        axes[i][j].imshow(x_hat[i * 3 + j], cmap='gray')
plt.show()