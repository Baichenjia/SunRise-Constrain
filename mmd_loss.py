# Compute MMD distance using pytorch

import torch
import torch.nn as nn
import numpy as np


device = torch.device('cpu')


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)                   # (batch-size*2, n_dims)
        total0 = total.unsqueeze(0).expand(                          # (batch-size*2, batch-size*2, n_dims)
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(                          # (batch-size*2, batch-size*2, n_dims)
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)                    # (batch-size*2, batch-size*2)
        if fix_sigma:
            bandwidth = fix_sigma                                    #
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)  # 标量
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        # 数目与 kernel_nu 相等, 每个元素的 shape=(64,64)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def laplacian_kernel(self, x1, x2, sigma=20.0):
        d12 = torch.sum(torch.abs(x1[None] - x2[:, None]), dim=-1)
        k12 = torch.exp(- d12 / sigma)
        return k12

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss
        elif self.kernel_type == 'laplacian':
            use_sqrt = False
            k11 = torch.mean(self.laplacian_kernel(source, source), dim=[0, 1])
            k12 = torch.mean(self.laplacian_kernel(source, target), dim=[0, 1])
            k22 = torch.mean(self.laplacian_kernel(target, target), dim=[0, 1])
            if use_sqrt:
                return torch.sqrt(k11 + k22 - 2 * k12 + 1e-8)
            else:
                return k11 + k22 - 2 * k12


def calculate_mmd(mem, dqn_list, round=10, batch_size=32):
    mmd = MMD_loss(kernel_type='rbf')
    dis_list = []

    for i in range(round):
        _, states, _, _, _, _, _, _, oldQ = mem.sample(batch_size)   # state.shape=(32,4,84,84)
        # current Q
        currentQ_list = [dqn_list[en_index].online_net(states.cuda()) for en_index in range(len(dqn_list))]
        currentQ = torch.stack(currentQ_list, dim=1).detach().to(device)
        # mmd distance
        dis = mmd(currentQ.view(-1, 51), oldQ.to(device).view(-1, 51))
        dis_list.append(dis.item())
        # print("i:", i, ", mmd:", dis.item())
    return np.mean(dis_list)


def calculate_kl(mem, dqn_list, round=10, num_ensemble=3, batch_size=32):
    dis_list = []
    for i in range(round):
        _, states, _, _, _, _, _, _, oldQ = mem.sample(batch_size)   # state.shape=(32,4,84,84)
        oldQ_d = torch.distributions.categorical.Categorical(probs=oldQ.view(-1, 51))
        # current Q
        currentQ_list = [dqn_list[en_index].online_net(states.cuda()) for en_index in range(len(dqn_list))]
        currentQ = torch.stack(currentQ_list, dim=1).detach().view(-1, 51)
        currentQ_d = torch.distributions.categorical.Categorical(probs=currentQ)
        dis = torch.distributions.kl.kl_divergence(currentQ_d, oldQ_d).mean()
        dis_list.append(dis.item())
        # print("i:", i, ", kl:", dis.item())
    return np.mean(dis_list)


if __name__ == '__main__':
    mmd1 = MMD_loss(kernel_type='rbf')
    mmd2 = MMD_loss(kernel_type='laplacian')
    x = torch.randn((32, 10))
    y = torch.randn((32, 10))
    dis1 = mmd1(x, y)
    print(dis1)

    dis2 = mmd2(x, y)
    print(dis2)
