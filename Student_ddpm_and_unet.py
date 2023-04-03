import torch
from torch import nn

from sinusoidal_embedding import sinusoidal_embedding
import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tt_to_tensor
import torch.nn.functional as F

class student_MyDDPM(nn.Module):
    def __init__(self, network, n_steps=1000, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(student_MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)  #注意：L[:3]表示截取前三个元素
        #因此上一行的操作就是在计算alphabar这一连乘的数值,返回的应该还是一个向量

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]  #在这里应当表示的是输入的时间t获得的a_bar,在这里是一个向量

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device) #randn表示从标准正态分布中抽取的一组随机数，其联合分布也服从标准正态分布，个变量直接是相互独立的,并且要和x0的形状要一致
            #需要在理解一下n和reshape的意义，但是eta一般情况下是需要进行上传的，但是在展现前向过程的时候（演示）并不需要输入eta
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta  #用于实现加噪声的步骤，xt=x0和zt的关系式，并且应当启用了广播机制
        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        #print('xbackward', x.shape)
        return self.network(x, t)

def myconv2d(x, weight, bias, stride, pad):
    n, c, h_in, w_in = x.shape
    d, c, k, j = weight.shape
    x_pad = torch.zeros(n, c, h_in+2*pad, w_in+2*pad)   # 对输入进行补零操作
    if pad>0:
        x_pad[:, :, pad:-pad, pad:-pad] = x
    else:
        x_pad = x

    x_pad = x_pad.unfold(2, k, stride)
    x_pad = x_pad.unfold(3, j, stride)        # 按照滑动窗展开
    out = torch.einsum(                          # 按照滑动窗相乘，
        'nchwkj,dckj->ndhw',                    # 并将所有输入通道卷积结果累加
        x_pad, weight)
    out = out + bias.view(1, -1, 1, 1)          # 添加偏置值 ，广播机制
    return out

class ttconv(nn.Module):
    def __init__(self, ranks,  each_core_dim, convnumbers,  **kwargs):
        super(ttconv, self).__init__(**kwargs)
        self.convnumbers = convnumbers
        self.ranks = list(ranks)
        self.each_core_dim = list(each_core_dim)
        self.coresnumber = len(self.each_core_dim)

        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)
        self.bias.data.uniform_(-0.1, 0.1)
        self.cores = []
        for i in range(self.convnumbers):
            self.cores.append([])
            # cores = [[[1, 2, 3], [3, 2, 4], [4, 3, 5], [5, 4]], [[1, 2, 3], [3, 2, 4], [4, 3, 5], [5, 4]]]
            for j in range(self.coresnumber):
                # cores[i].append([i*j])
                if j == 0:
                    self.cores[i].append(nn.Parameter(tl.zeros(1, self.each_core_dim[j], self.ranks[j]), requires_grad=True))
                elif j == self.coresnumber - 1:
                    self.cores[i].append(nn.Parameter(tl.zeros(self.ranks[j - 1], self.each_core_dim[j], 1), requires_grad=True))
                else:
                    self.cores[i].append(nn.Parameter(tl.zeros(self.ranks[j - 1], self.each_core_dim[j], self.ranks[j]), requires_grad=True))

        for i in range(self.convnumbers):
            for j in range(self.coresnumber):
                self.register_parameter('cores_{}{}'.format(i, j), self.cores[i][j])
        #self.register_parameter('cores_{}'.format(index), self.cores[index])
#unify code:
        for i in range(self.convnumbers):
            for j in range(self.coresnumber):
                self.cores[i][j].data.uniform_(-0.77, 0.77)

      #  self.Tensor_weights = tt_to_tensor(self.cores) #注意之前的数据类型一定要是tl
    #下一步定义前向函数
    def forward(self, x, stride, pad):
        kenerl = tt_to_tensor(self.cores[0])
        #print('grad1', kenerl.requires_grad)
        kenerl = torch.unsqueeze(kenerl, 0)
        #print('grad2', kenerl.requires_grad)
        for i in range(1, self.convnumbers):
            kenerl1 = tt_to_tensor(self.cores[i])
            kenerl1 = torch.unsqueeze(kenerl1, 0)
            kenerl = torch.cat((kenerl, kenerl1), 0)
        #print('grad3', kenerl.requires_grad)
        outfort = myconv2d(x, kenerl, self.bias, stride, pad)
        #print('grad4', outfort.requires_grad)
        return outfort

class student_MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(student_MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

class student_MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(student_MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.te1 = self._make_te(time_emb_dim, 1)
        self.conv1d = nn.Conv2d(1, 3, 5, 1, 2)
        self.te2 = self._make_te(time_emb_dim, 3)
        self.TTconv2d = ttconv(ranks=(3, 3), each_core_dim=(3, 3, 3), convnumbers=3)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.te3 = self._make_te(time_emb_dim, 3)
        self.TTconv3d = ttconv(ranks=(3, 3), each_core_dim=(3, 3, 3), convnumbers=1)
        # self.te3 = self._make_te(time_emb_dim, 16)

        # self.norm = nn.BatchNorm2d()
        self.norm1 = nn.BatchNorm2d(1)
        self.norm2 = nn.BatchNorm2d(3)
        self.norm3 = nn.BatchNorm2d(3)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        #print('x2.sahpe', x.shape)
        t = self.time_embed(t)
        n = len(x)

        #x = self.norm1(x)
        x = F.relu(self.conv1d(x + self.te1(t).reshape(n, -1, 1, 1)))
        #print('xk1=', x.shape)
        #x = self.norm2(x)
        x = F.relu(self.TTconv2d(x + self.te2(t).reshape(n, -1, 1, 1), 1, 1))
        #x = self.norm3(x)
        #print('xk2=', x.shape)
        x = F.relu(self.TTconv3d(x + self.te3(t).reshape(n, -1, 1, 1), 1, 1))
        #print('xk3=', x.shape)
        return x
        #return out


    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
