import einops
import imageio
import numpy as np
import torch

from append_function import append_function


def generated_new_datasets(x_list, eta_theta_list, ddpm, n_samples=16, device=None, c=1, h=28, w=28):

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise x是所产生的随机噪声
        x = torch.randn(n_samples, c, h, w).to(device)
        x_list = append_function(x, x_list, 0)
        '''
        x_list = []
        eta_theta_list = []
        '''
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):  #该操作用于对时间步进行镜像反转
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()  #生成同一个时间t所对应的时间张量
            #x_list = append_function(x, x_list, idx)
            eta_theta = ddpm.backward(x, time_tensor) #调用反向函数计算出此时的均值
            eta_theta_list = append_function(eta_theta, eta_theta_list, idx)
            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta) #注意加减乘除的结合方式
            #if t == 0:
                #x_list = append_function(x, x_list, idx + 1)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt() #这是所计算出来的方差,一共有两种方案可以选择想要的方差

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z
                x_list = append_function(x, x_list, idx+1)


    return x_list, eta_theta_list