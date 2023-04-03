import random
import imageio
import numpy as np
from argparse import ArgumentParser

from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import FashionMNIST, MNIST
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
#from torchvision.datasets.mnist import MNIST, FashionMNIST
import tensorly as tl
import torch.nn.functional as F

from Student_ddpm_and_unet import student_MyDDPM, student_MyUNet
from generated_new_images import generated_new_images
from show_images import show_images
from sinusoidal_embedding import sinusoidal_embedding

#from Redataset import MyDataset
#from mnist_diffusion import Teacher_MyDDPM, Teacher_MyUNet, generated_new_images

tl.set_backend('pytorch')
from tensorly import tt_to_tensor

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions

store_path = "ddpm_mnist_student.pt"
no_train = False
fashion = False
batch_size = 128
n_epochs = 20
lr = 0.001


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)] #需要再理解一下归一化
)
ds_fn = FashionMNIST if fashion else MNIST
dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
loader = DataLoader(dataset, batch_size, shuffle=True)


def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path = store_path):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")): #tqdm表示的是显示进程的进度条，enumerate关键字用于输出不同的batch以及一个batch对应的指标是1，2，3，4
            # Loading data
            x0 = batch[0].to(device) #注意batch[0]表示的意思是一个batch中的图片，而batch[1]表示的是一个batch中的标签，batch在不断的变化，直到遍历整个数据集
            n = len(x0) #x0表示的是一个batch中的所有图片，其数目为batch_size

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)  #注意eta的维数应当与x0的维数是一样的
            t = torch.randint(0, n_steps, (n,)).to(device)  #t表示的意义是给一个batch里的图片随机生成一个时间t用于进行训练

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta) #用于表示加噪声的过程

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            show_images(generated_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        scheduler_1.step()
        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)

n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = student_MyDDPM(student_MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr, weight_decay=0.01)
scheduler_1 = StepLR(optimizer, step_size=3, gamma=0.75)
training_loop(ddpm, loader, n_epochs, optim=optimizer, device=device, store_path=store_path)
