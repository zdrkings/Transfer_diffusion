import torch

from get_teacher_data import get_teacher_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_steps = 1000
lr = 0.0001


x_teacher_list = []
eta_theta_teacher_list = []

x_teacher_list, eta_theta_teacher_list = get_teacher_data(x_teacher_list, eta_theta_teacher_list)
test1 = torch.stack(x_teacher_list[1], 0)
test2 = torch.stack(x_teacher_list[998], 0)#用于表示不同时刻的图片(一个batch的)
print(test1[0].shape)
print(test2.shape)