import torch
from torch import nn
from torch.optim import Adam

from Student_ddpm_and_unet import student_MyUNet, student_MyDDPM
from get_teacher_data import get_teacher_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_steps = 1000
lr = 0.001
store_path = 'ddpm_mnist_student.pt'
'''
x_teacher_list = []
eta_theta_teacher_list = []

x_teacher_list, eta_theta_teacher_list = get_teacher_data(x_teacher_list, eta_theta_teacher_list)
test1 = torch.stack(x_teacher_list[1], 0)
test2 = torch.stack(x_teacher_list[998], 0)#用于表示不同时刻的图片(一个batch的)
print(test1[0].shape)
print(test2.shape)
print(len(x_teacher_list[1]))
'''
best_model_student = student_MyDDPM(student_MyUNet(), n_steps=n_steps, device=device)
best_model_student.load_state_dict(torch.load(store_path, map_location=device))
best_model_student.train()

def training_loop(ddpm,  n_epochs, optim, device):
  mse = nn.MSELoss()
  for epoch in range(n_epochs):
    epoch_loss = 0.0
    x_teacher_list = []
    eta_theta_teacher_list = []
    x_teacher_list, eta_theta_teacher_list = get_teacher_data(x_teacher_list, eta_theta_teacher_list)
    for i in range(len(x_teacher_list)-1, 0, -1): #应该反着打印,从而实现一步教一步
      x = torch.stack(x_teacher_list[i], 0).to(device)
      t = i
      time_tensor = (torch.ones(len(x_teacher_list[i]), 1) * t).to(device).long()
      eta_student = ddpm.backward(x, time_tensor)
      eta_teacher = torch.stack(eta_theta_teacher_list[i], 0)
      loss = mse(eta_student, eta_teacher)
      optim.zero_grad()
      loss.backward()
      optim.step()
      epoch_loss += loss.item() / len(x_teacher_list)
    log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"
    print(log_string)

training_loop(ddpm=best_model_student, n_epochs=20, optim=Adam(best_model_student.parameters(), lr=lr), device=device)

#如何保存一个好一点的模型，然后生成图片


