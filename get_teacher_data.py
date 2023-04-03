import torch

from Teacher_ddpm_and_unet import Teacher_MyDDPM, Teacher_MyUNet
from generated_new_datasets import generated_new_datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_steps = 1000
store_path = 'ddpm_model_teacher.pt'


best_model_teacher = Teacher_MyDDPM(Teacher_MyUNet(), n_steps=n_steps, device=device)
best_model_teacher.load_state_dict(torch.load(store_path, map_location=device))
best_model_teacher.eval()
print("Model loaded: Generating new datas")

def get_teacher_data(x_teacher_list, eta_theta_teacher_list):

  x_teacher_list, eta_theta_teacher_list = generated_new_datasets(
     x_list=x_teacher_list,
     eta_theta_list=eta_theta_teacher_list,
     ddpm=best_model_teacher,
     n_samples=128,
     device=device,
   )
  x_teacher_list = x_teacher_list[::-1]
  eta_theta_teacher_list = eta_theta_teacher_list[::-1]
  #idx_teacher_list = [n for n in range(0, 1000)]
  #idx_teacher_list = idx_teacher_list[::-1]
  return x_teacher_list, eta_theta_teacher_list
