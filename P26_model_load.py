import torch
import torchvision
from torch import nn
from P26_model_save import *


# Load Method 1 - Save Method 1
model = torch.load("vgg16_method1.pth")
# print(model)


# Load Method 2
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)


# Traps
# class Tudui(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
# No need for: tudui = Tudui()

model = torch.load("tudui_method1.pth")
print(model)