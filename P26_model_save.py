import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16()
# Save Method 1, structure + parameters
torch.save(vgg16, "vgg16_method1.pth")


# Save Method 2, parameters
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# Traps
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")