import torch
import torchvision
from PIL import Image
from torch import nn


train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor())


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image_path = "./imgs/airplane.png"
image = Image.open(image_path)
# print(image)

image = image.convert("RGB") # Convert to 3 channels

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
# print(image.shape)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("tudui_29_gpu.pth") # if no gpu add: map_location=torch.device('cpu')
# print(model)

image = image.to(device) # MUST NEED THIS else error
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)

# print(output)
idx = output.argmax(1).item()
key = list(filter(lambda x: train_data.class_to_idx[x] == idx, train_data.class_to_idx))[0]
print(key)
