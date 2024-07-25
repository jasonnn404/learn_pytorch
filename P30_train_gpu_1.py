import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from P27_model import *
import time

# Load dataset
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"Length of train dataset: {train_data_size}")
print(f"Length of test dataset: {test_data_size}")

# Using DataLoader to load datasets
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# Build Neural Network
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

tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

# Construct loss function
loss_fn = nn.CrossEntropyLoss() # good for Classification
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()


# Optimizer
# learning_rate = 0.01
# 1e-2 = 1 x (10)^(-2) = 1 / 100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# Set up Training Network's parameters
# Track train steps
total_train_step = 0
# Track test steps
total_test_step = 0
# Training times(count)
epoch = 10

# Add tensorboard
writer = SummaryWriter("./logs_train")

start_time = time.time()
for i in range(epoch):
    print(f"----------Training Cycle {i+1} Starting----------")

    # Training begin
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # Optimizing model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(f"Training count: {total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # Testing begin
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"Total test loss: {total_test_loss}")
    print(f"Total test accuracy: {total_accuracy / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # Save current epoch model
    torch.save(tudui, f"tudui_{i}.pth")
    # torch.save(tudui.state_dict(), f"tudui_{}.pth")
    print("Model saved")

writer.close()