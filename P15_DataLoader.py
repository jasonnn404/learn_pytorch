import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Test dataset
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# Test dataset's first image and target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images(f"Epoch: {epoch}", imgs, step)
        step += 1

writer.close()