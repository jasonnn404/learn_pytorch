from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


# Tensor data type
# Through transforms.ToTensor we will address 2 questions
# 1. How to use transforms (python)
# 2. Why we need Tensor data type

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()