import torch
from torchvision import transforms


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 100
LR = 0.0001
IMAGE_SIZE = 224
IMG_CHANNELS = 3
BATCH_SIZE = 64
TRAIN_ROOT = r"E:\GAN's\day_night\train"
TEST_ROOT = r"E:\GAN's\day_night\val" # If available

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])