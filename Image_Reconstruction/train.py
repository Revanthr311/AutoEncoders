import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import config
from model import AutoEncoder
from torchvision.utils import save_image


def imshow(img, title):
    npimg = img.cpu().numpy()  # Move tensor to CPU before converting to NumPy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    print("Loading Data...")

    train_data = datasets.ImageFolder(root=config.TRAIN_ROOT, transform=config.TRANSFORM)
    train_dl = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    print("===========================")
    print("Creating Model Instances...")

    model = AutoEncoder(config.IMG_CHANNELS).to(config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr= config.LR)

    print("===============================")
    print("Training the model...")

    for epoch in range(config.EPOCH):
        for real, _ in tqdm(train_dl):
            real = real.to(config.DEVICE)

            encoded_img, decoded_img = model(real)
            loss = criterion(decoded_img, real)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("========================")
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        if epoch % 10 == 0: 
            imshow(torchvision.utils.make_grid(real[:32], padding=2, normalize=True), 'Real_Image')

            #imshow(torchvision.utils.make_grid(encoded_img[:32], padding=2, normalize=True), 'Encoded_Image')

            imshow(torchvision.utils.make_grid(decoded_img[:32], padding=2, normalize=True), 'Decoded_Image')



