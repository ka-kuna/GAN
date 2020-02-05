import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os
import pickle
import statistics

def load_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(root="./data",
                                          train=True, download=True,
                                          transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                               shuffle=True, num_workers=4)
    return train_loader

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 1, 0), # 4x4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 1, 0), #7x7
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 2, 2, 0), #14x14
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 1, 2, 2, 0), #28x28
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AvgPool2d(7),
            nn.Conv2d(64, 1, 1) # fcの代わり
        )

    def forward(self, x):
        return self.model(x).squeeze()


def train():
    # モデル
    device = "cuda"
    model_G, model_D = Generator(), Discriminator()
    model_G, model_D = nn.DataParallel(model_G), nn.DataParallel(model_D)
    model_G, model_D = model_G.to(device), model_D.to(device)

    params_G = torch.optim.Adam(model_G.parameters(),
                lr=0.0002, betas=(0.5, 0.999))
    params_D = torch.optim.Adam(model_D.parameters(),
                lr=0.0002, betas=(0.5, 0.999))
    
    # ロスを計算するためのラベル変数
    ones = torch.ones(512).to(device)
    zeros = torch.zeros(512).to(device)
    loss_f = nn.BCEWithLogitsLoss()

    # エラー推移
    result = {}
    result["log_loss_G"] = []
    result["log_loss_D"] = []

    # 訓練
    for i in range(100):
        log_loss_G, log_loss_D = [], []
        for real_img, _ in tqdm(load_datasets()):
            batch_len = len(real_img)

            # Gの訓練
            # 偽画像を作成
            z = torch.randn(batch_len, 64, 1, 1).to(device)
            fake_img = model_G(z)

            # 偽画像を一時保存
            fake_img_tensor = fake_img.detach()

            # 偽画像を本物と騙せるようにロスを計算
            out = model_D(fake_img)
            loss_G = loss_f(out, ones[:batch_len])
            log_loss_G.append(loss_G.item())

            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_G.backward()
            params_G.step()

            # Discriminatoの訓練
            # sample_dataの実画像
            real_img = real_img.to(device)
            # 実画像を実画像と識別できるようにロスを計算
            real_out = model_D(real_img)
            loss_D_real = loss_f(real_out, ones[:batch_len])

            # 偽の画像の偽と識別できるようにロスを計算
            fake_out = model_D(fake_img_tensor)
            loss_D_fake = loss_f(fake_out, zeros[:batch_len])

            # 実画像と偽画像のロスを合計
            loss_D = loss_D_real + loss_D_fake
            log_loss_D.append(loss_D.item())

            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_D.backward()
            params_D.step()

        result["log_loss_G"].append(statistics.mean(log_loss_G))
        result["log_loss_D"].append(statistics.mean(log_loss_D))
        print("log_loss_G =", result["log_loss_G"][-1], ", log_loss_D =", result["log_loss_D"][-1])

        # 画像を保存
        if not os.path.exists("mnist_generated"):
            os.mkdir("mnist_generated")
        torchvision.utils.save_image(fake_img_tensor[:min(batch_len, 100)],
                                f"mnist_generated/epoch_{i:03}.png")
    # ログの保存
    with open("mnist_generated/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)

def save_test():
    for X, _ in load_datasets():
        torchvision.utils.save_image(X[:100], "mnist_generated/true.png")
        break

if __name__ == "__main__":
    train()