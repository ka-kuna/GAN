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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.STL10(root="./data",
                                          split="unlabeled",
                                          download=True,
                                          transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                               shuffle=True, num_workers=4, pin_memory=True)
    return train_loader

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 6, 1, 0), # 6x6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 2, 2, 0), # 12x12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 2, 2, 0), # 24x24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 2, 2, 0), # 48x48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 2, 2, 0), #96x96
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True), #96x96

            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True), #48x48

            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True), #24x24

            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True), #12x12

            nn.AvgPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True), #6x6

            nn.AvgPool2d(6),
            nn.Conv2d(512, 1, 1) # fcの代わり
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
    dataset = load_datasets()
    for i in range(300):
        log_loss_G, log_loss_D = [], []
        for real_img, _ in tqdm(dataset):
            batch_len = len(real_img)

            # Gの訓練
            # 偽画像を作成
            z = torch.randn(batch_len, 128, 1, 1).to(device)
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
        if not os.path.exists("stl_generated"):
            os.mkdir("stl_generated")
        torchvision.utils.save_image(fake_img_tensor[:min(batch_len, 100)],
                                f"stl_generated/epoch_{i:03}.png")
    # ログの保存
    with open("stl_generated/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)

if __name__ == "__main__":
    train()