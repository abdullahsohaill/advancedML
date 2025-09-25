import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=128, base_channels=64, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 4 * 4)

        self.conv1 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels * 4)

        self.conv2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)

        self.conv3 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels)

        self.conv4 = nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(base_channels // 2)

        self.final = nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, 4, 4)
        x = self.upsample(self.relu(self.bn1(self.conv1(x))))   # 4 -> 8
        x = self.upsample(self.relu(self.bn2(self.conv2(x))))   # 8 -> 16
        x = self.upsample(self.relu(self.bn3(self.conv3(x))))   # 16 -> 32
        x = self.upsample(self.relu(self.bn4(self.conv4(x))))   # 32 -> 64
        x = self.final(x)
        return self.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)

        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)

        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, 1)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x).view(-1)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
