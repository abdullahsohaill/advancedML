import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim, pretrained=False):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.mu = nn.Linear(in_features, latent_dim)
        self.logvar = nn.Linear(in_features, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, input_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, input_size, input_size)
            conv_out = self.conv(sample)
            self.conv_out_shape = conv_out.shape[1:]
            in_features = conv_out.numel() // sample.size(0)
        self.mu = nn.Linear(in_features, latent_dim)
        self.logvar = nn.Linear(in_features, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, conv_out_shape, output_channels=3, target_size=64):
        super().__init__()
        self.conv_out_shape = conv_out_shape  # e.g. (256,4,4) or (512,1,1)
        in_features = int(np.prod(conv_out_shape))
        self.fc = nn.Linear(latent_dim, in_features)

        C, H, W = conv_out_shape
        size = H
        channels = C

        layers = []
        while size < target_size:
            out_channels = max(channels // 2, 32)  # progressively reduce channels
            layers.append(nn.ConvTranspose2d(channels, out_channels, 4, 2, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
            channels = out_channels
            size *= 2

        layers.append(nn.Conv2d(channels, output_channels, 3, 1, 1))
        layers.append(nn.Sigmoid())  # keep output in [0,1]

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), *self.conv_out_shape)
        return self.deconv(x)


class VAE(nn.Module):
    def __init__(self, in_channel, latent_dim, input_size, encoder_type="conv", resnet_pretrained=True, freeze_resnet=False):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        if self.encoder_type == "resnet":
            self.encoder = ResNetEncoder(latent_dim, pretrained=resnet_pretrained)
            with torch.no_grad():
                sample = torch.zeros(1, in_channel, input_size, input_size)
                feat = self.encoder.features(sample)
                conv_out_shape = feat.shape[1:]
            if freeze_resnet:
                for p in self.encoder.features.parameters():
                    p.requires_grad = False
        else:
            self.encoder = Encoder(in_channel, latent_dim, input_size)
            conv_out_shape = self.encoder.conv_out_shape
        self.decoder = Decoder(latent_dim, conv_out_shape)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
