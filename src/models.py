# models.py - stores different types of architectures
from .utils import (
  nn, torch
)

class DeblurringSimple(nn.Module):
  def __init__(self, input_channels=3, output_channels=3, kernel_size=5, padding=2, bias=False):
    super().__init__()
    self.convolution = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, bias=bias)

  def forward(self, x):
    return self.convolution(x)


class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, init_features=32):
    super().__init__()
    features = init_features

    self.encoder1 = self._block(in_channels, features)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder2 = self._block(features, features * 2)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder3 = self._block(features * 2, features * 4)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder4 = self._block(features * 4, features * 8)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.bottleneck = self._block(features * 8, features * 16)

    self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
    self.decoder4 = self._block((features * 8) * 2, features * 8)
    self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
    self.decoder3 = self._block((features * 4) * 2, features * 4)
    self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
    self.decoder2 = self._block((features * 2) * 2, features * 2)
    self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
    self.decoder1 = self._block(features * 2, features)

    self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

  def _block(self, in_channels, features):
    return nn.Sequential(
      nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(features),
      nn.ReLU(inplace=True),
      nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(features),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    enc1 = self.encoder1(x)
    enc2 = self.encoder2(self.pool1(enc1))
    enc3 = self.encoder3(self.pool2(enc2))
    enc4 = self.encoder4(self.pool3(enc3))

    bottleneck = self.bottleneck(self.pool4(enc4))

    dec4 = self.upconv4(bottleneck)
    dec4 = torch.cat((dec4, enc4), dim=1)
    dec4 = self.decoder4(dec4)
    dec3 = self.upconv3(dec4)
    dec3 = torch.cat((dec3, enc3), dim=1)
    dec3 = self.decoder3(dec3)
    dec2 = self.upconv2(dec3)
    dec2 = torch.cat((dec2, enc2), dim=1)
    dec2 = self.decoder2(dec2)
    dec1 = self.upconv1(dec2)
    dec1 = torch.cat((dec1, enc1), dim=1)
    dec1 = self.decoder1(dec1)

    return self.conv(dec1)


class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)
    self.relu = nn.ReLU(inplace=True)
    
  def forward(self, x):
    residual = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += residual
    out = self.relu(out)
    return out


class DeblurCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    self.conv5 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    residual = x
    out = self.relu(self.conv1(x))
    out = self.relu(self.conv2(out))
    out = self.relu(self.conv3(out))
    out = self.relu(self.conv4(out))
    out = self.conv5(out)
    out += residual
    return out


class DeblurGANv2Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base, 3, 1, 1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base) for _ in range(8)]
        )
        self.conv2 = nn.Conv2d(base, out_channels, 3, 1, 1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.conv2(x)
        return torch.sigmoid(x + residual)