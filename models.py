# models.py - stores different types of architectures
from utils import (
  nn, torch
)

class DeblurringSimple(nn.Module):
  def __init__(self, input_channels=3, output_channels=3, kernel_size=5, padding=2, bias=False):
    super().__init__()
    self.convolution = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, bias=bias)
    
    
  def forward(self, x):
    x = self.convolution(x)
    return x


class DoubleConv(nn.Module):
  """(Convolution -> Batch Norm -> ReLU) x 2"""
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)

class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3):
    super(UNet, self).__init__()
    
    # Encoder (Down-sampling)
    self.inc = DoubleConv(in_channels, 64)
    self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
    self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
    self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
    self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

    # Decoder (Up-sampling)
    self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.conv1 = DoubleConv(1024, 512)
    
    self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.conv2 = DoubleConv(512, 256)
    
    self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.conv3 = DoubleConv(256, 128)
    
    self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.conv4 = DoubleConv(128, 64)
    
    # Final layer
    self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    self.final_activation = nn.Sigmoid()

  def forward(self, x):
    # Encoder
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    
    # Decoder with Skip Connections
    x = self.up1(x5)
    x = torch.cat([x4, x], dim=1)
    x = self.conv1(x)
    
    x = self.up2(x)
    x = torch.cat([x3, x], dim=1)
    x = self.conv2(x)
    
    x = self.up3(x)
    x = torch.cat([x2, x], dim=1)
    x = self.conv3(x)
    
    x = self.up4(x)
    x = torch.cat([x1, x], dim=1)
    x = self.conv4(x)
    
    logits = self.outc(x)
    return self.final_activation(logits)


class ResidualBlock(nn.Module):
  """Residual block with two convolutions and skip connection"""
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
  """Multi-layer CNN with residual connections for deblurring"""
  def __init__(self, in_channels=3, out_channels=3, base_filters=64):
    super().__init__()
    
    # Initial feature extraction
    self.conv_in = nn.Sequential(
      nn.Conv2d(in_channels, base_filters, kernel_size=7, padding=3),
      nn.ReLU(inplace=True)
    )
    
    # Deep feature processing with residual blocks
    self.res_blocks = nn.Sequential(
      ResidualBlock(base_filters),
      ResidualBlock(base_filters),
      ResidualBlock(base_filters),
      ResidualBlock(base_filters),
      ResidualBlock(base_filters)
    )
    
    # Reconstruction
    self.conv_out = nn.Sequential(
      nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(base_filters, out_channels, kernel_size=3, padding=1),
      nn.Sigmoid()
    )
    
  def forward(self, x):
    x = self.conv_in(x)
    x = self.res_blocks(x)
    x = self.conv_out(x)
    return x


class AutoEncoder(nn.Module):
  """Simple encoder-decoder architecture for deblurring"""
  def __init__(self, in_channels=3, out_channels=3):
    super().__init__()
    
    # Encoder
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # /2
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # /4
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # /8
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    
    # Decoder
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # *2
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # *2
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # *2
      nn.Sigmoid()
    )
    
  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


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