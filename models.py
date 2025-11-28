# models.py - stores different types of architectures
from utils import (
  nn
)

class DeblurringSimple(nn.Module):
  def __init__(self, input_channels=3, output_channels=3, kernel_size=5, padding=2, bias=False):
    super().__init__()
    self.convolution = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, bias=bias)
    
    
  def forward(self, x):
    x = self.convolution(x)
    return x

