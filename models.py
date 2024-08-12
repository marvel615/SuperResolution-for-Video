import torch
from torch import nn
import utils
from functools import partial

##########################################################
# ZSSR Model
##########################################################
class ZSSRResNet(nn.Module):
  """A super resolution model. """

  def __init__(self, scale_factor, kernel_size=3):
    """ Trains a ZSSR model on a specific image.
    Args:
      scale_factor (int): ratio between SR and LR image sizes.
      kernel_size (int): size of kernels to use in convolutions.
    """

    super().__init__()

    self.scale_factor = scale_factor
    self.kernel_size = kernel_size

    self.layers = nn.ModuleList()

    #1st layer
    self.layers.append(nn.Conv2d(3, 64, self.kernel_size, stride=1, padding=1))

    #2-7 layers
    for _ in range(1, 7):
      self.layers.append(nn.Conv2d(64, 64, self.kernel_size, stride=1, padding=1))

    #8 layer
    self.output = nn.ConvTranspose2d(64, 3, self.kernel_size, stride=self.scale_factor,
    padding=1, output_padding=1)



  def forward(self, x):
    """ Apply super resolution on an image.
    First, resize the input image using `utils.rr_resize`.
    Then pass the image through your CNN.
    Finally, add the CNN's output in a residual manner to the original resized
    image.
    Args:
      x (torch.Tensor): LR input.
      Has shape `(batch_size, num_channels, height, width)`.

    Returns:
      output (torch.Tensor): HR input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """

    x_start = utils.rr_resize(x, scale_factors=self.scale_factor)


    # layers 1-7
    for layer in self.layers:
      x = layer(x)
      x = nn.ReLU()(x)

    x = self.output(x) # layer 8

    x_end = x + x_start

    return x_end

##########################################################
# U-Net like Model
##########################################################

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UpConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpConvBlock3D, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SuperResolutionUNet(nn.Module):
    def __init__(self):
        super(SuperResolutionUNet, self).__init__()
        self.enc1 = ConvBlock3D(3, 16)
        self.enc2 = ConvBlock3D(16, 32)
        self.enc3 = ConvBlock3D(32, 64)

        self.pool = nn.MaxPool3d(kernel_size=(1,2,2))

        self.dec2 = UpConvBlock3D(64+32, 32)
        self.dec1 = nn.ConvTranspose3d(32+16, 3, kernel_size=3, stride=1, padding=1)

        self.upsample = nn.Upsample(scale_factor=(1,2,2), mode='nearest')

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B, 3, T, H, W] -> [B, 16, T, H, W]
        e2 = self.pool(e1) # [B, 16, T, H, W] -> [B, 16, T, H/2, W/2]
        e2 = self.enc2(e2) # [B, 16, T, H/2, W/2] -> [B, 32, T, H/2, W/2]
        e3 = self.pool(e2) # [B, 32, T, H/2, W/2] -> [B, 32, T, H/4, W/4]
        e3 = self.enc3(e3) # [B, 32, T, H/4, W/4] -> [B, 64, T, H/4, W/4]

        # Decoder
        d2 = self.upsample(e3) # [B, 64, T, H/4, W/4] -> [B, 64, T, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1) # [B, 64, T, H/2, W/2] -> [B, 96, T, H/2, W/2]
        d1 = self.dec2(d2) # [B, 96, T, H/2, W/2] -> [B, 32, T, H/2, W/2]
        d1 = self.upsample(d1) # [B, 32, T, H/2, W/2] -> [B, 32, T, H, W]
        d1 = torch.cat([d1, e1], dim=1) # [B, 32, T, H, W] -> [B, 48, T, H, W]
        out = self.dec1(d1) # [B, 48, T, H, W] -> [B, 3, T, H, W]

        return out
