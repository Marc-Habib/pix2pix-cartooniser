import torch
import torch.nn as nn

class UNetDown(nn.Module):
    """
    A single "downsampling" block for the U-Net encoder:
      - Convolution
      - Optional BatchNorm
      - LeakyReLU activation
      - Possibly dropout
      - Stride=2 to reduce spatial dimension by half
    """
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """
    A single "upsampling" block for the U-Net decoder:
      - Transposed convolution (stride=2) doubles spatial dimension
      - BatchNorm
      - ReLU
      - Skip connection from the encoder
      - Possibly dropout
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # x is the upsampled feature, skip_input is from the encoder
        x = self.model(x)
        # Concatenate across channel dimension
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    """
    Full U-Net Generator from the Pix2Pix paper.
    - 8 downsampling layers
    - 8 upsampling layers with skip connections
    - Final output has 3 channels (for RGB) with Tanh activation
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        
        # Downsampling (encoder) layers
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # Upsampling (decoder) layers
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        # Final layer: transposed conv to get back to 3 channels, Tanh to [-1, +1]
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)   # 1/2
        d2 = self.down2(d1)  # 1/4
        d3 = self.down3(d2)  # 1/8
        d4 = self.down4(d3)  # 1/16
        d5 = self.down5(d4)  # 1/32
        d6 = self.down6(d5)  # 1/64
        d7 = self.down7(d6)  # 1/128
        d8 = self.down8(d7)  # 1/256

        # Decoder
        u1 = self.up1(d8, d7)     # cat => channels double
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        # Final output
        output = self.final(u7)
        return output
