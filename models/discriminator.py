import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator:
    - Takes in (real_image + cartoon_image) or (real_image + fake_cartoon).
    - Outputs a grid of real/fake predictions (N x 1 x H' x W').
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        # in_channels*2 => because we concatenate (input, target) along channels
        def conv_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 1) input is (in_channels*2) => e.g. 3 + 3 = 6
            *conv_block(in_channels*2, 64, normalize=False),
            # 2)
            *conv_block(64, 128),
            # 3)
            *conv_block(128, 256),
            # Now we reduce stride:
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, 4, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Final output: 1-channel "patch" output
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, 4, padding=1, bias=False) 
        )

    def forward(self, img_input, img_target):
        # Concatenate along channel dimension
        x = torch.cat((img_input, img_target), 1)
        return self.model(x)
