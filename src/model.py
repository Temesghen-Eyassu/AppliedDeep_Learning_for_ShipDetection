# PyTorch and supporting libraries
import torch                                            # Main PyTorch package, provides Tensors
import torch.nn as nn                                   # Neural network modules (Conv2d, BatchNorm2d, ReLU, Sequential, ModuleList)
import torch.nn.functional as F                         # Stateless functions like max_pool2d, interpolate, relu


# U-Net class definition

class UNet(nn.Module):
    """
    U-Net model for semantic segmentation (e.g., ship detection).
    Encoder-decoder architecture with skip connections.
    Input: (B, in_channels, H, W)
    Output: (B, out_channels, H, W)
    """


    # Constructor: define layers/modules

    def __init__(self, in_channels=2, out_channels=1, features=[32,64,128]):
        """
        in_channels: number of channels in input images ( 2 for VV+VH)
        out_channels: number of channels in output masks (1 for binary segmentation)
        features: list of channel sizes for each encoder block
        """
        super().__init__()                                  # Initialize parent nn.Module
        self.encoder = nn.ModuleList()                      # for encoder blocks
        self.decoder = nn.ModuleList()                      # for decoder blocks


        # Encoder blocks (downsampling path)
        for f in features:
            self.encoder.append(self._block(in_channels, f))  # Add conv block
            in_channels = f                                   # Update input channels for next block
            # Example shapes for input 2x64x64:
            # Block1: 2x64x64  - 32x64x64
            # Block2: 32x32x32 - 64x32x32
            # Block3: 64x16x16 - 128x16x16


        # Bottleneck (deepest layer)
        self.bottleneck = self._block(features[-1], features[-1]*2)                # Shape: 128x8x8 - 256x8x8


        # Decoder blocks (upsampling path)

        for f in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))  # Upsample spatially
            self.decoder.append(self._block(f*2, f))                                  # Conv block after concatenation
            # Example shapes:
            # Up1: ConvTranspose2d(256-128) - 128x16x16 - concat with skip1 128x16x16 - 256x16x16 - block - 128x16x16
            # Up2: ConvTranspose2d(128→64)  - 64x32x32  - concat with skip2 64x32x32  - 128x32x32 - block - 64x32x32
            # Up3: ConvTranspose2d(64→32)   - 32x64x64  - concat with skip3 32x64x64  - 64x64x64  - block - 32x64x64


        # Final convolution to reduce channels to out_channels

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
        # Shape: 32x64x64 → 1x64x64


    # Forward method: defines computation for input tensor

    def forward(self, x):
        """
        x: input tensor (B, in_channels, H, W)
        returns: output tensor (B, out_channels, H, W)
        """
        skips = []  # List to store skip connections


        # Encoder forward pass

        for down in self.encoder:
            x = down(x)                             # Apply conv block
            skips.append(x)                         # Save output for skip connection
            x = F.max_pool2d(x, 2)        # Downsample spatial dims by 2
            # Shape examples:
            # Block1: 2x64x64  - 32x64x64  - pool - 32x32x32
            # Block2: 32x32x32 - 64x32x32  - pool - 64x16x16
            # Block3: 64x16x16 - 128x16x16 - pool - 128x8x8


        # Bottleneck

        x = self.bottleneck(x)                                # 128x8x8 - 256x8x8

        skips = skips[::-1]                                   # Reverse skip list for decoder alignment


        # Decoder forward pass

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)                          # Upsample (ConvTranspose2d)
            skip = skips[idx//2]                              # Corresponding skip
            if x.shape != skip.shape:                         # Adjust spatial dims if mismatch
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)           # Concatenate along channel dimension
            x = self.decoder[idx+1](x)                        # Apply conv block
            # Shape examples:
            # Up1: 256x8x8   - 128x16x16  - concat skip1 128x16x16 - 256x16x16 - block - 128x16x16
            # Up2: 128x16x16 - 64x32x32   - concat skip2 64x32x32  - 128x32x32 - block - 64x32x32
            # Up3: 64x32x32  - 32x64x64   - concat skip3 32x64x64  - 64x64x64  - block - 32x64x64

        # Final output

        return self.final(x)                                             # 32x64x64 - 1x64x64 (binary segmentation mask)


    # Conv block helper: Conv - BN - ReLU - Conv - BN - ReLU

    def _block(self, in_ch, out_ch):
        """
        Constructs a convolutional block.
        Each block: Conv2d - BatchNorm2d - ReLU - Conv2d - BatchNorm2d - ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),  # preserves spatial dims
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), # preserves spatial dims
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
