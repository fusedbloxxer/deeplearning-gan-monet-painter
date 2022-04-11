from torch import nn
import torch


class Generator(nn.Module):
    def __init__(self, noise: int, features: int, channels: int, bias: bool, device: torch.device, activ_fun: nn.Module):
        super().__init__()

        # Parameters
        self.features = features

        # Fully-Connected Layer
        self.fc = nn.Linear(noise, features * 4 * 4, bias, device)

        # Convolutional Layers
        self.network = nn.Sequential()

        # Add multiple conv -> batch -> relu layers
        for layer in range(6):
            # Create intermediary layer
            convt = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(
                    in_channels=self.features // (2 ** layer),
                    out_channels=self.features // (2 ** layer),
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                    bias=bias,
                    device=device
                ),
                activ_fun,
                nn.BatchNorm2d(
                    num_features=self.features // (2 ** layer),
                    device=device
                ),
                nn.Conv2d(
                    in_channels=self.features // (2 ** layer),
                    out_channels=self.features // (2 ** layer),
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                    bias=bias,
                    device=device
                ),
                activ_fun,
                nn.BatchNorm2d(
                    num_features=self.features // (2 ** layer),
                    device=device
                ),
                nn.Conv2d(
                    in_channels=self.features // (2 ** layer),
                    out_channels=self.features // (2 ** (layer + 1)),
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                    bias=bias,
                    device=device
                ),
                activ_fun,
                nn.BatchNorm2d(
                    num_features=self.features // (2 ** (layer + 1)),
                    device=device
                ),
            )

            # Append layer to network
            self.network.append(convt)

        # Add final layer without BatchNorm and use Tanh
        self.network.append(nn.Sequential(
            nn.Conv2d(
                in_channels=self.features // (2 ** 6),
                out_channels=channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=bias,
                device=device
            ),
            nn.Tanh()
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input of size (Batch, NoiseLen)
        out = self.fc(x)

        # Transform to img of size (Batch, Features, Height, Width)
        out = out.view(-1, self.features, 4, 4)

        # Apply the transposed convolutional layers
        out = self.network(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_chan: int, features: int, channels: int, device: torch.device, bias: bool, activ_fun: nn.Module):
        super().__init__()

        # Create the network using conv layers
        self.network = nn.Sequential()

        # Add initial layer without BatchNorm
        self.network.append(nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=features,
                kernel_size=(3, 3),
                device=device,
                bias=bias,
                stride=2,
                padding=1,
            ),
            activ_fun,
        ))

        # Add each internediary conv layer
        for layer in range(6):
            # Create intermediary layer
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=features * (2 ** layer),
                    out_channels=features * (2 ** (layer + 1)),
                    kernel_size=(3, 3),
                    device=device,
                    bias=bias,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(
                    features * (2 ** (layer + 1)),
                    device=device
                ),
                activ_fun,
            )

            # Append layer to network
            self.network.append(conv)

        # Add final convolutional layer using Signmoid
        self.network.append(nn.Sequential(
            nn.Conv2d(
                in_channels=features * (2 ** 6),
                out_channels=channels,
                kernel_size=(2, 2),
                device=device,
                bias=bias,
                stride=2,
            ),
            nn.Sigmoid()
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
