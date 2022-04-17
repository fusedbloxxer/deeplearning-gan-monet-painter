from torch import distributions as distr
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class Generator(Model):
    def __init__(self, in_dim: torch.Size, hidden_dim: torch.Size,
                 out_dim: torch.Size, activ_fun: nn.Module,
                 distrib: distr.Distribution = None, bias: bool = True):
        super().__init__()

        # Initialize optional parameters
        if distrib is None:
            distrib = distr.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

        # Initialize members
        self.distrib = distrib

        # Channel sizes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # Activation function
        self.activ_fun = activ_fun
        self.out_fun = nn.Tanh()

        # Layers
        self.in_fc = nn.Linear(in_dim.numel(), hidden_dim.numel(), bias=bias)
        self.tconv1 = nn.ConvTranspose2d(hidden_dim[0], hidden_dim[0] // 2, 4, 2, 1, bias=bias)
        self.tconv2 = nn.ConvTranspose2d(hidden_dim[0] // 2, hidden_dim[0] // 4, 4, 2, 1, bias=bias)
        self.tconv3 = nn.ConvTranspose2d(hidden_dim[0] // 4, hidden_dim[0] // 8, 4, 2, 1, bias=bias)
        self.tconv4 = nn.ConvTranspose2d(hidden_dim[0] // 8, hidden_dim[0] // 16, 4, 2, 1, bias=bias)
        self.tconv5 = nn.ConvTranspose2d(hidden_dim[0] // 16, hidden_dim[0] // 32, 4, 2, 1, bias=bias)
        self.tconv6 = nn.ConvTranspose2d(hidden_dim[0] // 32, hidden_dim[0] // 64, 4, 2, 1, bias=bias)
        self.tconv7 = nn.ConvTranspose2d(hidden_dim[0] // 64, out_dim[0], 3, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(hidden_dim[0] // 2)
        self.bn2 = nn.BatchNorm2d(hidden_dim[0] // 4)
        self.bn3 = nn.BatchNorm2d(hidden_dim[0] // 8)
        self.bn4 = nn.BatchNorm2d(hidden_dim[0] // 16)
        self.bn5 = nn.BatchNorm2d(hidden_dim[0] // 32)
        self.bn6 = nn.BatchNorm2d(hidden_dim[0] // 64)
        self.pool1 = nn.AdaptiveAvgPool2d(out_dim[1:])

    def generate(self, n_samples: torch.Size = torch.Size((16,))) -> torch.Tensor:
        samples_size = torch.Size((n_samples.numel(), self.in_dim.numel()))
        noise = self.distrib.sample(samples_size)
        noise = noise.to(self.device).squeeze(-1)
        return self(noise)

    # def generate(self, n_samples: torch.Size = torch.Size((16,))) -> torch.Tensor:
    #     return torch.randn((16, 3, 256, 256)).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.in_fc(x)
        out = out.view((-1, *self.hidden_dim[:]))
        out = self.activ_fun(self.bn1(self.tconv1(out)))
        out = self.activ_fun(self.bn2(self.tconv2(out)))
        out = self.activ_fun(self.bn3(self.tconv3(out)))
        out = self.activ_fun(self.bn4(self.tconv4(out)))
        out = self.activ_fun(self.bn5(self.tconv5(out)))
        out = self.activ_fun(self.bn6(self.tconv6(out)))
        out = self.activ_fun(self.tconv7(out))
        out = self.pool1(out)
        out = self.out_fun(out)
        out = out.view((-1, *self.out_dim[:]))
        return out

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return torch.randn((16, 3, 256, 256)).to(self.device)


class Discriminator(Model):
    def __init__(self, in_dim: torch.Size, hidden_dim: torch.Size,
                 out_dim: torch.Size, activ_fun: nn.Module, bias: bool = True):
        super().__init__()

        # Channel sizes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # Activation function
        self.activ_fun = activ_fun
        self.out_fun = nn.Sigmoid()

        # Layers
        self.conv1 = nn.Conv2d(in_dim[0], hidden_dim[0], 3, 2, 0, bias=bias)
        self.conv2 = nn.Conv2d(hidden_dim[0], hidden_dim[0] * 2, 3, 2, 0, bias=bias)
        self.conv3 = nn.Conv2d(hidden_dim[0] * 2, hidden_dim[0] * 4, 3, 2, 0, bias=bias)
        self.conv4 = nn.Conv2d(hidden_dim[0] * 4, hidden_dim[0] * 8, 3, 2, 0, bias=bias)
        self.conv5 = nn.Conv2d(hidden_dim[0] * 8, hidden_dim[0] * 16, 3, 2, 0, bias=bias)
        self.conv6 = nn.Conv2d(hidden_dim[0] * 16, out_dim[0], 3, 2, 0, bias=bias)
        self.bn1 = nn.BatchNorm2d(hidden_dim[0])
        self.bn2 = nn.BatchNorm2d(hidden_dim[0] * 2)
        self.bn3 = nn.BatchNorm2d(hidden_dim[0] * 4)
        self.bn4 = nn.BatchNorm2d(hidden_dim[0] * 8)
        self.bn5 = nn.BatchNorm2d(hidden_dim[0] * 16)
        self.pool1 = nn.AdaptiveAvgPool2d(out_dim[:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activ_fun(self.bn1(self.conv1(x)))
        out = self.activ_fun(self.bn2(self.conv2(out)))
        out = self.activ_fun(self.bn3(self.conv3(out)))
        out = self.activ_fun(self.bn4(self.conv4(out)))
        out = self.activ_fun(self.bn5(self.conv5(out)))
        out = self.activ_fun(self.conv6(out))
        out = self.pool1(out)
        out = self.out_fun(out)
        out = out.view((-1,))
        return out


class GAN(Model):
    def __init__(self, generator: Generator, discriminator: Discriminator):
        super(GAN, self).__init__()
        self.G = generator
        self.D = discriminator

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.D(x)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.G(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.D(self.G(z))
