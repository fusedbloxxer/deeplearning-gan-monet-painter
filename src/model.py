from torch import distributions as distr
from torch.nn import functional
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class GenerativeBlock(nn.Module):
    def __init__(self, in_chan: int, num_ops: int, activ_fun: nn.Module,
                 bias: bool = True, batchnorm: bool = True,
                 hidden_chan: int = None, out_chan: int = None,
                 step_down: bool = False):
        super(GenerativeBlock, self).__init__()

        # Initialize configurable parameters
        self.factor = 1 if not step_down else 2
        self.activ_fun = activ_fun
        self.batchnorm = batchnorm
        self.bias = bias

        # Transform the input parameters
        if num_ops < 1:
            raise ValueError('num_ops must be at least 1')

        if step_down and in_chan // 2 ** num_ops < 1:
            raise ValueError(f'in_chan must be divisible by 2 ** num_ops')

        if out_chan is None:
            out_chan = in_chan // 2 ** (num_ops if step_down else 1)

        if hidden_chan is None:
            hidden_chan = in_chan // self.factor ** 1

        # Retain the dimensions of the input, output and hidden layers
        self.hidden_chan = hidden_chan
        self.out_chan = out_chan
        self.in_chan = in_chan

        # Initialize the layers
        self.layers = []

        # Add the layers to the model
        if num_ops == 1:
            self.layers.append(self.make_block(in_chan, out_chan))
        elif num_ops == 2:
            self.layers.append(self.make_block(in_chan, hidden_chan))
            self.layers.append(self.make_block(hidden_chan, out_chan))
        else:
            self.layers.append(self.make_block(in_chan, hidden_chan))
            self.layers.append(self.make_group(num_ops))
            self.layers.append(self.make_block(hidden_chan // self.factor ** (num_ops - 2), out_chan))

        # Create an upsampling block
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)
        self.layers.append(self.upsample)

        # Bundle the layers into a sequential block
        self.layers = nn.Sequential(*self.layers)

    def make_group(self, num_ops: int) -> nn.Module:
        # Retain the blocks in a list
        group = []

        # Initialize the inner layers
        for i in range(num_ops - 2):
            in_chan  = self.hidden_chan // self.factor ** i
            out_chan = self.hidden_chan // self.factor ** (i + 1)
            group.append(self.make_block(in_chan, out_chan))

        # Bundle the group into a sequential block
        group = nn.Sequential(*group)
        return group

    def make_block(self, in_chan: int, out_chan: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, 1, 1, bias=self.bias),
            nn.BatchNorm2d(out_chan) if self.batchnorm else nn.Identity(),
            self.activ_fun,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Generator(Model):
    def __init__(self, in_dim: torch.Size, hidden_dim: torch.Size,
                 out_dim: torch.Size, activ_fun: nn.Module,
                 batchnorm: bool = True, distrib: distr.Distribution = None,
                 bias: bool = True):
        super().__init__()

        # Initialize optional parameters
        if distrib is None:
            distrib = distr.Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))

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
        self.fc1 = nn.Linear(in_dim.numel(), hidden_dim.numel(), bias=bias)
        self.gen1 = GenerativeBlock(hidden_dim[0]     ,  4, activ_fun, bias, batchnorm)
        self.gen2 = GenerativeBlock(hidden_dim[0] // 2,  4, activ_fun, bias, batchnorm)
        self.gen3 = GenerativeBlock(hidden_dim[0] // 4,  4, activ_fun, bias, batchnorm)
        self.gen4 = GenerativeBlock(hidden_dim[0] // 8,  4, activ_fun, bias, batchnorm)
        self.gen5 = GenerativeBlock(hidden_dim[0] // 16, 4, activ_fun, bias, batchnorm)
        self.gen6 = GenerativeBlock(hidden_dim[0] // 32, 4, activ_fun, bias, batchnorm)
        self.conv = nn.Conv2d(hidden_dim[0] // 64, out_dim[0], 3, 1, 1, bias=bias)

    def generate(self, n_samples: torch.Size = torch.Size((16,))) -> torch.Tensor:
        samples_size = torch.Size((n_samples.numel(), self.in_dim.numel()))
        noise = self.distrib.sample(samples_size)
        noise = noise.to(self.device).squeeze(-1)
        return self(noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = out.view((-1, *self.hidden_dim[:]))
        out = self.gen1(out)
        out = self.gen2(out)
        out = self.gen3(out)
        out = self.gen4(out)
        out = self.gen5(out)
        out = self.gen6(out)
        out = self.conv(out)
        out = self.out_fun(out)
        out = out.view((-1, *self.out_dim[:]))
        return out


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
        self.conv1 = nn.Conv2d(in_dim[0], hidden_dim[0], 3, 2, 1, bias=bias)
        self.conv2 = nn.Conv2d(hidden_dim[0], hidden_dim[0] * 2, 3, 2, 1, bias=bias)
        self.conv3 = nn.Conv2d(hidden_dim[0] * 2, hidden_dim[0] * 4, 3, 2, 1, bias=bias)
        self.conv4 = nn.Conv2d(hidden_dim[0] * 4, hidden_dim[0] * 8, 3, 2, 1, bias=bias)
        self.conv5 = nn.Conv2d(hidden_dim[0] * 8, hidden_dim[0] * 16, 3, 2, 1, bias=bias)
        self.conv6 = nn.Conv2d(hidden_dim[0] * 16, hidden_dim[0] * 32, 3, 2, 1, bias=bias)

        self.bn1 = nn.BatchNorm2d(hidden_dim[0])
        self.bn2 = nn.BatchNorm2d(hidden_dim[0] * 2)
        self.bn3 = nn.BatchNorm2d(hidden_dim[0] * 4)
        self.bn4 = nn.BatchNorm2d(hidden_dim[0] * 8)
        self.bn5 = nn.BatchNorm2d(hidden_dim[0] * 16)
        self.pool1 = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(hidden_dim[0] * 32 * 4 * 4, hidden_dim[0] * 128, bias=bias)
        self.fc2 = nn.Linear(hidden_dim[0] * 128, out_dim[0], bias=bias)

        self.bn6 = nn.BatchNorm1d(hidden_dim[0] * 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activ_fun(self.bn1(self.conv1(x)))
        out = self.activ_fun(self.bn2(self.conv2(out)))
        out = self.activ_fun(self.bn3(self.conv3(out)))
        out = self.activ_fun(self.bn4(self.conv4(out)))
        out = self.activ_fun(self.bn5(self.conv5(out)))
        out = self.activ_fun(self.conv6(out))

        out = self.pool1(out)
        out = out.view((-1, self.hidden_dim[0] * 32 * 4 * 4))

        out = self.activ_fun(self.bn6(self.fc1(out)))
        out = self.fc2(out)

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
