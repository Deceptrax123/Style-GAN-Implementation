import torch
from torch.autograd import Variable
from torch.nn import Linear, Conv2d, Module, BatchNorm1d, UpsamplingBilinear2d


class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.f = MappingNetwork()
        self.g1 = SynthesisNetwork(in_channels=512, out_channels=256, noise_shape1=(
            512, 4, 4), noise_shape2=(256, 8, 8))
        self.g2 = SynthesisNetwork(in_channels=256, out_channels=128, noise_shape1=(
            256, 8, 8), noise_shape2=(128, 16, 16))
        self.g3 = SynthesisNetwork(in_channels=128, out_channels=64, noise_shape1=(
            128, 16, 16), noise_shape2=(64, 32, 32))
        self.g4 = SynthesisNetwork(in_channels=64, out_channels=32, noise_shape1=(
            64, 32, 32), noise_shape2=(32, 64, 64))
        self.g5 = SynthesisNetwork(in_channels=32, out_channels=16, noise_shape1=(
            32, 64, 64), noise_shape2=(16, 128, 128))
        self.g6 = SynthesisNetwork(in_channels=16, out_channels=8, noise_shape1=(
            16, 128, 128), noise_shape2=(8, 256, 256))
        self.g7 = SynthesisNetwork(in_channels=8, out_channels=4, noise_shape1=(
            8, 256, 256), noise_shape2=(4, 512, 512))
        self.g8 = SynthesisNetwork(in_channels=4, out_channels=2, noise_shape1=(
            4, 512, 512), noise_shape2=(2, 1024, 1024))

        self.conv = Conv2d(in_channels=2, out_channels=3,
                           kernel_size=(1, 1), stride=1)

    def forward(self, z, constant):
        f = self.f(z)
        x = self.g1(f, constant)
        x = self.g2(f, x)
        x = self.g3(f, x)
        x = self.g4(f, x)
        x = self.g5(f, x)
        x = self.g6(f, x)
        x = self.g7(f, x)
        x = self.g8(f, x)

        x = self.conv(x)

        return x


class MappingNetwork(Module):
    def __init__(self):
        super(MappingNetwork, self).__init__()

        self.bn = BatchNorm1d(512)  # Normalization Layer

        self.linear1 = Linear(in_features=512, out_features=256)
        self.linear2 = Linear(in_features=256, out_features=128)
        self.linear3 = Linear(in_features=128, out_features=64)
        self.linear4 = Linear(in_features=64, out_features=32)
        self.linear5 = Linear(in_features=32, out_features=64)
        self.linear6 = Linear(in_features=64, out_features=128)
        self.linear7 = Linear(in_features=128, out_features=256)
        self.linear8 = Linear(in_features=256, out_features=512)

    def forward(self, x):
        x = self.bn(x)
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu()
        x = self.linear4(x).relu()
        x = self.linear5(x).relu()
        x = self.linear6(x).relu()
        x = self.linear7(x).relu()
        x = self.linear8(x).relu()

        return x


class SynthesisNetwork(Module):
    def __init__(self, in_channels, out_channels, noise_shape1, noise_shape2):
        super(SynthesisNetwork, self).__init__()

        self.ada1 = AdaIn()
        self.ada2 = AdaIn()
        self.ada3 = AdaIn()
        self.ada4 = AdaIn()

        self.upsample = UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = Conv2d(in_channels=in_channels, out_channels=in_channels,
                            stride=1, kernel_size=(3, 3), padding=1)
        self.conv2 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            stride=1, kernel_size=(3, 3), padding=1)
        self.conv3 = Conv2d(in_channels=out_channels, out_channels=out_channels,
                            stride=1, kernel_size=(3, 3), padding=1)

        self.gaussian1 = GaussianNoise(shape=noise_shape1)
        self.gaussian2 = GaussianNoise(shape=noise_shape1)
        self.gaussian3 = GaussianNoise(shape=noise_shape2)
        self.gaussian4 = GaussianNoise(shape=noise_shape2)

    def forward(self, f, const):
        B1 = self.gaussian1().to(device='mps')
        B2 = self.gaussian2().to(device='mps')
        B3 = self.gaussian3().to(device='mps')
        B4 = self.gaussian4().to(device='mps')

        # First Block
        x = torch.add(const, B1)
        x = self.ada1(x, f)

        x = self.conv1(x)
        x = torch.add(x, B2)

        x = self.ada2(x, f)

        # Upsampled Block
        x = self.upsample(x)
        x = self.conv2(x)
        x = torch.add(x, B3)
        x = self.ada3(x, f)
        x = self.conv3(x)
        x = torch.add(x, B4)
        x = self.ada4(x, f)

        return x


class GaussianNoise(Module):
    def __init__(self, shape, std=1):
        super().__init__()

        self.noise = Variable(torch.zeros(shape))
        self.std = std

    def forward(self):
        self.noise.data.normal_(0, std=self.std)

        return self.noise


class AdaIn(Module):
    def __init__(self):
        super(AdaIn, self).__init__()

    def forward(self, x, y):
        mu = torch.mean(x)
        std = torch.std(x)

        mu_style = torch.mean(y)
        std_style = torch.std(y)
        z_norm = (torch.mul(torch.add(x, -mu), 1/std))

        # Perform element wise-multiplication
        adaptive_norm = torch.add(torch.mul(z_norm, std_style), mu_style)

        return adaptive_norm
