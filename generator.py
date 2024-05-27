import torch
from torch.autograd import Variable
from torch.nn import Linear, Conv2d, ConvTranspose2d, ReLU, Module, BatchNorm1d


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
        self.linear7 = Linear(out_features=128, in_features=256)
        self.linear8 = Linear(in_features=256, out_features=512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        x = self.linear7(x)
        x = self.linear8(x)

        return x


class SynthesisNetwork(Module):
    def __init__(self, in_channels, out_channels, noise_shape1, noise_shape2):
        super(SynthesisNetwork, self).__init__()

        self.ada1 = AdaIn()
        self.ada2 = AdaIn()
        self.ada3 = AdaIn()
        self.ada4 = AdaIn()

        self.upsample = ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                        stride=2, padding=1, output_padding=1, kernel_size=(3, 3))

        self.conv1 = Conv2d(in_channels=in_channels, out_channels=in_channels,
                            stride=1, kernel_size=(3, 3), padding=1)
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels,
                            stride=1, kernel_size=(3, 3), padding=1)
        self.conv3 = Conv2d(in_channels=out_channels, out_channels=out_channels,
                            stride=1, kernel_size=(3, 3), padding=1)

        self.gaussian1 = GaussianNoise(shape=noise_shape1)
        self.gaussian2 = GaussianNoise(shape=noise_shape1)
        self.gaussian3 = GaussianNoise(shape=noise_shape2)
        self.gaussian4 = GaussianNoise(shape=noise_shape2)

    def forward(self, f, const):
        B1 = self.gaussian1()
        B2 = self.gaussian2()
        B3 = self.gaussian3()
        B4 = self.gaussian4()

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
    def __init__(self, shape, std=0.05):
        super().__init__()

        self.noise = Variable(torch.zeros(shape, shape))
        self.std = std

    def forward(self):
        self.noise.data.normal_(0, std=self.std)

        return self.noise


class AdaIn(Module):
    def __init__(self):
        super(AdaIn, self).__init__()

    def forward(self, x, y):
        mu = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)

        mu_style = torch.mean(y, dim=1)
        std_style = torch.std(y, dim=1)
        z_norm = (x-mu)/std  # Find z score

        # Perform element wise-multiplication
        adaptive_norm = (std_style*z_norm)+mu_style

        return adaptive_norm
