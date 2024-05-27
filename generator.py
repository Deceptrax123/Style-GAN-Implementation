import torch
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
