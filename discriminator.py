from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, Dropout2d, Module
from torchsummary import summary


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2d(in_channels=3, out_channels=8,
                            stride=2, kernel_size=(3, 3), padding=1)
        self.bn1 = BatchNorm2d(8)
        self.dp1 = Dropout2d(0.2)
        self.lr1 = LeakyReLU(0.2)

        self.conv2 = Conv2d(in_channels=8, out_channels=16,
                            stride=2, kernel_size=(3, 3), padding=1)
        self.bn2 = BatchNorm2d(16)
        self.dp2 = Dropout2d(0.2)
        self.lr2 = LeakyReLU(0.2)

        self.conv3 = Conv2d(in_channels=16, out_channels=32,
                            stride=2, kernel_size=(3, 3), padding=1)
        self.bn3 = BatchNorm2d(32)
        self.dp3 = Dropout2d(0.2)
        self.lr3 = LeakyReLU(0.2)

        self.conv4 = Conv2d(in_channels=32, out_channels=64,
                            stride=2, kernel_size=(3, 3), padding=1)
        self.bn4 = BatchNorm2d(64)
        self.dp4 = Dropout2d(0.2)
        self.lr4 = LeakyReLU(0.2)

        self.conv5 = Conv2d(in_channels=64, out_channels=128,
                            stride=2, kernel_size=(3, 3), padding=1)
        self.bn5 = BatchNorm2d(128)
        self.dp5 = Dropout2d(0.2)
        self.lr5 = LeakyReLU(0.2)

        self.conv6 = Conv2d(in_channels=128, out_channels=256,
                            stride=2, kernel_size=(3, 3), padding=1)
        self.bn6 = BatchNorm2d(256)
        self.dp6 = Dropout2d(0.2)
        self.lr6 = LeakyReLU(0.2)

        self.conv7 = Conv2d(in_channels=256, out_channels=512,
                            kernel_size=(3, 3), padding=1, stride=2)
        self.bn7 = BatchNorm2d(512)
        self.dp7 = Dropout2d(0.2)
        self.lr7 = LeakyReLU(0.2)

        self.conv8 = Conv2d(in_channels=512, out_channels=1,
                            kernel_size=(4, 4), padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dp1(x)
        x = self.lr1(x)

        x = self.conv2(x)
        x = self.dp2(x)
        x = self.bn2(x)
        x = self.lr2(x)

        x = self.conv3(x)
        x = self.dp3(x)
        x = self.bn3(x)
        x = self.lr3(x)

        x = self.conv4(x)
        x = self.dp4(x)
        x = self.bn4(x)
        x = self.lr4(x)

        x = self.conv5(x)
        x = self.dp5(x)
        x = self.bn5(x)
        x = self.lr5(x)

        x = self.conv6(x)
        x = self.dp6(x)
        x = self.bn6(x)
        x = self.lr6(x)

        x = self.conv7(x)
        x = self.dp7(x)
        x = self.bn7(x)
        x = self.lr7(x)

        x = self.conv8(x)

        return x


# if __name__ == '__main__':
#     model = Discriminator()
#     summary(model, input_size=(3, 512, 512), batch_size=-1, device='cpu')
