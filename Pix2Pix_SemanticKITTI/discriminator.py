import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

#input image(x) and the output image(y) is concatenated along channel and given as inpu, therefore in_channels*2
class Discriminator(nn.Module):
    def __init__(self, in_channels=8, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels + 1,  # Here, we concatenate 'x' with 8 channels and 'y' with 1 channel
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):  # y can be the real or generated image
        x = torch.cat([x, y], dim=1)  # concatenating along the channel dimension
        x = self.initial(x)
        x = self.model(x)
        return x


def test():
    x = torch.randn((1, 8, 256, 256))
    y = torch.randn((1, 1, 256, 256))
    model = Discriminator(in_channels=8)
    preds = model(x, y)
    #print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()