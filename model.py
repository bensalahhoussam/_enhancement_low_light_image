import torch
import torch.nn as nn
import torch.nn.functional as F


def lrelu(x):
    return F.leaky_relu(x, negative_slope=0.2)


class UpSampleConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleConcat, self).__init__()
        # Deconvolution (same as conv2d_transpose)
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        # Concatenate along channel dimension
        x = torch.cat([x1, x2], dim=1)
        return x


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()

        def conv_block(in_c, out_c, name):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.conv1 = conv_block(3, 32, 'conv1')
        self.conv2 = conv_block(32, 64, 'conv2')
        self.conv3 = conv_block(64, 128, 'conv3')
        self.conv4 = conv_block(128, 256, 'conv4')
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool(conv4)

        conv5 = self.conv5(pool4)

        feature = torch.mean(conv5, dim=(2, 3))  # [B, 512]

        return conv1, conv2, conv3, conv4, feature


class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()

        self.fc = nn.Linear(512, 256)

        self.up6_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up7 = UpSampleConcat(256, 128)
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up8 = UpSampleConcat(128, 64)
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up9 = UpSampleConcat(64, 32)
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, conv1, conv2, conv3, conv4, feature):
        # global feature
        feature = F.relu(self.fc(feature))
        feature = feature.unsqueeze(-1).unsqueeze(-1)
        feature = feature.expand(-1, -1, conv4.size(2), conv4.size(3))

        up6 = torch.cat([conv4, feature], dim=1)
        conv6 = self.up6_conv(up6)

        up7 = self.up7(conv6, conv3)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7, conv2)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8, conv1)
        conv9 = self.conv9(up9)

        out = torch.sigmoid(self.out_conv(conv9))
        return out


class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.encoder = EncoderNet()
        self.decoder = DecoderNet()

    def forward(self, x):
        conv1, conv2, conv3, conv4, feat = self.encoder(x)
        out = self.decoder(conv1, conv2, conv3, conv4, feat)
        return out


