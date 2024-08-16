import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class DiceCoef(Function):
    @staticmethod
    def forward(ctx, y_true, y_pred):
        smooth = 1e-7
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            torch.sum(y_true_f * y_true_f) + torch.sum(y_pred_f * y_pred_f) + smooth
        )

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_true, y_pred):
        return 1.0 - DiceCoef.apply(y_true, y_pred)

class JaccardCoef(Function):
    @staticmethod
    def forward(ctx, y_true, y_pred):
        smooth = 1e-7
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (intersection + smooth) / (
            torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection + smooth
        )

class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, y_true, y_pred):
        return 1.0 - JaccardCoef.apply(y_true, y_pred)

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, y_true, y_pred):
        # Simplified SSIM, consider using a dedicated SSIM library for better accuracy
        return 1.0 - torch.mean(F.mse_loss(y_true, y_pred))

class DISLoss(nn.Module):
    def __init__(self):
        super(DISLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.jaccard_loss = JaccardLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, y_true, y_pred):
        return self.dice_loss(y_true, y_pred) + self.jaccard_loss(y_true, y_pred) + self.ssim_loss(y_true, y_pred)

class PyramidPoolingBlock(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(PyramidPoolingBlock, self).__init__()
        self.stages = []
        for bin_size in bin_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                nn.Upsample(scale_factor=bin_size, mode='bilinear', align_corners=False)
            ))
        self.stages = nn.ModuleList(self.stages)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out = [x]
        for stage in self.stages:
            out.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False))
        return torch.cat(out, dim=1)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], dropout_rate=0.1):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)
        self.pyramid_pool = PyramidPoolingBlock(features[-1] * 2, bin_sizes=[1, 2, 4, 8])

        self.upconv = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for feature in reversed(features):
            self.upconv.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.pyramid_pool(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.upconv)):
            x = self.upconv[idx](x)
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](concat_skip)

        return torch.sigmoid(self.final_conv(x))

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
