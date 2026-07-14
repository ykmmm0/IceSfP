import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.backbone import build_backbone
from tensorboardX import SummaryWriter
import numpy as np
from modeling.man import make_man_blocks
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt


def show_feature_map(x, title='', max_channels=32):
    x = x.detach().cpu()
    B, C, H, W = x.shape
    print(f'[INFO] {title}: shape = {x.shape}')
    show_c = min(C, max_channels)
    plt.figure(figsize=(15, 3 * show_c))
    for i in range(show_c):
        plt.subplot((show_c + 3) // 4, 4, i + 1)
        plt.imshow(x[0, i], cmap='gray')
        plt.title(f'{title} - ch {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class MSConv(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, padding=0)
        self.conv3 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv5 = nn.Conv2d(in_c, out_c, 5, padding=2)
        self.fuse = nn.Conv2d(out_c * 3, out_c, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        out = torch.cat([c1, c3, c5], dim=1)
        return self.fuse(out)


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(True), nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.avg(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class FusionModule(nn.Module):

    def __init__(self, in_channels, out_channels, output_size, atten=False):
        super(FusionModule, self).__init__()
        self.size = output_size
        self.atten = atten
        self.fuse = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False), nn.InstanceNorm2d(in_channels), nn.LeakyReLU(0.1, inplace=True))
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(in_channels), nn.LeakyReLU(0.1, inplace=True))
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x_raw, x_prior):
        x = torch.cat([x_raw, x_prior], dim=1)
        x = self.fuse(x)
        x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.out_conv(x)
        return x


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        H = int(H / 2)
        W = int(W / 2)
        q = F.interpolate(q, size=(H, W), mode='bilinear', align_corners=False)
        k = F.interpolate(k, size=(H, W), mode='bilinear', align_corners=False)
        v = F.interpolate(v, size=(H, W), mode='bilinear', align_corners=False)
        q = q.view(B, 1, C, H, W).view(B, 1, C, H * W).permute(0, 1, 3, 2)
        k = k.view(B, 1, C, H, W).view(B, 1, C, H * W)
        v = v.view(B, 1, C, H, W).view(B, 1, C, H * W).permute(0, 1, 3, 2)
        attn = torch.matmul(q, k) / np.sqrt(H * W)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 1, 3, 2)
        out = out.view(B, C, H, W)
        H *= 2
        W *= 2
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


class upSample(nn.Module):

    def __init__(self, in_channels, out_channels, output_size):
        super(upSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=3, padding=3 // 2)
        self.leaky_relu = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn2 = nn.InstanceNorm2d(num_features=out_channels)
        self.size = output_size
        self.se = SEBlock(out_channels)

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=1)
        x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.se(x)
        return x


class FuseBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super().__init__()
        self.snorm = SPADE(inplanes)
        self.conv = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x, polar_image):
        x = self.snorm(x, polar_image)
        x = self.conv(x)
        x = self.relu(x)
        return x


class SPADEDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.up4 = UpSam(256, 256)
        self.up3 = UpSam(512, 128)
        self.up2 = UpSam(256, 64)
        self.up1 = UpSam(128, 32)
        self.up0 = FuseBlock(80, 32)
        self.final = FinalLayer(32, 3)

    def forward(self, x, x0, x1, x2, x3, x4, polar_image):
        x = self.up4(x, x4, polar_image)
        x = self.up3(x, x3, polar_image)
        x = self.up2(x, x2, polar_image)
        x = self.up1(x, x1, polar_image)
        x = torch.cat((x, x0), dim=1)
        x = self.up0(x, polar_image)
        out = self.final(x)
        return out
import math


class ThreeBranchCMA(nn.Module):

    def __init__(self, raw_channels, physics_channels):
        super().__init__()
        self.raw_channels = raw_channels
        self.physics_channels = physics_channels
        self.raw_proj = nn.Conv2d(raw_channels, raw_channels, kernel_size=1)
        self.physics_proj = nn.Conv2d(physics_channels, raw_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, raw_feat, physics_feat):
        B, C, H, W = raw_feat.shape
        raw = self.raw_proj(raw_feat)
        physics = self.physics_proj(physics_feat)
        raw_flat = raw.view(B, C, -1).permute(0, 2, 1)
        physics_flat = physics.view(B, C, -1)
        attn = torch.bmm(raw_flat, physics_flat) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        physics_weighted = torch.bmm(attn, physics_flat.permute(0, 2, 1))
        physics_weighted = physics_weighted.permute(0, 2, 1).view(B, C, H, W)
        fused = raw + self.gamma * physics_weighted
        return fused


class FinalLayer(nn.Module):

    def __init__(self, inplanes, planes):
        super(FinalLayer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.In = nn.InstanceNorm2d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.In(out)
        out = self.relu(out)
        out += x
        out = self.conv2(out)
        return out


class UpSam(nn.Module):

    def __init__(self, inplanes, planes, padding=1):
        super(UpSam, self).__init__()
        self.BupSam = nn.UpsamplingBilinear2d(scale_factor=2)
        self.snorm1 = SPADE(inplanes)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=padding)
        self.snorm2 = SPADE(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding)

    def forward(self, x, x_pre, resize_image):
        out = self.BupSam(x)
        out = self.snorm1(out, resize_image)
        out = self.relu(out)
        out1 = self.conv1(out)
        out = self.snorm2(out1, resize_image)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + out1
        x_pre = F.interpolate(x_pre, size=(out.size(2), out.size(3)), mode='nearest')
        out = torch.cat([out, x_pre], dim=1)
        return out


class SPADE(nn.Module):

    def __init__(self, out_channels):
        super(SPADE, self).__init__()
        self.norm = nn.BatchNorm2d(out_channels, affine=False)
        self.conv = nn.Sequential(spectral_norm(nn.Conv2d(4, out_channels, 3, 1, 1)), nn.ReLU())
        self.conv_gamma = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.conv_beta = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, x, polar_image):
        polar_image = F.interpolate(polar_image, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        polar_image = self.conv(polar_image)
        return self.norm(x) * self.conv_gamma(polar_image) + self.conv_beta(polar_image) + self.norm(x)


class SPADEDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.up4 = UpSam(256, 256)
        self.up3 = UpSam(512, 128)
        self.up2 = UpSam(256, 64)
        self.up1 = UpSam(128, 32)
        self.final = FinalLayer(80, 3)

    def forward(self, x, x0, x1, x2, x3, x4, polar_image):
        x = self.up4(x, x4, polar_image)
        x = self.up3(x, x3, polar_image)
        x = self.up2(x, x2, polar_image)
        x = self.up1(x, x1, polar_image)
        x = torch.cat((x, x0), dim=1)
        out = self.final(x)
        return out


class Icesfp(nn.Module):

    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, sync_bn=True, freeze_bn=False, device=None):
        super(Icesfp, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.kernel_size = 9
        self.lamda = 1
        self.m = 0.5
        self.mean_kernel = torch.ones([1, 1, self.kernel_size, self.kernel_size]) / self.kernel_size ** 2
        self.mean_kernel = self.mean_kernel.to(device)
        self.mean_kernel = nn.Parameter(data=self.mean_kernel, requires_grad=False)
        self.sum_kernel_1 = torch.ones([1, 1, self.kernel_size, self.kernel_size])
        self.sum_kernel_1 = self.sum_kernel_1.to(device)
        self.sum_kernel_1 = nn.Parameter(data=self.sum_kernel_1, requires_grad=False)
        self.sum_kernel_3 = torch.ones([3, 3, self.kernel_size, self.kernel_size])
        self.sum_kernel_3 = self.sum_kernel_3.to(device)
        self.sum_kernel_3 = nn.Parameter(data=self.sum_kernel_3, requires_grad=False)
        self.backbone_orig = build_backbone(in_channels=4, backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm, Fusion=True)
        self.aspp_orig = build_aspp(backbone, output_stride, BatchNorm)
        self.backbone_prior = build_backbone(in_channels=10, backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm, Fusion=True)
        self.aspp_prior = build_aspp(backbone, output_stride, BatchNorm)
        self.backbone_atten = build_backbone(in_channels=1, backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm, Fusion=True)
        self.aspp_atten = build_aspp(backbone, output_stride, BatchNorm)
        self.cma = ThreeBranchCMA(raw_channels=256, physics_channels=256)
        self.SPADEDecoder = SPADEDecoder()
        self.fusion_module_0 = FusionModule(in_channels=64, out_channels=16, output_size=512)
        self.fusion_module_1 = FusionModule(in_channels=64, out_channels=32, output_size=256)
        self.fusion_module_2 = FusionModule(in_channels=256, out_channels=64, output_size=128)
        self.fusion_module_3 = FusionModule(in_channels=512, out_channels=128, output_size=64)
        self.fusion_module_4 = FusionModule(in_channels=1024, out_channels=256, output_size=32)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        if freeze_bn:
            self.freeze_bn()
        self.writer = SummaryWriter()

    def forward(self, orig, prior, confidence_map_init=None):
        img = orig
        img_split = torch.split(img, 1, 1)
        aolp = img_split[1]
        DoLP = orig[:, 0:1, :, :]
        AoLP = orig[:, 1:2, :, :]
        AoLP_rad = AoLP * math.pi
        AoLP_sin = torch.sin(2 * AoLP_rad)
        AoLP_cos = torch.cos(2 * AoLP_rad)
        I0 = orig[:, 3:4, :, :]
        I45 = orig[:, 4:5, :, :]
        I90 = orig[:, 5:6, :, :]
        I135 = orig[:, 6:7, :, :]
        orig = torch.cat([I0, I45, I90, I135], dim=1)
        orig_pol = torch.cat([I0, I45, I90, I135], dim=1)
        syn_N1 = prior[:, 0:3, :, :]
        syn_N2 = prior[:, 3:6, :, :]
        syn_N3 = prior[:, 6:9, :, :]
        syn_N4 = prior[:, 9:12, :, :]
        syn_N5 = prior[:, 12:15, :, :]
        syn_N6 = prior[:, 15:18, :, :]
        prior = torch.cat([syn_N3, syn_N4, syn_N5, confidence_map_init], dim=1)
        x_orig, x_orig_0, x_orig_1, x_orig_2, x_orig_3, x_orig_4 = self.backbone_orig(orig)
        x_orig = self.aspp_orig(x_orig)
        x_prior, x_prior_0, x_prior_1, x_prior_2, x_prior_3, x_prior_4 = self.backbone_prior(prior)
        x_prior = self.aspp_prior(x_prior)
        confidence_map = confidence_map_init
        x_fusion = self.cma(x_orig, x_prior)
        x_fusion_0 = self.fusion_module_0(x_orig_0, x_prior_0)
        x_fusion_1 = self.fusion_module_1(x_orig_1, x_prior_1)
        x_fusion_2 = self.fusion_module_2(x_orig_2, x_prior_2)
        x_fusion_3 = self.fusion_module_3(x_orig_3, x_prior_3)
        x_fusion_4 = self.fusion_module_4(x_orig_4, x_prior_4)
        x = self.SPADEDecoder(x_fusion, x_fusion_0, x_fusion_1, x_fusion_2, x_fusion_3, x_fusion_4, orig_pol)
        return (x, confidence_map)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
