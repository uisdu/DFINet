import torch
import torch.nn as nn
import torch.nn.functional as F
import math
""" Squeeze and Excitation block """
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

""" 3x3->3x3 Residual block """
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.se = SELayer(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4


class MDADRB(nn.Module):
    def __init__(self, in_c, out_c):
        super(MDADRB, self).__init__()

        self.conv1 =Conv2d_cd(in_c,out_c, kernel_size=3, padding=1)
       
        self.bn1 = nn.BatchNorm2d(out_c)
        
                  
        self.conv2 = Conv2d_cd(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 =nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.se =  CoordAttention(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
    
        
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x2)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4

class DFFFM(nn.Module):
    def __init__(self, in_c, out_c):
        super(DFFFM, self).__init__()

        # 生成硬注意力掩码
        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # 融合卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, m):
        device = x.device

        # 1️⃣ fmask 硬化（硬注意力）
        fmask_soft = self.fmask(x)
        fmask_hard = (fmask_soft > 0.5).float().to(device)  # 二值化
        fmask_hard = fmask_hard.detach()

        # 2️⃣ 将输入 mask 调整到特征图尺寸
        # m 原本是 soft mask，0~1
        m_resized = torch.nn.functional.interpolate(m, size=(x.shape[2], x.shape[3]), mode='nearest')
        m_hard = (m_resized > 0.5).float()  # 二值化

        # 3️⃣ 逻辑 OR 融合
        mask_combined = torch.logical_or(fmask_hard.bool(), m_hard.bool()).float()

        # 4️⃣ 应用 mask 并融合特征
        x1 = x * mask_combined
        x1 = torch.cat((x1, x), dim=1)
        x_out = self.conv3(x1)

        return x_out

##CAattention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h





#即对应文中的MPM模块
class HPPM(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(HPPM, self).__init__()
        #空间池化
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        #strip pooling
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        #PPM分支的输出结果
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        #strip pooling的输出结果
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        #拼接+1x1卷积
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


#CDC卷积
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class DSFM(nn.Module):
    def __init__(self, dim, height=2, reduction=8, gamma=0.5):
        super().__init__()
        self.height = height
        self.gamma = gamma

        d = max(int(dim / reduction), 4)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

        self.alpha = nn.Parameter(torch.zeros(2))

    def forward(self, in_feats, mask):
        x_low, x_high = in_feats
        B, C, H, W = x_low.shape

        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

       
        g_pd = mask.detach()                     

        uncertainty = 1 - torch.abs(mask - 0.5) * 2
        g_iou = uncertainty.detach()           

        x_low_pd  = x_low  + x_low  * g_pd
        x_high_pd = x_high + x_high * g_pd

        x_low_iou  = x_low  * (1 + self.gamma * g_iou * (g_pd - 0.5))
        x_high_iou = x_high * (1 + self.gamma * g_iou * (g_pd - 0.5))

        feats_pd  = torch.stack([x_low_pd,  x_high_pd],  dim=1)
        feats_iou = torch.stack([x_low_iou, x_high_iou], dim=1)

        def fuse(feats):
            feats_sum = feats.sum(dim=1)
            attn = self.mlp(feats_sum)
            attn = self.softmax(attn.view(B, self.height, C, 1, 1))
            return torch.sum(feats * attn, dim=1)

        fused_pd  = fuse(feats_pd)
        fused_iou = fuse(feats_iou)

        # ===== Adaptive balance =====
        w = torch.softmax(self.alpha, dim=0)
        out = w[0] * fused_pd + w[1] * fused_iou

        return out
