from turtle import forward
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import Type, Union, List, Optional, Callable
from utils import _log_api_usage_once
import numpy as np 
import math


class adam(nn.Module):
    def __init__(self, inplanes: int, dilation=(1,3)):
        super(adam, self).__init__()
        self.inplanes = inplanes
        self.F1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, dilation=dilation[0], padding=dilation[0]),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU()
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, dilation=dilation[1], padding=dilation[1]),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU()
        )
        self.Conv_q = nn.Conv2d(self.inplanes, 1, kernel_size=1)
        
        r = 8
        L = 48
        d = max(int(self.inplanes/r), L)
        self.W1 = nn.Linear(2 * self.inplanes, d)
        self.W2 = nn.Linear(d, self.inplanes)
        self.W3 = nn.Linear(d, self.inplanes)
    
    def forward(self, x):
        U1 = self.F1(x)
        U2 = self.F2(x)
        U = U1 + U2

        p = torch.mean(torch.mean(U, dim=2), dim=2).reshape(-1, self.inplanes, 1, 1)
        q = torch.matmul(
            U.reshape([-1, self.inplanes, x.shape[2] * x.shape[3]]),
            (nn.Softmax(dim = 1)(self.Conv_q(x))).reshape([-1, x.shape[2] * x.shape[3], 1])
            ).reshape([-1, self.inplanes, 1, 1])
        s = torch.sigmoid(torch.cat([p, q], 1)).reshape([-1, self.inplanes * 2])

        z1 = self.W2(nn.ReLU()(self.W1(s)))
        z2 = self.W3(nn.ReLU()(self.W1(s)))
        a1 = (torch.exp(z1) / (torch.exp(z1) + torch.exp(z2))).reshape([-1, self.inplanes, 1, 1])
        a2 = (torch.exp(z2) / (torch.exp(z1) + torch.exp(z2))).reshape([-1, self.inplanes, 1, 1])
        
        V = U1 * a1 + U2 * a2 + x  
        return V


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        inplanes: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(inplanes, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ADAMNet_Seg(nn.Module):
    # Using BasicBlock
    def __init__(
        self,
        inplanes: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.sres = ResNet(inplanes, block, layers, 3, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.bres = ResNet(inplanes, block, layers, 3, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        
        self.ad1 = adam(576, (3, 12))
        self.ad2 = adam(704, (3, 8))
        self.ad3 = adam(896, (3, 6))
        self.ad4 = adam(1280, (3, 5))
        self.ad5 = adam(512, (1, 3))
        
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # TODO: Choose upsample mode
        self.Up_to_c1 = nn.Upsample(scale_factor=16, mode='nearest')
        self.Up_to_c2 = nn.Upsample(scale_factor=8, mode='nearest')
        self.Up_to_c3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.Up_to_c4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(576, 1, 1)
        self.conv2 = nn.Conv2d(704, 1, 1)
        self.conv3 = nn.Conv2d(896, 1, 1)
        self.conv4 = nn.Conv2d(1280, 1, 1)
        self.conv5 = nn.Conv2d(512, 1, 1)
        
        self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')

        
    def forward(self, bx: Tensor) -> Tensor:
        '''
        bx: big image
        sx: small image
        '''
        sx = nn.functional.interpolate(bx, (256, 192))
        sx = self.sres.conv1(sx)
        sx = self.sres.bn1(sx)
        sx = self.sres.relu(sx)
        sx = self.sres.maxpool(sx)
        sx_c2 = self.sres.layer1(sx)
        sx_c3 = self.sres.layer2(sx_c2)
        sx_c4 = self.sres.layer3(sx_c3)
        sx_c5 = self.sres.layer4(sx_c4)
        
        bx = self.bres.conv1(bx)
        bx = self.bres.bn1(bx)
        bx = self.bres.relu(bx)
        bx = self.bres.maxpool(bx)
        bx_c1 = self.bres.layer1(bx)
        bx_c2 = self.bres.layer2(bx_c1)
        bx_c3 = self.bres.layer3(bx_c2)
        bx_c4 = self.bres.layer4(bx_c3)
        
        adam5 = self.ad5(sx_c5)
        
        c1 = torch.cat([bx_c1, self.Up_to_c1(adam5)], 1)
        c2 = torch.cat([bx_c2, sx_c2, self.Up_to_c2(adam5)], 1)
        c3 = torch.cat([bx_c3, sx_c3, self.Up_to_c3(adam5)], 1)
        c4 = torch.cat([bx_c4, sx_c4, self.Up_to_c4(adam5)], 1)

        adam1 = self.ad1(c1)
        adam2 = self.ad2(c2)
        adam3 = self.ad3(c3)
        adam4 = self.ad4(c4)

        adam1 = self.conv1(adam1)
        adam2 = self.conv2(adam2)
        adam3 = self.conv3(adam3)
        adam4 = self.conv4(adam4)
        adam5 = self.conv5(adam5)

        Up1 = self.Upsample(adam5)
        Up2 = self.Upsample(adam4 + Up1)
        Up3 = self.Upsample(adam3 + Up2)
        Up4 = self.Upsample(adam2 + Up3)
        Up5 = self.Upsample(adam1 + Up4)
        Up6 = self.Upsample(Up5)

        out = self.sigmoid(Up6)
        return out


class ADAMNet50_Seg(nn.Module):
    # Using Bottleneck
    def __init__(
        self,
        inplanes: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.sres = ResNet(inplanes, block, layers, 3, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.bres = ResNet(inplanes, block, layers, 3, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        
        self.ad1 = adam(576*4, (3, 12))
        self.ad2 = adam(704*4, (3, 8))
        self.ad3 = adam(896*4, (3, 6))
        self.ad4 = adam(1280*4, (3, 5))
        self.ad5 = adam(512*4, (1, 3))
        
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Up_to_c1 = nn.Upsample(scale_factor=16, mode='nearest')
        self.Up_to_c2 = nn.Upsample(scale_factor=8, mode='nearest')
        self.Up_to_c3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.Up_to_c4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(576*4, 1, 1)
        self.conv2 = nn.Conv2d(704*4, 1, 1)
        self.conv3 = nn.Conv2d(896*4, 1, 1)
        self.conv4 = nn.Conv2d(1280*4, 1, 1)
        self.conv5 = nn.Conv2d(512*4, 1, 1)
        
        self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')

        
    def forward(self, bx: Tensor) -> Tensor:
        '''
        bx: big image
        sx: small image
        '''
        sx = nn.functional.interpolate(bx, (256, 192))
        sx = self.sres.conv1(sx)
        sx = self.sres.bn1(sx)
        sx = self.sres.relu(sx)
        sx = self.sres.maxpool(sx)
        sx_c2 = self.sres.layer1(sx)
        sx_c3 = self.sres.layer2(sx_c2)
        sx_c4 = self.sres.layer3(sx_c3)
        sx_c5 = self.sres.layer4(sx_c4)
        
        bx = self.bres.conv1(bx)
        bx = self.bres.bn1(bx)
        bx = self.bres.relu(bx)
        bx = self.bres.maxpool(bx)
        bx_c1 = self.bres.layer1(bx)
        bx_c2 = self.bres.layer2(bx_c1)
        bx_c3 = self.bres.layer3(bx_c2)
        bx_c4 = self.bres.layer4(bx_c3)
        
        adam5 = self.ad5(sx_c5)
        
        c1 = torch.cat([bx_c1, self.Up_to_c1(adam5)], 1)
        c2 = torch.cat([bx_c2, sx_c2, self.Up_to_c2(adam5)], 1)
        c3 = torch.cat([bx_c3, sx_c3, self.Up_to_c3(adam5)], 1)
        c4 = torch.cat([bx_c4, sx_c4, self.Up_to_c4(adam5)], 1)

        adam1 = self.ad1(c1)
        adam2 = self.ad2(c2)
        adam3 = self.ad3(c3)
        adam4 = self.ad4(c4)

        adam1 = self.conv1(adam1)
        adam2 = self.conv2(adam2)
        adam3 = self.conv3(adam3)
        adam4 = self.conv4(adam4)
        adam5 = self.conv5(adam5)

        Up1 = self.Upsample(adam5)
        Up2 = self.Upsample(adam4 + Up1)
        Up3 = self.Upsample(adam3 + Up2)
        Up4 = self.Upsample(adam2 + Up3)
        Up5 = self.Upsample(adam1 + Up4)
        Up6 = self.Upsample(Up5)

        out = self.sigmoid(Up6)
        return out
    

class ADAMNet_Seg_Trans(nn.Module):
    # Using Transposed Convolution for Upsampling
    def __init__(
        self,
        inplanes: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.sres = ResNet(inplanes, block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.bres = ResNet(inplanes, block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        
        self.adam1 = adam(576)
        self.adam2 = adam(704)
        self.adam3 = adam(896)
        self.adam4 = adam(1280)
        self.adam5 = adam(512)
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Up_to_c1 = nn.Upsample(scale_factor=16, mode='nearest')
        self.Up_to_c2 = nn.Upsample(scale_factor=8, mode='nearest')
        self.Up_to_c3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.Up_to_c4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(576, 1, 1)
        self.conv2 = nn.Conv2d(704, 1, 1)
        self.conv3 = nn.Conv2d(896, 1, 1)
        self.conv4 = nn.Conv2d(1280, 1, 1)
        self.conv5 = nn.Conv2d(512, 1, 1)

        self.Trans = nn.ConvTranspose2d(1, 1, 2, 2)
        
    def forward(self, bx: Tensor) -> Tensor:
        sx = nn.functional.interpolate(bx, (192, 256))
        sx = self.sres.conv1(sx)
        sx = self.sres.bn1(sx)
        sx = self.sres.relu(sx)
        sx = self.sres.maxpool(sx)
        sx_c2 = self.sres.layer1(sx)
        sx_c3 = self.sres.layer2(sx_c2)
        sx_c4 = self.sres.layer3(sx_c3)
        sx_c5 = self.sres.layer4(sx_c4)
        
        bx = self.bres.conv1(bx)
        bx = self.bres.bn1(bx)
        bx = self.bres.relu(bx)
        bx = self.bres.maxpool(bx)
        bx_c1 = self.bres.layer1(bx)
        bx_c2 = self.bres.layer2(bx_c1)
        bx_c3 = self.bres.layer3(bx_c2)
        bx_c4 = self.bres.layer4(bx_c3)
        
        adam5 = self.adam5(sx_c5)
        
        c1 = torch.cat([bx_c1, self.Up_to_c1(adam5)], 1)
        c2 = torch.cat([bx_c2, sx_c2, self.Up_to_c2(adam5)], 1)
        c3 = torch.cat([bx_c3, sx_c3, self.Up_to_c3(adam5)], 1)
        c4 = torch.cat([bx_c4, sx_c4, self.Up_to_c4(adam5)], 1)

        adam1 = self.adam1(c1)
        adam2 = self.adam2(c2)
        adam3 = self.adam3(c3)
        adam4 = self.adam4(c4)

        adam1 = self.conv1(adam1)
        adam2 = self.conv2(adam2)
        adam3 = self.conv3(adam3)
        adam4 = self.conv4(adam4)
        adam5 = self.conv5(adam5)

        Up1 = self.Trans(adam5)
        Up2 = self.Trans(adam4 + Up1)
        Up3 = self.Trans(adam3 + Up2)
        Up4 = self.Trans(adam2 + Up3)
        Up5 = self.Trans(adam1 + Up4)
        Up6 = self.Trans(Up5)

        out = self.sigmoid(Up6)
        return out


class ADAMNet_Seg_Interpolate(nn.Module):
    # Using Interpolation for Upsampling
    def __init__(
        self,
        inplanes: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.sres = ResNet(inplanes, block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.bres = ResNet(inplanes, block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        
        self.adam1 = adam(576)
        self.adam2 = adam(704)
        self.adam3 = adam(896)
        self.adam4 = adam(1280)
        
        self.adam5 = adam(512)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.Up_to_c1 = nn.Upsample(scale_factor=16, mode='nearest')
        self.Up_to_c2 = nn.Upsample(scale_factor=8, mode='nearest')
        self.Up_to_c3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.Up_to_c4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(576, 1, 1)
        self.conv2 = nn.Conv2d(704, 1, 1)
        self.conv3 = nn.Conv2d(896, 1, 1)
        self.conv4 = nn.Conv2d(1280, 1, 1)
        self.conv5 = nn.Conv2d(512, 1, 1)
        
    def forward(self, bx: Tensor) -> Tensor:
        sx = nn.functional.interpolate(bx, (192, 256))
        sx = self.sres.conv1(sx)
        sx = self.sres.bn1(sx)
        sx = self.sres.relu(sx)
        sx = self.sres.maxpool(sx)
        sx_c2 = self.sres.layer1(sx)
        sx_c3 = self.sres.layer2(sx_c2)
        sx_c4 = self.sres.layer3(sx_c3)
        sx_c5 = self.sres.layer4(sx_c4)
        
        bx = self.bres.conv1(bx)
        bx = self.bres.bn1(bx)
        bx = self.bres.relu(bx)
        bx = self.bres.maxpool(bx)
        bx_c1 = self.bres.layer1(bx)
        bx_c2 = self.bres.layer2(bx_c1)
        bx_c3 = self.bres.layer3(bx_c2)
        bx_c4 = self.bres.layer4(bx_c3)
        
        adam5 = self.adam5(sx_c5)
        
        c1 = torch.cat([bx_c1, self.Up_to_c1(adam5)], 1)
        c2 = torch.cat([bx_c2, sx_c2, self.Up_to_c2(adam5)], 1)
        c3 = torch.cat([bx_c3, sx_c3, self.Up_to_c3(adam5)], 1)
        c4 = torch.cat([bx_c4, sx_c4, self.Up_to_c4(adam5)], 1)

        adam1 = self.adam1(c1)
        adam2 = self.adam2(c2)
        adam3 = self.adam3(c3)
        adam4 = self.adam4(c4)

        adam1 = self.conv1(adam1)
        adam2 = self.conv2(adam2)
        adam3 = self.conv3(adam3)
        adam4 = self.conv4(adam4)
        adam5 = self.conv5(adam5)
        Up1 = F.interpolate(adam5, scale_factor=2, mode='bilinear')
        Up2 = F.interpolate(adam4 + Up1, scale_factor=2, mode='bilinear')
        Up3 = F.interpolate(adam3 + Up2, scale_factor=2, mode='bilinear')
        Up4 = F.interpolate(adam2 + Up3, scale_factor=2, mode='bilinear')
        Up5 = F.interpolate(adam1 + Up4, scale_factor=2, mode='bilinear')
        Up6 = F.interpolate(Up5, scale_factor=2, mode='bilinear')

        out = self.sigmoid(Up6)
        return out
