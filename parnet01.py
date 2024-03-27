import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
import time as tm
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
from Utils.utils import (num_param, Concat2d, MultiBatchNorm2d, model_info,
                    trace_net, round)
from Utils.ClassifierHead import ClassifierHead


class SimpNet(nn.Module):
    def __init__(self, planes,
                 num_blocks,
                 dropout_lin,
                 ):
        super().__init__()
        print(f"Initializing ParNet with planes: {planes}")

        self.num_classes = 1000
        block = SimpleConvBlock

        last_planes = planes[-1]
        planes = planes[0:-1]
        strides = [2] * len(planes)
        assert (num_blocks[-1] == 1)
        assert (num_blocks[-2] != 1)
        num_blocks = num_blocks[0:-1]

        # inits
        self.inits = nn.ModuleList()
        in_planes = min(64, planes[0])
        inits = nn.Sequential(
            block(inplanes=3, planes=in_planes, kernel_size=3, stride=2),
            block(inplanes=in_planes, planes=planes[0], kernel_size=3, stride=2)
        )
        self.inits.append(inits)
        for i, (stride, in_plane, out_plane) in enumerate(
                zip(strides[1:], planes[0:-1], planes[1:])):
            self.inits.append(
                block(
                    inplanes=in_plane * block.expansion,
                    planes=out_plane * block.expansion,
                    stride=stride,
                    ))

        # streams
        self.streams = nn.ModuleList()

        def stream_block(stream_id, i, plane):
            _args = {
                'stride': 1,
                'use_se': True  # Changed from 'se_block' to 'use_se' to match SimplifiedBlock's parameter
            }
            out_block = block(plane, plane, kernel_size=3, **_args)
            return out_block

        for stream_id, (num_block, plane) in enumerate(
                zip(num_blocks, planes)):
            stream = nn.ModuleList()

            for i in range(num_block - 1):
                stream.append(
                    stream_block(stream_id, i, plane * block.expansion))
            self.streams.append(nn.Sequential(*stream))

        # downsamples_2
        self.downsamples_2 = nn.ModuleList()
        in_planes = planes[0:-1]
        out_planes = planes[1:]
        for i, (stride, in_plane, out_plane) in enumerate(zip(
                strides[1:], in_planes, out_planes)):
            if i == 0:
                self.downsamples_2.append(
                    block(inplanes=in_plane * block.expansion,
                          planes=out_plane * block.expansion,
                          stride=stride,
                          ))
            else:
                layer = nn.Sequential(
                    MultiBatchNorm2d(
                        in_plane * block.expansion,
                        in_plane * block.expansion),
                    Concat2d(shuffle=True),
                    block(
                        inplanes= 2 * in_plane * block.expansion,
                        planes= out_plane * block.expansion,
                        stride=2,
                        groups= 2, kernel_size=3))
                self.downsamples_2.append(layer)

        # combine
        in_planes_combine = planes[-1]
        combine = [
            MultiBatchNorm2d(
                in_planes_combine * block.expansion,
                in_planes_combine * block.expansion),
            Concat2d(shuffle=True),
            block(
                inplanes=2 * in_planes_combine * block.expansion,
                planes=in_planes_combine * block.expansion,
                stride=1,
                groups=2),
            block(
                inplanes=planes[-1], planes=last_planes,
                stride=2)
        ]
        self.combine = nn.Sequential(*combine)

        # head
        self.head = ClassifierHead(
            last_planes * block.expansion,
            self.num_classes,
            pool_type='avg',
            drop_rate=dropout_lin)
        self.num_features = last_planes * block.expansion

    def forward(self, img):
        x = img
        x_list = []
        for i, init in enumerate(self.inits):
            x = init(x)
            x_list.append(x)

        y_old = None
        for i, (x, stream) in enumerate(zip(x_list, self.streams)):
            y = stream(x)

            if y_old is None:
                y_old = self.downsamples_2[i](y)
            elif i < len(self.downsamples_2):
                y_old = self.downsamples_2[i]((y, y_old))
            else:
                y_old = (y, y_old)

        out = self.combine(y_old)
        out = self.head(out)

        return out


class SimpleConvBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, groups=1, activation=nn.ReLU, use_se=False, se_block=False):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride, padding=kernel_size//2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.activation = activation()
        
        self.use_se = use_se or se_block  # Incorporate `se_block` into the condition for using an SE block
        if self.use_se:
            self.se = SEBlock(planes)  # Assumes SEBlock is defined elsewhere

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_se:
            x = self.se(x)
        return self.activation(x)
    
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
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

if __name__ == '__main__':
    net = SimpNet(
        planes=[round(1 * x) for x in (64, 128, 256, 512)],
        num_blocks=[5, 6, 6, 1],
        dropout_lin=0.0
    )
    y = torch.randn(1, 3, 224, 224)
    print(f"Num Parameters: {num_param(net)}")
    out = net(y)
    trace_net(net, y)
