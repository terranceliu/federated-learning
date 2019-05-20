#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, args.num_classes)

        # self.weight_keys = [['fc3.weight', 'fc3.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]

        # self.weight_keys = [['conv1.weight', 'conv1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ]

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class AllConvNet(nn.Module):
    def __init__(self, args, dropout=True):
        super(AllConvNet, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(args.num_channels, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.class_conv = nn.Conv2d(192, args.num_classes, 1)
        self.weight_keys = [['conv1.weight', 'conv1.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv3.weight', 'conv3.bias'],
                            ['conv4.weight', 'conv4.bias'],
                            ['conv5.weight', 'conv5.bias'],
                            ['conv6.weight', 'conv6.bias'],
                            ['conv7.weight', 'conv7.bias'],
                            ['conv8.weight', 'conv8.bias'],
                            ['class_conv.weight', 'class_conv.bias']
                            ]

    def forward(self, x):
        if self.dropout:
            x = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        if self.dropout:
            conv3_out = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        if self.dropout:
            conv6_out = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = class_out.reshape(class_out.size(0), class_out.size(1), -1).mean(-1)
        return pool_out


class ResnetCifar(nn.Module):
    def __init__(self, args):
        super(ResnetCifar, self).__init__()
        self.extractor = models.resnet18(pretrained=False)
        self.fflayer = nn.Sequential(nn.Linear(1000, args.num_classes))

    def forward(self, x):
        x = self.extractor(x)
        x = self.fflayer(x)
        return F.log_softmax(x, dim=1)

class ResnetCifar(nn.Module):
    def __init__(self, args):
        super(ResnetCifar, self).__init__()
        self.extractor = models.resnet18(pretrained=False)
        self.fflayer = nn.Sequential(nn.Linear(1000, args.num_classes))

    def forward(self, x):
        x = self.extractor(x)
        x = self.fflayer(x)
        return F.log_softmax(x, dim=1)