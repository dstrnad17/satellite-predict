#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:34:07 2024

@author: dunnchadnstrnad
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 20)
        self.layer3 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = SimpleNN()
writer = SummaryWriter()
x = torch.randn(1, 10)
writer.add_graph(model, x)
writer.close()