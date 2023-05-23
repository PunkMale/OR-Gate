#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.nnet.pooling import *


class TDNN(nn.Module):
    def __init__(self, embedding_dim, pooling_type, n_mels=40, **kwargs):
        super(TDNN, self).__init__()
        self.td_layer1 = torch.nn.Conv1d(in_channels=n_mels, out_channels=512, dilation=1, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.td_layer2 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.td_layer3 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=3, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.td_layer4 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=1, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.td_layer5 = torch.nn.Conv1d(in_channels=512, out_channels=1500, dilation=1, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm1d(1500)

        if pooling_type == "Temporal_Average_Pooling" or pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.fc1 = nn.Linear(1500, 512)

        elif pooling_type == "Temporal_Statistics_Pooling" or pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.fc1 = nn.Linear(1500 * 2, 512)

        elif pooling_type == "Self_Attentive_Pooling" or pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(1500)
            self.fc1 = nn.Linear(1500, 512)

        elif pooling_type == "Attentive_Statistics_Pooling" or pooling_type == "ASP":
            self.pooling = AttentiveStatisticsPooling(1500)
            self.fc1 = nn.Linear(1500 * 2, 512)

        else:
            raise ValueError('{} pooling type is not defined'.format(pooling_type))

        self.fc2 = nn.Linear(512, embedding_dim)

    def forward(self, x):
        '''
        x [batch_size, dim, time]
        '''
        x = F.relu(self.td_layer1(x))
        x = self.bn1(x)

        x = F.relu(self.td_layer2(x))
        x = self.bn2(x)

        x = F.relu(self.td_layer3(x))
        x = self.bn3(x)

        x = F.relu(self.td_layer4(x))
        x = self.bn4(x)

        x = F.relu(self.td_layer5(x))
        x = self.bn5(x)

        x = self.pooling(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def TDNN_Encoder(embedding_dim=512, pooling_type="TSP", **kwargs):
    model = TDNN(embedding_dim, pooling_type, **kwargs)
    return model


if __name__ == '__main__':
    model = TDNN_Encoder()
    total = sum([param.nelement() for param in model.parameters()])
    print(total / 1e6)
    data = torch.randn(10, 40, 100)
    out = model(data)
    print(data.shape)
    print(out.shape)
