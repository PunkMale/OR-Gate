import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .pooling import *
except:
    from pooling import *

''' Res2Conv1d + BatchNorm1d + ReLU
'''
class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out



''' Conv1d + BatchNorm1d + ReLU
'''
class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))



''' The SE connection of 1D case.
'''
class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channesl, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out



''' SE-Res2Block.
    Note: residual connection is implemented in the ECAPA_TDNN model, not here.
'''
def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )


''' Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
'''
class ECAPA_TDNN(nn.Module):
    def __init__(self, embedding_dim=256, pooling_type="ASP", n_mels=80, channels=512, **kwargs):
        super().__init__()
        self.layer1 = Conv1dReluBn(n_mels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)

        if pooling_type == "TemporalAveragePooling" or pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn1 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embedding_dim)
            self.bn2 = nn.BatchNorm1d(embedding_dim)

        elif pooling_type == "TemporalStatisticsPooling" or pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn1 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embedding_dim)
            self.bn2 = nn.BatchNorm1d(embedding_dim)

        elif pooling_type == "SelfAttentivePooling" or pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(cat_channels)
            self.bn1 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embedding_dim)
            self.bn2 = nn.BatchNorm1d(embedding_dim)

        elif pooling_type == "Attentive_Statistics_Pooling" or pooling_type == "ASP":
            self.pooling = Attentive_Statistics_Pooling(cat_channels)
            self.bn1 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embedding_dim)
            self.bn2 = nn.BatchNorm1d(embedding_dim)

        else:
            raise ValueError('{} pooling type is not defined'.format(pooling_type))


    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.pooling(out)
        out = self.bn1(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        out = self.bn2(out)
        return out


if __name__ == '__main__':
    x = torch.zeros(10, 80, 200)
    model = ECAPA_TDNN()
    print(model)
    out = model(x)
    total = sum([param.nelement() for param in model.parameters()])
    print("total param: {:.2f}M".format(total/1e6))
    print(out.shape)
