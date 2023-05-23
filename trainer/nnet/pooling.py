import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling


class TemporalAveragePooling(nn.Module):
    def __init__(self, **kwargs):
        """TAP
        Paper: Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super(TemporalAveragePooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        x = torch.mean(x, axis=2)
        return x


class TemporalStatisticsPooling(nn.Module):
    def __init__(self, **kwargs):
        """TSP
        Paper: X-vectors: Robust DNN Embeddings for Speaker Recognition
        Link： http://www.danielpovey.com/files/2018_icassp_xvectors.pdf
        """
        super(TemporalStatisticsPooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels*2)
        """
        mean = torch.mean(x, axis=2)
        var = torch.var(x, axis=2)
        x = torch.cat((mean, var), axis=1)
        return x


''' Self attentive weighted mean pooling.
'''


class SelfAttentivePooling(nn.Module):
    def __init__(self, dim, **kwargs):
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        # attention dim = 128
        super(SelfAttentivePooling, self).__init__()
        self.linear1 = nn.Conv1d(dim, dim, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(dim, dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        return mean


''' Attentive weighted mean and standard deviation pooling.
'''


class Attentive_Statistics_Pooling(nn.Module):
    def __init__(self, dim, **kwargs):
        # Use AttentiveStatisticsPooling and BatchNorm1d from speechbrain
        super(Attentive_Statistics_Pooling, self).__init__()
        self.pooling = AttentiveStatisticsPooling(dim)

    def forward(self, x):
        x = self.pooling(x)
        return x


# class Attentive_Statistics_Pooling(nn.Module):
#     def __init__(self, dim, **kwargs):
#         # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
#         # attention dim = 128
#         super(Attentive_Statistics_Pooling, self).__init__()
#         self.linear1 = nn.Conv1d(dim, dim, kernel_size=1) # equals W and b in the paper
#         self.linear2 = nn.Conv1d(dim, dim, kernel_size=1) # equals V and k in the paper
#
#     def forward(self, x):
#         # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
#         alpha = torch.tanh(self.linear1(x))
#         alpha = torch.softmax(self.linear2(alpha), dim=2)
#         mean = torch.sum(alpha * x, dim=2)
#         residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
#         std = torch.sqrt(residuals.clamp(min=1e-9))
#         return torch.cat([mean, std], dim=1)


if __name__ == "__main__":
    data = torch.randn(10, 128, 100)
    pooling = SelfAttentivePooling(128)
    out = pooling(data)
    print(data.shape)
    print(out.shape)
