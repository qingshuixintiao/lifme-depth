import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, _, _ = x.size()
        y = x.mean(dim=[2, 3])  # Global average pooling
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channel, 1, 1)  # Squeeze-and-Excite
        return x * y.expand_as(x)
class SpatialAttention(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(concat)
        return x * self.sigmoid(attention)

class RAEM(nn.Module):
    def __init__(self, channel):
        super(RAEM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )

        self.se_block = SEBlock(channel)
        self.spatial_attention = SpatialAttention(channel)

    def forward(self, x):
        residual = x  # Save the input for residual connection

        out = self.conv1(x)
        out = self.se_block(out)
        out = self.spatial_attention(out)

        # Add residual connection to the output
        out = out + residual
        return out
