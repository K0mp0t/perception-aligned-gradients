from torch import nn
from torch.nn import functional as F
from torchinfo import summary


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, lrelu_slope=0.01, pool_padding=0):
        super().__init__()

        self.lrelu_slope = lrelu_slope

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pooling = nn.MaxPool2d(2, 2, padding=pool_padding)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, self.lrelu_slope)
        
        x = self.pooling(x)
        
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.bn2(x)

        return x
    

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            ResBlock(1, 8), # (1, 28, 28) -> (8, 14, 14)
            ResBlock(8, 16, pool_padding=1), # (8, 14, 14) -> (16, 8, 8)
            ResBlock(16, 32) # (16, 8, 8) -> (32, 4, 4)
        )

        self.head = nn.Sequential(
            nn.AvgPool2d((4, 4)), # (32, 4, 4) -> (32, 1, 1)
            nn.Flatten(), # (32, 1, 1) -> (32,)
            nn.BatchNorm1d(32),
            nn.Linear(32, 10)
        )

    
    def forward(self, x):
        return self.head(self.blocks(x))
    

    def print_summary(self):
        summary(self, (32, 1, 28, 28))


    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def requires_grad(self):
        return next(self.parameters()).requires_grad
    
    # def disable_gradient(self):
    #     for p in self.parameters():
    #         p.requires_grad = False

    # def enable_gradient(self):
    #     for p in self.parameters():
    #         p.requires_grad = True
