import paddle
from paddle import nn

from sct_sea_model import SCT_Sea, Semantic_Branch



class SCTHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 channels,
                num_classes=2,
                 dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.bn1 = nn.BatchNorm(self.in_channels)
        self.conv1 = nn.Conv2D(
            self.in_channels,
            self.channels,
            kernel_size=3,
            padding=1)  
        self.bn2 = nn.BatchNorm2D(self.channels)
        self.relu = nn.ReLU()
        self.conv_seg = nn.Conv2D(self.channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.upsample = nn.Upsample((512,512),mode="bilinear")


    def forward(self, inputs):
        x = self.upsample(inputs)
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.dropout(x)
        out = self.conv_seg(self.relu(self.bn2(x)))
        return out

class sctsea(nn.Layer):
    def __init__(self,in_channels=256,channels=128):
        super().__init__()
        self.backbone = SCT_Sea()
        self.decoder = SCTHead(in_channels,channels)
    def forward(self, x):
        return self.decoder(self.backbone(x))
