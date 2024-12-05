import paddle
from paddle import nn
from Seaformer import Block
import random


class Sea_Block(nn.Layer):
    def __init__(self,in_dim,dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU):
        super().__init__()
        self.base_block = Block(dim, key_dim, num_heads, mlp_ratio=mlp_ratio, attn_ratio=attn_ratio, drop=drop,
                 drop_path=drop_path, act_layer=act_layer)
        self.conv1 = nn.Conv2D(in_dim,dim,3,2,1)
        self.conv2 = nn.Conv2D(dim,dim,1)
        self.conv2_1 = nn.Conv2D(dim,dim,1)
        self.conv3 = nn.Conv2D(dim,dim,3,1,1)
        self.bn1 = nn.BatchNorm2D(dim)
        self.act = act_layer()
    def forward(self,x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn1(self.conv2(x)))
        x = self.base_block(x)
        x = self.base_block(x)
        x = self.act(self.bn1(self.conv2_1(x)))
        x = self.act(self.bn1(self.conv3(x)))
        return x

class Semantic_Branch(nn.Layer):
    def __init__(self,in_dim,embed_dim=[128,256,320],key_dim=[16,20,24],num_heads=8):
        super().__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2D(in_dim,in_dim*2,3,2,1),
            nn.BatchNorm2D(in_dim),
            nn.ReLU(),
            nn.Conv2D(in_dim*2,in_dim,3,2,1),
            nn.BatchNorm2D(in_dim),
            nn.ReLU()
            )
        self.layers = nn.LayerList()
        self.layers.append(Sea_Block(in_dim,embed_dim[0],key_dim[0],num_heads))
        for i in range(1,len(embed_dim)):
            self.layers.append(Sea_Block(embed_dim[i-1],embed_dim[i],key_dim[i],num_heads))
    def forward(self,x):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return outs

class SCT_Sea(nn.Layer):
    def __init__(self,
                 layer_nums=[2, 2, 2],
                 base_channels=64,
                 spp_channels=128,
                 in_channels=3,
                 num_heads=8,
                 drop_rate=0.,
                 ):
        super().__init__()
        self.base_channels = base_channels
        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels,base_channels,3,2,1),
            nn.BatchNorm2D(base_channels),
            nn.ReLU(),
            nn.Conv2D(base_channels,base_channels,3,2,1),
            nn.BatchNorm2D(base_channels),
            nn.ReLU()
            )
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(
            BasicBlock,base_channels,base_channels,layer_nums[0])
        self.layer2 = self._make_layer(
            BasicBlock, base_channels, base_channels * 2, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, base_channels * 2, base_channels * 4, layer_nums[2], stride=2)
        self.layer3_2 = CFBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 4,
            num_heads=num_heads,
            drop_rate=drop_rate)
        self.convdown4 = nn.Sequential(
            nn.Conv2D(
                base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(base_channels*8),
            nn.ReLU())
        self.layer4 = CFBlock(
            in_channels=base_channels * 8,
            out_channels=base_channels * 8,
            num_heads=num_heads,
            drop_rate=drop_rate)
        self.layer5 = CFBlock(
            in_channels=base_channels * 8,
            out_channels=base_channels * 8,
            num_heads=num_heads,
            drop_rate=drop_rate)
        self.spp = DAPPM_head(
            base_channels * 8, spp_channels, base_channels * 2)
        self.semantic_branch = Semantic_Branch(64) 
        self.fitblock1 = nn.Sequential(
            nn.Conv2D(256,128,1),
            nn.BatchNorm2D(128),
            nn.ReLU()
        )
        self.fitblock2 = nn.Sequential(
            nn.Upsample((16,16),mode="bilinear"),
            nn.Conv2D(256,512,1),
            nn.BatchNorm2D(512),
            nn.ReLU()
        )
        self.fitblock3 = nn.Sequential(
            nn.Conv2D(832,512,1),
            nn.BatchNorm2D(512),
            nn.ReLU()
        )
        self.convdown2 = nn.Conv2D(1024,512,1)



    def _make_layer(self,block,in_channels,out_channels,blocks,stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
        nn.Conv2D(
                in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2D(out_channels))

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=True))
            else:
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self,x):
        stem = self.conv1(x)
        xs = self.semantic_branch(stem)
        x1 = self.layer1(stem)
        x2 = self.layer2(self.relu(x1))
        x2 = paddle.concat([x2,xs[0]],axis=1)
        x2 = self.fitblock1(x2)
        x3_1 = self.layer3(self.relu(x2))
        x3 = self.layer3_2(self.relu(x3_1))
        x4_down=self.convdown4(x3)
        x4_down = paddle.concat([x4_down,self.fitblock2(xs[1])],axis=1)
        x4_down = self.relu(self.convdown2(x4_down))
        x4 = self.layer4(self.relu(x4_down))
        x5 = self.layer5(self.relu(x4))
        x5 = self.fitblock3(paddle.concat([x5,xs[2]],axis=1))
        x6 = self.spp(x5)
        x7 = nn.functional.interpolate(
            x6, size=x2.shape[2:])
        x_out = paddle.concat([x2, x7], axis=1)
        return x_out

class MLP(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.BatchNorm2D(in_channels)
        self.conv1 = nn.Conv2D(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2D(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)
    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class ConvolutionalAttention(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8):
        super(ConvolutionalAttention,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.softmax = nn.Softmax(axis=3)
        self.norm = nn.BatchNorm2D(in_channels)
        self.kv = paddle.create_parameter([inter_channels,in_channels,7,1],dtype="float32")
        self.kv3 = paddle.create_parameter([inter_channels,in_channels,1,7],dtype="float32")
    def _act_dn(self, x):
        x_shape = x.shape
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])
        x = self.softmax(x)
        x = x / (paddle.sum(x, axis=2, keepdim=True) + 1e-06)
        x = x.reshape([x_shape[0], self.inter_channels, h, w])
        return x
    def forward(self,x):
        x = self.norm(x)
        x1 = nn.functional.conv2d(x,self.kv,stride=1,padding=(3,0))
        x1 = self._act_dn(x1)
        x1 = nn.functional.conv2d(x1,self.kv.transpose([1,0,2,3]),stride=1,padding=(3,0))
        x3 = nn.functional.conv2d(x,self.kv3,stride=1,padding=(0,3))
        x3 = self._act_dn(x3)
        x3 = nn.functional.conv2d(x3,self.kv3.transpose([1,0,2,3]),stride=1,padding=(0,3))
        x = x1 + x3
        return x

class CFBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.):
        super().__init__()
        in_channels_l = in_channels
        out_channels_l = out_channels
        self.attn_l = ConvolutionalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=64,
            num_heads=num_heads)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
    def forward(self, x):
        x_res = x
        x = x_res + self.attn_l(x)
        x = x + self.mlp_l(x)
        return x

class DAPPM_head(nn.Layer):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2D(
                in_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels, inter_channels, kernel_size=1))
        self.scale2 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2D(
                in_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels, inter_channels, kernel_size=1))
        self.scale3 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2D(
                in_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels, inter_channels, kernel_size=1))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.BatchNorm2D(
                in_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels, inter_channels, kernel_size=1))
        self.scale0 = nn.Sequential(
            nn.BatchNorm2D(
                in_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels, inter_channels, kernel_size=1))
        self.process1 = nn.Sequential(
            nn.BatchNorm2D(
                inter_channels),
            nn.ReLU(),
            nn.Conv2D(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process2 = nn.Sequential(
            nn.BatchNorm2D(
                inter_channels),
            nn.ReLU(),
            nn.Conv2D(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process3 = nn.Sequential(
            nn.BatchNorm2D(
                inter_channels),
            nn.ReLU(),
            nn.Conv2D(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.process4 = nn.Sequential(
            nn.BatchNorm2D(
                inter_channels),
            nn.ReLU(),
            nn.Conv2D(
                inter_channels,
                inter_channels,
                kernel_size=3,
                padding=1))
        self.compression = nn.Sequential(
            nn.BatchNorm2D(
                inter_channels * 5),
            nn.ReLU(),
            nn.Conv2D(
                inter_channels * 5,
                out_channels,
                kernel_size=1))
        self.shortcut = nn.Sequential(
            nn.BatchNorm2D(
                in_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x_shape = x.shape[2:]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1((nn.functional.interpolate(
                self.scale1(x), size=x_shape) + x_list[0])))
        x_list.append((self.process2((nn.functional.interpolate(
            self.scale2(x), size=x_shape) + x_list[1]))))
        x_list.append(
            self.process3((nn.functional.interpolate(
                self.scale3(x), size=x_shape) + x_list[2])))
        x_list.append(
            self.process4((nn.functional.interpolate(
                self.scale4(x), size=x_shape,) + x_list[3])))

        out = self.compression(paddle.concat(x_list, axis=1)) + self.shortcut(x)
        return out

class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2D(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out if self.no_relu else self.relu(out)

