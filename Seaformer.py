import paddle
from paddle import nn


def _make_divisible(v,divisor,min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * 9:
        new_v += divisor
    return new_v

def drop_path(x,drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor.floor_()
    output = x // keep_prob * random_tensor
    return output

class DropPath(nn.Layer):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Conv2d_BN(nn.Layer):
    def __init__(self,in_channels,out_channels,ks=1,stride=1,pad=0,dilation=1,groups=1):
        super().__init__()
        self.conv = nn.Conv2D(in_channels,out_channels,ks,stride,pad,dilation,groups)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class MLP(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features)
        self.dwconv = nn.Conv2D(hidden_features, hidden_features, 3, 1, 1, bias_attr=False, groups=hidden_features)
        self.act = nn.ReLU()
        self.fc2 = Conv2d_BN(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class InvertedResidual(nn.Layer):
    def __init__(
        self,
        inp: int,
        oup: int,
        ks: int,
        stride: int,
        expand_ratio: int,
        activations = None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1))
            layers.append(activations())
        layers.extend([
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks//2, groups=hidden_dim),
            activations(),
            Conv2d_BN(hidden_dim, oup, ks=1)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class StackedMV2Block(nn.Layer):
    def __init__(
            self,
            cfgs,
            stem,
            inp_channel=16,
            activation=nn.ReLU,
            width_mult=1.):
        super().__init__()
        self.stem = stem
        if stem:
            self.stem_block = nn.Sequential(
                Conv2d_BN(3, inp_channel, 3, 2, 1),
                activation()
            )
        self.cfgs = cfgs
        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t,
                                     activations=activation)
            self.add_sublayer(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        if self.stem:
            x = self.stem_block(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        return x


class SqueezeAxialPositionalEmbedding(nn.Layer):
    def __init__(self, dim, shape):
        super().__init__()
        self.pos_embed = paddle.create_parameter([1,dim,shape],dtype="float32")

    def forward(self, x):
        B, C, N = x.shape
        x = x + nn.functional.interpolate(self.pos_embed, size=[N], mode='linear', align_corners=False)
        return x


class Sea_Attention(nn.Layer):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1)
        self.to_k = Conv2d_BN(dim, nh_kd, 1)
        self.to_v = Conv2d_BN(dim, self.dh, 1,)

        self.proj = nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim))
        self.proj_encode_row = nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(2 * self.dh, 2 * self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.dh)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.dh, dim, ks=1)
        self.sigmoid = h_sigmoid()
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = paddle.concat([q, k, v], axis=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)


        qrow = self.pos_emb_rowq(q.mean(-1)).reshape((B, self.num_heads, -1, H)).transpose((0, 1, 3, 2))
        krow = self.pos_emb_rowk(k.mean(-1)).reshape((B, self.num_heads, -1, H))
        vrow = v.mean(-1).reshape((B, self.num_heads, -1, H)).transpose((0, 1, 3, 2))

        attn_row = paddle.matmul(qrow, krow) * self.scale
        attn_row = self.softmax(attn_row)
        xx_row = paddle.matmul(attn_row, vrow)
        xx_row = self.proj_encode_row(xx_row.transpose((0, 1, 3, 2)).reshape((B, self.dh, H, 1)))


        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape((B, self.num_heads, -1, W)).transpose((0, 1, 3, 2))
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape((B, self.num_heads, -1, W))
        vcolumn = v.mean(-2).reshape((B, self.num_heads, -1, W)).transpose((0, 1, 3, 2))

        attn_column = paddle.matmul(qcolumn, kcolumn) * self.scale
        attn_column = self.softmax(attn_column)
        xx_column = paddle.matmul(attn_column, vcolumn)
        xx_column = self.proj_encode_column(xx_column.transpose((0, 1, 3, 2)).reshape((B, self.dh, 1, W)))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv
        return xx


class Block(nn.Layer):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Sea_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                  activation=act_layer)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,drop=drop)


    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class BasicLayer(nn.Layer):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=None):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.LayerList()
        for i in range(self.block_num):
            self.transformer_blocks.append(Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer))

    def forward(self, x):

        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class h_sigmoid(nn.Layer):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6



class SeaFormer(nn.Layer):
    def __init__(self,
                 cfgs,
                 channels,
                 emb_dims,
                 key_dims,
                 depths=[2, 2],
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=[2, 4],
                 drop_path_rate=0.,
                 act_layer=nn.ReLU6,
                 num_classes=1000,
                 decoder_emb_dim=[128, 160, 192]
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.depths = depths
        self.cfgs = cfgs
        self.light_head = LightHead(decoder_emb_dim,num_classes)

        for i in range(len(cfgs)):
            smb = StackedMV2Block(cfgs=cfgs[i], stem=True if i == 0 else False, inp_channel=channels[i],
                                 )
            setattr(self, f"smb{i + 1}", smb)

        for i in range(len(depths)):
            dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depths[i])]
            trans = BasicLayer(
                block_num=depths[i],
                embedding_dim=emb_dims[i],
                key_dim=key_dims[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
                act_layer=act_layer)
            setattr(self, f"trans{i + 1}", trans)



    def forward(self, x):
        outputs = []
        num_smb_stage = len(self.cfgs)
        num_trans_stage = len(self.depths)
        for i in range(num_smb_stage):
            smb = getattr(self, f"smb{i + 1}")
            x = smb(x)
            if i == 1:
                outputs.append(x)
            if num_trans_stage + i >= num_smb_stage:
                trans = getattr(self, f"trans{i + num_trans_stage - num_smb_stage + 1}")
                x = trans(x)
                outputs.append(x)
        # outputs = self.light_head(outputs)
        return outputs

if __name__ == '__main__':
    model_cfgs = dict(
        cfg1=[

            [3, 3, 32, 1],
            [3, 4, 64, 2],
            [3, 4, 64, 1]],
        cfg2=[
            [5, 4, 128, 2],
            [5, 4, 128, 1]],
        cfg3=[
            [3, 4, 192, 2],
            [3, 4, 192, 1]],
        cfg4=[
            [5, 4, 256, 2]],
        cfg5=[
            [3, 6, 320, 2]],
        channels=[32, 64, 128, 192, 256, 320],
        depths=[3, 3, 3],
        key_dims=[16, 20, 24],
        emb_dims=[192, 256, 320],
        num_heads=8,
        mlp_ratios=[2, 4, 6],
        drop_path_rate=0.1,
    )

    model = SeaFormer(
        cfgs=[model_cfgs['cfg1'], model_cfgs['cfg2'], model_cfgs['cfg3'], model_cfgs['cfg4'], model_cfgs['cfg5']],
        channels=model_cfgs['channels'],
        key_dims=model_cfgs['key_dims'],
        emb_dims=model_cfgs['emb_dims'],
        depths=model_cfgs['depths'],
        num_heads=model_cfgs['num_heads'],
        mlp_ratios=model_cfgs['mlp_ratios'],
        drop_path_rate=model_cfgs['drop_path_rate'],
        num_classes=2)

    image = paddle.randn([1,3,512,512])
    out = model(image)
    for o in out:
        print(o.shape)