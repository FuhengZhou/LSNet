import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from biformer1 import BiLevelRoutingAttention
from mmcv.cnn import ConvModule, build_norm_layer
from thop import clever_format, profile
from einops import rearrange


def get_padding(input_size, output_size, kernel_size, stride, padding='SAME'):
    if padding == 'VALID':
        return 0

    if padding == 'SAME':
        pad_total = ((output_size - 1) * stride + kernel_size - input_size)//2
        return pad_total

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def get_stride(input_size, output_size, kernel_size, padding):
        stride = (input_size + 2 * padding - kernel_size)//(output_size - 1)
        return stride
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size = 3):
        super(SpatialAttention,self).__init__()
        padding = 3//2
        self.conv1 = nn.Conv2d(2,1,kernel_size,1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out_avg = torch.mean(x, dim=1, keepdim=True)
        out_max,_ = torch.max(x, dim=1, keepdim=True)
        out_pool = torch.cat([out_max,out_avg],dim=1)
        out = self.conv1(out_pool)
        out = self.sigmoid(out)

        return out*x
class CBAM(nn.Module):
    def __init__(self, channel, ratio=3, kernel_size=3):
        super(CBAM,self).__init__()
        # 定义部分
        self.channel_attention = SEBlock(channel,ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        out_channel_attention = self.channel_attention(x)
        out_spatial_attention = self.spatial_attention(out_channel_attention)
        return out_spatial_attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=3):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.fc(x)
        w, b = w.split(w.data.size(1) // 2, dim=1)
        w = torch.sigmoid(w)
        return x * w + b
class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True,bia=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=bia)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act((self.conv(x)))

class Mutiple_Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Mutiple_Conv, self).__init__()
        self.conv1 = Conv(c1, c2, 3, s, g=g)
        self.conv2 = Conv(c1, c2, 5, s, g=g)
        self.conv3 = Conv(c1, c2, 7, s, g=g)
        self.act = CBAM(c2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = x1 + x2 + x3
        x = self.act(x)
        return x

class res_conv(nn.Module):

    def __init__(self, si, so, c1, c2, k=3, s=None, p=1, g=2, act=True):
        super(res_conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=get_stride(si, so, k, p), padding=p,  groups=g, bias=False)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act((self.conv(x)))


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape
class Conv_C3(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act((self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.SiLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, p=1, d=1, g=1,
                 padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert p == 1

        padding_11 = p - k // 2

        self.nonlinearity = nn.SiLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=k,
                                         stride=s,
                                         padding=p, dilation=d, groups=g, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=c1) if c1 == c2 and s == 1 else None
            self.rbr_dense = conv_bn(in_channels=c1, out_channels=c2, kernel_size=k,
                                     stride=s, padding=p, groups=g)
            self.rbr_1x1 = conv_bn(in_channels=c1, out_channels=c2, kernel_size=1, stride=s,
                                   padding=padding_11, groups=g)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

class SimConv(nn.Module):

    def __init__(self, c1, c2, k, s, groups=1, bias=False, padding=None):
        super().__init__()
        if padding is None:
            padding = k // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=padding, groups=groups, bias=bias, )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act((self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x

class BottleRep(nn.Module):

    def __init__(self, c1, c2, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(c1, c2)
        self.conv2 = basic_block(c1, c2)
        if c1 != c2:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs

class InjectionMultiSum_Auto_pool2(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=nn.ReLU6,
            global_inp=None,
    ) -> None:
        super().__init__()
        self.norm_cfg = norm_cfg

        if not global_inp:
            global_inp = inp

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x):
        x_l, x_g = x
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)

        if use_pool:
            avg_pool = nn.functional.adaptive_avg_pool2d
            output_size = np.array([H, W])


            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out



class PyramidPoolAgg(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_max_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        output_size = np.array([H, W])

        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d

        out = [self.pool(inp, output_size) for inp in inputs]

        return torch.cat(out, dim=1)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class top_Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio



        self.attn = BiLevelRoutingAttention(dim=dim, qk_dim=key_dim, num_heads=num_heads, auto_pad=True,
                                            attn_ratio=attn_ratio)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1

class TopBasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=[x.item() for x in torch.linspace(0, 0.1, 3)],
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg, act_layer=act_layer))

    def forward(self, x):

        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

class AdvPoolFusion(nn.Module):
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d

        N, C, H, W = x2.shape
        output_size = np.array([H, W])

        x1 = self.pool(x1, output_size)

        return torch.cat([x1, x2], 1)

class BepC2f(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv_C3(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv_C3((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(BottleRep(self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape


        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d
        output_size = np.array([H, W])
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out

class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, c2):
        super().__init__()


        self.cv1 = Conv(in_channel_list[0], c2, 1, 1)
        self.cv_fuse = Conv(c2 * 3, c2, 1, 1)
        self.downsample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        N, C, H, W = x[1].shape


        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
        output_size = np.array([H, W])

        x0 = self.downsample(x[0], output_size)

        x1 = self.cv1(x[1])
        x2 = F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))

class finall_Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(finall_Conv, self).__init__()
        self.conv1 =nn.Sequential(
            nn.Conv2d(c1, 12, 3, s, autopad(3, p), groups=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(12, 9, 3, s, autopad(3, p), groups=3, bias=True),
            nn.SiLU()
        )
        self.conv2 = nn.Conv2d(9, c2, k, s, autopad(k, p), groups=g, bias=True)
    def forward(self, x):
        return self.conv2((self.conv1(x)))



class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.d_res_conv_r = Conv(1, 16,k=1)
        self.d_res_conv_g = Conv(1, 16, k=1)
        self.d_res_conv_b = Conv(1, 16, k=1)

        self.o_res_conv_r = Conv(1, 16, k=1)
        self.o_res_conv_g = Conv(1, 16, k=1)
        self.o_res_conv_b = Conv(1, 16, k=1)

        self.d_res_conv_rgb = Conv(3, 3, k=3)
        self.o_res_conv_rgb = Conv(3, 3, k=3)

        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)

        self.conv3 = nn.Sequential(Conv(51, 32, k=1,bia=True),
                                   Conv(32, 3, k=1,bia=True)
                                   )
        self.conv4 = nn.Sequential(Conv(51, 32, k=1,bia=True),
                                   Conv(32, 3, k=1,bia=True)
                                   )
        self.conv5 =Conv(3,3,k=1)

        self.d_biformer_rgb = BiLevelRoutingAttention(dim=12, qk_dim=48, topk=32, num_heads=2, n_win=16)
        self.o_biformer_rgb = BiLevelRoutingAttention(dim=12, qk_dim=48, topk=32, num_heads=2, n_win=16)

        self.finallconv = finall_Conv(3, 3)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        xr, xg, xb = torch.chunk(x, 3, dim=1)



        d_x_rgb = self.d_res_conv_rgb(x)
        d_xr = self.d_res_conv_r(xr)
        d_xg = self.d_res_conv_g(xg)
        d_xb = self.d_res_conv_b(xb)
        d_x_rgb=rearrange(d_x_rgb,'b c (h1 h) (w1 w)  -> b (h1 w1 c) h w',h1=2,w1=2)
        d_x_rgb = self.d_biformer_rgb(d_x_rgb, 1)
        d_x_rgb = self.up1(d_x_rgb)


        o_x_rgb = self.o_res_conv_rgb(x)
        o_xr = self.o_res_conv_r(xr)
        o_xg = self.o_res_conv_g(xg)
        o_xb = self.o_res_conv_b(xb)
        o_x_rgb = rearrange(o_x_rgb, 'b c (h1 h) (w1 w)  -> b (h1 w1 c) h w', h1=2, w1=2)
        o_x_rgb = self.o_biformer_rgb(o_x_rgb, 2)
        o_x_rgb = self.up2(o_x_rgb)

        d_x=torch.cat([d_xr,d_xg,d_xb,d_x_rgb],dim=1)
        o_x = torch.cat([o_xr, o_xg, o_xb, o_x_rgb], dim=1)
        d_x=self.conv3(d_x)
        o_x = self.conv4(o_x)
        out=x+d_x-o_x




        return out

if __name__ == '__main__':
    model = model()
    input = torch.randn([1, 3, 256, 256])
    flops, params = profile(model, (input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print("Flops:", flops)
    print("Params:", params)

