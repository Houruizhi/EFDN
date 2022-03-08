import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys
def make_ud_model(in_channels, out_channels, up = True, num_features = 64, 
    back_projection = True, act_type = 'relu', norm_type = 'bn'):
    """
    up: ture, deconv; false, conv
    """
    if back_projection:
        stride = 2
        padding = 2
        kernel_size = 6
        if up:
            conv = DeconvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                padding=padding, act_type=act_type, norm_type=norm_type)
        else:
            conv = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                padding=padding, act_type=act_type, norm_type=norm_type)
    else:
        stride = 1
        padding = 1
        kernel_size = 3
        conv = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                padding=padding, act_type=act_type, norm_type=norm_type)
 
    return conv
class LocalVariance(nn.Module):
    def __init__(self, in_channels, win_size):
        super(LocalVariance, self).__init__()
        self.in_channels = in_channels
        weight = torch.ones(in_channels, 1, win_size, win_size)
        weight.div_(in_channels*win_size ** 2)
        self.weight_mean = nn.Parameter(weight.clone())
        self.weight_var = nn.Parameter(weight.clone())
        self.filter_mean = lambda x: F.conv2d(x, self.weight_mean, padding=win_size//2, groups=in_channels)
        self.filter_var = lambda x: F.conv2d(x, self.weight_var, padding=win_size//2, groups=in_channels)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        mean = self.filter_mean(x)
        variance = self.filter_var((x - mean)**2)
        return self.tanh(variance)
################
# Basic blocks
################

def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    elif act_type == 'shrinkage':
        layer = SoftShrinkage()
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!'%act_type)
    return layer


def norm(n_feature, norm_type='bn', affine=True):
    norm_type = norm_type.lower()
    layer = None
    if norm_type =='bn':
        layer = nn.BatchNorm2d(n_feature)
    elif norm_type =='in':
        layer = nn.InstanceNorm2d(n_feature, affine=affine)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!'%norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!'%pad_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict'%sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def ConvBlock(in_channels, out_channels, kernel_size, groups = 1, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,\
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    if pad_type and pad_type != 'zero':
        p = pad(pad_type, padding) 
        padding = 0
    else:
        p = None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, groups = groups, stride=stride, padding=padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)

class SoftShrinkage(nn.Module):
    def __init__(self):
        super(SoftShrinkage, self).__init__()
        self.lam = nn.Parameter(torch.tensor(0.1))
        # self.lam = 0.1
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x-self.lam)+self.lam
        # return torch.where((x>self.lam)|(x<-self.lam),x,x*0)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

################
# Advanced blocks
################
def feature_extract_block(in_channels, num_features, expend_scale, kernel_size, act_type='relu', norm_type=None):
    fe = []
    fe.append(ConvBlock(in_channels, expend_scale*num_features, kernel_size, padding = kernel_size//2,\
        stride = 1, act_type = act_type, norm_type = norm_type))
    fe.append(ConvBlock(expend_scale*num_features, num_features, kernel_size, padding = kernel_size//2,\
        stride = 1, act_type = act_type, norm_type = norm_type))
    return sequential(*fe)
def feature_integrate_block(num_features, out_channels, expend_scale, kernel_size, act_type='relu', norm_type=None):
    fi = []
    fi.append(ConvBlock(expend_scale*num_features, num_features, kernel_size, padding = kernel_size//2,\
        stride = 1, act_type = act_type, norm_type = norm_type))
    fi.append(ConvBlock(num_features, out_channels, 1, padding = 0,\
        stride = 1, act_type = None, norm_type = None))
    return sequential(*fi)
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), 1)
        return output

################
# Upsampler
################
class SPUpsampler(nn.Sequential):
    """
    sub-pixel convolution
    """
    def __init__(self, scale, in_channels, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels, 4 * in_channels, kernel_size = 3, padding = 1, bias = bias))
                m.append(nn.PixelShuffle(2))
                m.append(nn.ReLU(True))
        elif scale == 3:
            m.append(nn.Conv2d(in_channels, 9 * in_channels, kernel_size = 3, padding = 1, bias = bias))
            m.append(nn.PixelShuffle(3))
            m.append(nn.ReLU(True))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)

def UpsampleConvBlock(upscale_factor, in_channels, out_channels, kernel_size, stride, valid_padding=True, padding=0, bias=True,\
                 pad_type='zero', act_type='relu', norm_type=None, mode='nearest'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = ConvBlock(in_channels, out_channels, kernel_size, stride, bias=bias, valid_padding=valid_padding, padding=padding, \
                     pad_type=pad_type, act_type=act_type, norm_type=norm_type)
    return sequential(upsample, conv)




def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, \
                groups=1, act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 
        padding, groups=groups, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)


################
# helper funcs
################
def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
    padding = (kernel_size-1) // 2
    return padding
