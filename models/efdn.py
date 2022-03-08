import math
import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, sequential
def make_model(args, parent=False):
    return EFDN(args)

def make_conv(in_channels, out_channels, up = True, num_features = 64, 
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

class NEBlock(nn.Module):
    def __init__(self, num_features, num_iter, back_projection, down_first, 
        groups, act_type, norm_type):
        super(NEBlock, self).__init__()
        self.num_iter = num_iter
        self.num_features = num_features

        self.blocks_1 = nn.ModuleList()
        self.blocks_2 = nn.ModuleList()
        self.compress_blocks_1 = nn.ModuleList()
        self.compress_blocks_2 = nn.ModuleList()

        for idx in range(self.num_iter):
            self.blocks_1.append(make_conv(num_features, num_features, up = not down_first, 
                back_projection=back_projection, act_type=act_type, norm_type=norm_type))
            self.blocks_2.append(make_conv(num_features, num_features, up = down_first, 
                back_projection=back_projection, act_type=None, norm_type=None))
            if idx > 0:
                self.compress_blocks_1.append(
                    CompressBlock((idx+1)*num_features, num_features,
                        groups=groups, act_type=act_type, norm_type=norm_type)
                        )
                self.compress_blocks_2.append(
                    CompressBlock((idx+1)*num_features, num_features,
                        groups=groups, act_type=act_type, norm_type=norm_type)
                        )

    def forward(self, x):

        features_first = []
        features_scaled = []

        features_first.append(x)

        for idx in range(self.num_iter):
            U = torch.cat(tuple(features_first), 1) 
            if idx > 0:
                U = self.compress_blocks_1[idx-1](U)
            D = self.blocks_1[idx](U)

            features_scaled.append(D)

            D = torch.cat(tuple(features_scaled), 1)
            if idx > 0:
                D = self.compress_blocks_2[idx-1](D)
            D = features_first[-1] - self.blocks_2[idx](D)
            
            features_first.append(D)

        output = torch.cat(tuple(features_first[1:]), 1)   # leave out input x, i.e. lr_features[0]

        return output

class FeatureExtractBlock(nn.Module):
    def __init__(self, in_channels=1, num_features=64, groups=4, kernel_size=3,
        act_type='relu', norm_type='bn'):
        super(FeatureExtractBlock, self).__init__()
        self.conv_in = ConvBlock(in_channels, 4*num_features, kernel_size=kernel_size, 
                    padding=kernel_size//2, act_type=act_type, norm_type=norm_type)
        
        self.feat_in = CompressBlock(4*num_features, num_features,
            groups=groups, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x = self.conv_in(x)
        return self.feat_in(x)

class CompressBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, groups=1, 
        act_type='relu', norm_type='bn'):
        super(CompressBlock, self).__init__()
        # self.compress = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.compress = sequential(*[
            ConvBlock(in_channels, 128, kernel_size=1, 
                act_type=act_type, norm_type=None),
            ConvBlock(128, 128, kernel_size=3, 
                groups=groups, act_type=act_type, norm_type=norm_type),
            ConvBlock(128, out_channels, kernel_size=1, 
                act_type=None, norm_type=None)
            ])
    def forward(self, x):
        return self.compress(x)

class EFDN(nn.Module):
    def __init__(self, args):
        super(EFDN, self).__init__()
        in_channels = args.in_channels
        out_channels = args.out_channels
        num_features = args.num_features 
        num_iter = args.num_iter
        back_projection = args.back_projection
        down_first = args.down_first
        groups = args.groups
        act_type = args.act_type
        norm_type = args.norm_type

        self.fe = FeatureExtractBlock(in_channels=in_channels, num_features=num_features, kernel_size = 3,
            groups=1, act_type=act_type, norm_type=norm_type)

        self.compress = CompressBlock(num_iter*num_features, num_features, 
            groups = groups, act_type=act_type, norm_type=norm_type)
        
        self.body = NEBlock(num_features, num_iter, back_projection, 
            down_first, groups=groups, act_type=act_type, norm_type=norm_type)
        
        self.reconstruct = ConvBlock(num_features, out_channels, kernel_size=3, 
            act_type=None, norm_type=None)
            
    def forward(self, x):
        inputs = x
        x = self.fe(x)
        x = self.body(x)
        x = self.compress(x)
        return inputs - self.reconstruct(x)