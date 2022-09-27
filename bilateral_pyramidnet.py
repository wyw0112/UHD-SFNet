import torch
import torch.nn as nn
from unet_model import UNet
from einops import rearrange
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, 
                 use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, 
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) # norm to [0,1] NxHxWx1
        hg, wg = hg*2-1, wg*2-1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide)
        return coeff.squeeze(2)

class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=bn)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)

        return output

class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):

        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class EncodeBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU()
        )
        self.pad1 = nn.AvgPool2d(4)
        '''
        self.MLP1 = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.LayerNorm(4)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.LayerNorm(4)
        )
        self.MLP3 = nn.Sequential(
            nn.Linear(dim, dim*3),
            nn.LayerNorm(dim*3),
            nn.ReLU(),
            nn.Linear(dim*3, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        '''
        self.layer1 = nn.Sequential(
            #nn.Linear(dim*4*4, dim*2*2),
            nn.Linear(dim*16*16, dim*2*2),
            #nn.LayerNorm(dim*2*2)
        )
        self.layer2 = nn.Linear(dim*2*2, 1)
        self.acti = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.pad1(x)
        #x = self.MLP1(x).transpose(3, 2)
        #x = self.MLP2(x).transpose(1, 3)
        #x = self.MLP3(x).permute(0,3,1,2)
        x = x.flatten(start_dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.acti(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, channel, 128, 128))
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x, patch):
        y = x.clone()
        #x = x+self.pos_embed
        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        for i in range(x.shape[0]):
            temp = x[i, :, :, :].unsqueeze(0)
            self.ph = patch[i]
            self.pw = patch[i]

            # Global representations
            _, _, h, w = temp.shape
            temp = rearrange(temp, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
            temp = self.transformer(temp)
            temp = rearrange(temp, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
            x[i, :, :, :] = temp
            # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.Enblock = EncodeBlock(dim=3, hidden_dim=16)
        self.mvit = MobileViTBlock(dim=16, depth=3, channel=3, kernel_size=3, patch_size=(16, 16), mlp_dim=128)

    def forward(self, x, patchnum):
        #patchnum = self.Enblock(x_t)
        #patchnum = 2**torch.round(3 * (patchnum) + 2).type(torch.int)
        #print('\npatchnum:', patchnum)
        #patchnum = torch.ones(x.shape[0]).type(torch.int)*16
        x = self.mvit(x, patchnum) + x
        return x




class BilateralNetwork(nn.Module):
    def __init__(self, size=256):
        super(BilateralNetwork, self).__init__()
        self.unet_1 = UNet(n_channels=3)
        self.unet_2 = UNet(n_channels=3)
        self.encode = EncodeBlock(dim=3, hidden_dim=16)
        self.net_1 = MobileViT()
        self.net_2 = MobileViT()
        self.guide = GuideNN()
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()
        self.conv = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, content):
        local_c = F.interpolate(content, (128, 128),mode='bilinear', align_corners=True)
        local_feature = F.interpolate(content, (256, 256),
                            mode='bilinear', align_corners=True)
        x1 = F.interpolate(content, (128, 128),
                            mode='bilinear', align_corners=True)
        x1_t = F.interpolate(content, (64, 64),
                             mode='bilinear', align_corners=True)
        #patchesnum = self.encode(x1_t)
        #patchesnum = 2**torch.round(3 * patchesnum + 2).type(torch.int)
        patchesnum = torch.ones(x1.shape[0]).type(torch.int)*16
        x1 = self.net_1(x1, patchesnum)
        #x1_t = F.interpolate(x1, (64, 64),
        #                     mode='bilinear', align_corners=True)
        #x1 = F.interpolate(x1, (128, 128),
        #                     mode='bilinear', align_corners=True)
                            
        x2 = self.unet_1(0.5 * self.net_2(x1, patchesnum) + 0.5 * local_c)
        
        local_feature = self.unet_2(local_feature)
        local_feature = F.interpolate(local_feature, (content.shape[2], content.shape[3]),
                           mode='bilinear', align_corners=True)
        guide = self.guide(content)
        slice_coeffs = self.slice(x2.view(-1, 12, 16, 16, 16), guide)
        output = self.apply_coeffs(slice_coeffs, content)
        
        return self.relu(self.conv(output) + local_feature) * content



