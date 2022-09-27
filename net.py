import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from unet_model import UNet
import torch
from mlp_mixer_pytorch import MLPMixer
import numpy as np
import cv2

model = MLPMixer(
    image_size = 256,
    channels = 3,
    patch_size = 16,
    dim = 512,
    depth = 12,
    num_classes = 1000
)

#img = torch.randn(1, 3, 256, 256)
#pred = model(img) # (1, 1000)
##########################################################################
## Layer Norm

class MLP(nn.Module):
    def __init__(self, num_classes, out_classes):
        super(MLP, self).__init__()
        self.mlp1 = nn.Sequential(
            #nn.Linear(num_classes, num_classes),
            #nn.Sigmoid(),
            nn.Linear(num_classes, out_classes),
            nn.Sigmoid()
        ).cuda()
        self.mlp2 = nn.Sequential(
            #nn.Linear(num_classes, num_classes),
            #nn.Sigmoid(),
            nn.Linear(num_classes, out_classes),
            nn.Sigmoid()
        ).cuda()
    def forward(self, x):
        x1 = self.mlp1(x).transpose(3, 2)
        x1 = self.mlp2(x1).transpose(3, 2)
        return x1
class c_MLP(nn.Module):
    def __init__(self, dim, out_dim):
        super(c_MLP, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.Sigmoid(),
            #nn.Linear(out_dim, out_dim),
            #nn.Sigmoid()
        ).cuda()
    def forward(self, x):
        x1 = self.mlp1(x.transpose(1, 3))
        return x1.transpose(1, 3)

class jubu(nn.Module):
    def __init__(self, dim, num_classes):
        super(jubu, self).__init__()
        self.dim = dim
        self.conv1 = nn.Sequential(
            nn.LayerNorm([num_classes, num_classes]),
            nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.LayerNorm([num_classes, num_classes]),
            nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.LayerNorm([num_classes, num_classes]),
            nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.u_net_pro = UNet(n_channels=dim, out_channels=dim)
    def forward(self, x):
        #x1 = self.conv1(x)
        #x1 = self.conv2(x1)
        #x1 = self.conv3(x1)
        x1 = self.u_net_pro(x)
        return x1

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias = True):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_classes):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads*num_heads, 1, 1))
        '''
        self.mlp_q = []
        self.mlp_k = []
        self.mlp_v = []
        for i in range(num_heads):
            self.mlp_q.append(MLP(num_classes, num_heads))
            self.mlp_k.append(MLP(num_classes, num_heads))
            self.mlp_v.append(MLP(num_classes, num_heads))
        '''
        self.mlp_q = MLP(num_classes, int(num_classes/16))
        self.mlp_k = MLP(num_classes, int(num_classes/16))
        self.mlp_v = MLP(num_classes, int(num_classes/16))
        self.mlp = c_MLP(dim, dim*16*16)
        self.c = nn.Conv2d(dim, dim*16*16, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        b, c, h, w = x.shape
        c_t = c*256
        h_t = int(h/16)
        w_t = int(w/16)
        #x = rearrange(x, 'b c h w -> b head c (h w)/head', hhead=self.num_heads)
        #x = x.reshape([b, self.num_heads, c, int(h*w/self.num_heads)])
        '''
        att = torch.zeros([b, c, h, w])
        t = int(h/self.num_heads)
        for i in range(self.num_heads):
            q = self.mlp_q[i](x)
            k = self.mlp_k[i](x)
            v = self.mlp_v[i](x)
            q = rearrange(q, 'b c h w -> b c (h w)')
            k = rearrange(k, 'b c h w -> b c (h w)')
            v = rearrange(v, 'b c h w -> b c (h w)')
            temp = (q @ k.transpose(-2, -1)) * self.temperature[i]
            temp = temp.softmax(dim=-1)
            att[:, :, i*t:(i+1)*t, i*t:(i+1)*t] = (temp @ v).reshape(b, c, t, t)
        '''
        temp = self.c(x)
        q = self.mlp_q(temp)
        k = self.mlp_k(temp)
        v = self.mlp_v(temp)
        if self.num_heads == 1:
            q = q.reshape(b, 1, c_t, h_t*w_t)
            k = k.reshape(b, 1, c_t, h_t*w_t)
            v = v.reshape(b, 1, c_t, h_t*w_t)
        else:
            q = rearrange(q, 'b c (h head1) (w head2) -> b (head1 head2) c (h w)', head1=self.num_heads, head2=self.num_heads)
            k = rearrange(k, 'b c (h head1) (w head2) -> b (head1 head2) c (h w)', head1=self.num_heads, head2=self.num_heads)
            v = rearrange(v, 'b c (h head1) (w head2) -> b (head1 head2) c (h w)', head1=self.num_heads, head2=self.num_heads)
        #q = rearrange(q, 'b c h w -> b c (h w)')
        #k = rearrange(k, 'b c h w -> b c (h w)')
        #v = rearrange(v, 'b c h w -> b c (h w)')

        #q = torch.nn.functional.normalize(q, dim=-1)
        #k = torch.nn.functional.normalize(k, dim=-1)
        
        #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        temp = (q @ k.transpose(-2, -1)) * self.temperature
        temp = temp.softmax(dim=-1)
        att = (temp @ v)
        #att = rearrange(att, 'b (head1 head2) c (h w) -> b c (h head1) (w head2)', head1=self.num_heads, head2=self.num_heads, h=int(h_t/self.num_heads), w=int(w_t/self.num_heads))
        att = att.reshape(b, c, h, w)
        return att
'''


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        #q = torch.nn.functional.normalize(q, dim=-1)
        #k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        #attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
'''
##########################################################################
class quanju(nn.Module):
    def __init__(self, dim, head_num, num_classes):
        super(quanju, self).__init__()
        #self.norm1 = nn.LayerNorm([num_classes, num_classes]).cuda()
        self.attn1 = Attention(dim, head_num, num_classes).cuda()
        self.feed1 = FeedForward(dim, 2)
        #self.feed2 = FeedForward(dim, 2)
        #self.norm2 = nn.LayerNorm([num_classes, num_classes]).cuda()
        #self.attn2 = Attention(dim, head_num, num_classes).cuda()


    def forward(self, x):
        x1 = self.attn1(x).cuda()
        x1 = self.feed1(x1).cuda()
        #x1 = self.feed2(x1).cuda()
        #x = x + self.attn1(self.norm1(x)).cuda()
        #x = x + self.attn2(self.norm2(x)).cuda()
        return x + x1
class con(nn.Module):
    def __init__(self, num_classes, dim):
        super(con, self).__init__()
        self.conv1 = nn.Sequential(
            nn.LayerNorm([num_classes, num_classes]),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            #nn.LayerNorm([num_classes, num_classes]),
            nn.Conv2d(dim, int(dim/2), kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(dim, int(dim), kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 12,
        num_blocks = [2,3,3,5],
        num_refinement_blocks = 2,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super(Restormer, self).__init__()
        '''
        self.begin1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        '''
        num_classes = 512
        dim = 3
        heads = [1, 2, 4, 8]
        self.begin1 = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.begin2 = nn.Conv2d(in_channels=6, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.img_jubu1 = jubu(dim, num_classes)
        self.img_quanju1 = quanju(dim, heads[0], num_classes)
        self.img_con1 = con(num_classes, dim*2)
        self.fft_jubu1 = jubu(dim, num_classes)
        self.fft_quanju1 = quanju(dim, heads[0], num_classes)
        self.fft_con1 = con(num_classes, dim*2)

        self.img_jubu2 = jubu(dim, num_classes)
        self.img_quanju2 = quanju(dim, heads[1], num_classes)
        self.img_con2 = con(num_classes, dim*2)
        self.fft_jubu2 = jubu(dim, num_classes)
        self.fft_quanju2 = quanju(dim, heads[1], num_classes)
        self.fft_con2 = con(num_classes, dim*2)

        self.con = con(num_classes, dim*2)
        self.end = nn.Sequential(
            #nn.LayerNorm([num_classes, num_classes]),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            #nn.LayerNorm([num_classes, num_classes]),
            nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

    def forward(self, in_img):
        img = F.interpolate(in_img, (512, 512), mode='bicubic', align_corners=True)
        img_fft = torch.fft.fft2(img)
        img_amp, img_pha = torch.abs(img_fft), torch.angle(img_fft)
        fft = torch.cat([img_amp, img_pha], 1)

        img1 = self.begin1(img)
        img_j = self.img_jubu1(img1)
        img_q = self.img_quanju1(img1)
        img_q = self.img_quanju2(img_q)
        t_img = torch.cat([img_j, img_q], 1)
        img2 = self.img_con1(t_img)

        fft1 = self.begin2(fft)
        fft_j = self.fft_jubu1(fft1)
        fft_q = self.fft_quanju1(fft1)
        fft_q = self.fft_quanju2(fft_q)
        t_fft = torch.cat([fft_j, fft_q], 1)
        fft2 = self.fft_con1(t_fft)
        '''
        #img1 = self.begin1(img)
        img_q = self.img_quanju1(img1)
        img_j = self.img_jubu1(img1)
        t_img1 = torch.cat([img_j, img_q], 1)
        img2 = self.img_con1(t_img1) + img1

        img_q = self.img_quanju2(img2)
        img_j = self.img_jubu2(img2)
        t_img2 = torch.cat([img_j, img_q], 1)
        img3 = self.img_con2(t_img2) + img2
        
        img_q = self.img_quanju3(img3)
        img_j = self.img_jubu3(img3)
        t_img3 = torch.cat([img_j, img_q], 1)
        img4 = self.img_con3(t_img3) + img3

        img_q = self.img_quanju4(img4)
        img_j = self.img_jubu4(img4)
        t_img4 = torch.cat([img_j, img_q], 1)
        img5 = self.img_con4(t_img4) + img4

        fft1 = self.begin2(fft)
        fft_j = self.fft_jubu1(fft1)
        fft_q = self.fft_quanju1(fft1)
        t_fft1 = torch.cat([fft_j, fft_q], 1)
        fft2 = self.fft_con1(t_fft1) + fft1

        fft_j = self.fft_jubu2(fft2)
        fft_q = self.fft_quanju2(fft2)
        t_fft2 = torch.cat([fft_j, fft_q], 1)
        fft3 = self.fft_con2(t_fft2) + fft2

        fft_j = self.fft_jubu3(fft3)
        fft_q = self.fft_quanju3(fft3)
        t_fft3 = torch.cat([fft_j, fft_q], 1)
        fft4 = self.fft_con3(t_fft3) + fft3

        fft_j = self.fft_jubu4(fft4)
        fft_q = self.fft_quanju4(fft4)
        t_fft4 = torch.cat([fft_j, fft_q], 1)
        fft5 = self.fft_con4(t_fft4) + fft4
        '''
        #out = fft5+img5
        out = torch.cat([img2, fft2], 1)
        out = self.con(out)
        out = self.end(out)
        out = F.interpolate(out, (in_img.shape[2], in_img.shape[3]), mode='bicubic', align_corners=True)
        out = out*in_img

        return out
