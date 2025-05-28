import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --------------------- Helpers for LayerNorm ---------------------

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = torch.Size([normalized_shape])

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = torch.Size([normalized_shape])

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, norm_type='WithBias'):
        super().__init__()
        if norm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# --------------------- Transformer Components ---------------------

class FeedForward(nn.Module):
    def __init__(self, dim, expansion, bias):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, bias):
        super().__init__()
        self.num_heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (hd c) h w -> b hd c (h w)', hd=self.num_heads)
        k = rearrange(k, 'b (hd c) h w -> b hd c (h w)', hd=self.num_heads)
        v = rearrange(v, 'b (hd c) h w -> b hd c (h w)', hd=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b hd c (h w) -> b (hd c) h w', hd=self.num_heads, h=h, w=w)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, expansion, bias, norm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, norm_type)
        self.attn = Attention(dim, heads, bias)
        self.norm2 = LayerNorm(dim, norm_type)
        self.ffn = FeedForward(dim, expansion, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# --------------------- Prompt Generation ---------------------

class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim, prompt_len, prompt_size, lin_dim):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear = nn.Linear(lin_dim, prompt_len)
        self.conv = nn.Conv2d(prompt_dim, prompt_dim, 3, padding=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1))
        wts = F.softmax(self.linear(emb), dim=1)
        p = (wts.view(b, -1, 1, 1, 1) * self.prompt_param).sum(dim=1)
        p = F.interpolate(p, (h, w), mode='bilinear', align_corners=False)
        return self.conv(p)

# --------------------- SegNet-style Pool/Unpool ---------------------

class DownsampleSegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        out, idx = self.pool(x)
        return out, idx

class UpsampleSegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x, idx, output_size):
        up = self.unpool(x, idx, output_size=output_size)
        return self.conv(up)

# --------------------- PromptIR with SegNet ---------------------

class PromptIR_SegNet(nn.Module):
    def __init__(self,
                 inp_channels=3, out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8], num_refine=4,
                 heads=[1, 2, 4, 8], expansion=2.66,
                 bias=False, norm_type='WithBias'):
        super().__init__()
        # Patch embed
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, padding=1, bias=bias)
        # Encoder
        self.enc1 = nn.Sequential(*[TransformerBlock(dim, heads[0], expansion, bias, norm_type) for _ in range(num_blocks[0])])
        self.down1 = DownsampleSegNet(dim, dim*2)
        self.enc2 = nn.Sequential(*[TransformerBlock(dim*2, heads[1], expansion, bias, norm_type) for _ in range(num_blocks[1])])
        self.down2 = DownsampleSegNet(dim*2, dim*4)
        self.enc3 = nn.Sequential(*[TransformerBlock(dim*4, heads[2], expansion, bias, norm_type) for _ in range(num_blocks[2])])
        self.down3 = DownsampleSegNet(dim*4, dim*8)
        # Bottleneck
        self.latent = nn.Sequential(*[TransformerBlock(dim*8, heads[3], expansion, bias, norm_type) for _ in range(num_blocks[3])])
        # Prompts + noise + reduce
        self.prompt3 = PromptGenBlock(dim*8, 5, prompt_size=32, lin_dim=dim*8)
        self.noise3  = TransformerBlock(dim*8*2, heads[2], expansion, bias, norm_type)
        self.reduce3 = nn.Conv2d(dim*8*2, dim*8, 1, bias=bias)
        self.prompt2 = PromptGenBlock(dim*4, 5, prompt_size=64, lin_dim=dim*4)
        self.noise2  = TransformerBlock(dim*4*2, heads[1], expansion, bias, norm_type)
        self.reduce2 = nn.Conv2d(dim*4*2, dim*4, 1, bias=bias)
        self.prompt1 = PromptGenBlock(dim*2, 5, prompt_size=128, lin_dim=dim*2)
        self.noise1  = TransformerBlock(dim*2*2, heads[0], expansion, bias, norm_type)
        self.reduce1 = nn.Conv2d(dim*2*2, dim*2, 1, bias=bias)
        # Decoder
        self.up3 = UpsampleSegNet(dim*8, dim*4)
        self.dec3= nn.Sequential(*[TransformerBlock(dim*4, heads[2], expansion, bias, norm_type) for _ in range(num_blocks[2])])
        self.up2 = UpsampleSegNet(dim*4, dim*2)
        self.dec2= nn.Sequential(*[TransformerBlock(dim*2, heads[1], expansion, bias, norm_type) for _ in range(num_blocks[1])])
        self.up1 = UpsampleSegNet(dim*2, dim)
        self.dec1= nn.Sequential(*[TransformerBlock(dim,   heads[0], expansion, bias, norm_type) for _ in range(num_blocks[0])])
        # Refinement & output
        self.refine = nn.Sequential(*[TransformerBlock(dim, heads[0], expansion, bias, norm_type) for _ in range(num_refine)])
        self.out    = nn.Conv2d(dim, out_channels, 3, padding=1, bias=bias)

    def forward(self, x):
        # Encoder
        x1 = self.patch_embed(x)
        x1 = self.enc1(x1)
        s1 = x1.size();  x2, idx1 = self.down1(x1)
        x2 = self.enc2(x2)
        s2 = x2.size();  x3, idx2 = self.down2(x2)
        x3 = self.enc3(x3)
        s3 = x3.size();  x4, idx3 = self.down3(x3)
        # Bottleneck
        z  = self.latent(x4)
        # Prompt & noise 3
        p3 = self.prompt3(z)
        z3= self.noise3(torch.cat([z, p3], dim=1))
        z3= self.reduce3(z3)
        # Decode stage 3
        d3 = self.up3(z3, idx3, output_size=s3)
        d3 = self.dec3(d3)
        # Prompt & noise 2
        p2 = self.prompt2(d3)
        z2= self.noise2(torch.cat([d3, p2], dim=1))
        z2= self.reduce2(z2)
        # Decode stage 2
        d2 = self.up2(z2, idx2, output_size=s2)
        d2 = self.dec2(d2)
        # Prompt & noise 1
        p1 = self.prompt1(d2)
        z1= self.noise1(torch.cat([d2, p1], dim=1))
        z1= self.reduce1(z1)
        # Decode stage 1
        d1 = self.up1(z1, idx1, output_size=s1)
        d1 = self.dec1(d1)
        # Refinement & output
        out= self.refine(d1)
        out= self.out(out) + x
        return out


if __name__ == "__main__":
    model = PromptIR_SegNet()
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    inp = torch.randn(1,3,256,256)
    print(model(inp).shape)
