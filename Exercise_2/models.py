#############################################################################
#   Tạo các lớp mô hình MyMSA, PyTorchMSA, PatchTokenizer và RowTokenizer   #
#############################################################################

import torch
import torch.nn as nn

# Tokenize theo các Patch (Lưới ô vuông) + CLS Token
class PatchTokenizer(nn.Module):

    def __init__(self, img_size = 28, patch_size = 7, in_channels = 1, embed_dim = 64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.n_patches, -1)
        x = self.proj(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1)
        x = x + self.pos_embed
        return x

# Tokenize theo các hàng của ảnh + CLS Token
class RowTokenizer(nn.Module):

    def __init__(self, img_size = 28, in_channels = 1, embed_dim = 64):
        super().__init__()
        self.n_patches = img_size
        self.proj = nn.Linear(img_size * in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, H, W * C)
        x = self.proj(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1)
        x = x + self.pos_embed
        return x
    
# MSA tự viết từ các phép toán cơ bản
class MyMSA(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim phải chia hết cho num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)                                 # embed_dim đầu ra bằng embed_dim đầu vào
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape                                                               # B: batch size, N: số lượng token, C: chiều embedding
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = self.k_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)          # (B, num_heads, N, N)
        attn_weights = torch.softmax(scores, dim = -1)                                  # (B, num_heads, N, N)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_linear(out)

# MSA dùng PyTorch nn.MultiheadAttention
class PyTorchMSA(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.msa = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first = True)

    def forward(self, x):
        out, _ = self.msa(x, x, x)
        return out

class ImageTransformerClassifier(nn.Module):

    def __init__(self, tokenizer, attention, embed_dim = 64, num_classes = 10):
        super().__init__()
        self.tokenizer = tokenizer
        self.attention = attention
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.tokenizer(x)
        if self.attention is not None:
            x = self.attention(x)
            x = x[:, 0]
        else:
            x = x[:, 1:].mean(dim = 1)
        return self.classifier(x)