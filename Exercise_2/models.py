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
        self.n_patches = (img_size // patch_size) ** 2                                  # Số lượng patch

        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)         # Lớp Linear để ánh xạ từ patch sang embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))                     # CLS token đặc trưng cho toàn bộ ảnh
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))    # Ma trận Positional Embedding để bổ sung thông tin vị trí không gian cho các token

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)           # Tách ảnh thành các patch có kích thước p x p
        x = x.contiguous().view(B, C, -1, p, p)         # Chuyển đổi tensor để có dạng (B, C, n_patches, p, p)
        x = x.permute(0, 2, 1, 3, 4).contiguous()       # Chuyển đổi tensor để có dạng (B, n_patches, C, p, p)
        x = x.view(B, self.n_patches, -1)               # Chuyển đổi tensor để có dạng (B, n_patches, C * p * p)
        x = self.proj(x)                                # Ánh xạ các patch sang embedding có kích thước embed_dim
        
        cls_tokens = self.cls_token.expand(B, -1, -1)   # Tạo CLS token cho mỗi ảnh trong batch
        x = torch.cat((cls_tokens, x), dim = 1)         # Nối CLS token vào đầu chuỗi các token của patch
        x = x + self.pos_embed                          # Bổ sung thông tin vị trí không gian cho các token bằng cách cộng ma trận Positional Embedding
        return x

# Tokenize theo các hàng của ảnh + CLS Token
class RowTokenizer(nn.Module):

    def __init__(self, img_size = 28, in_channels = 1, embed_dim = 64):
        super().__init__()
        self.n_patches = img_size                                                       # Số lượng hàng của ảnh (mỗi hàng là một token)

        self.proj = nn.Linear(img_size * in_channels, embed_dim)                        # Lớp Linear để ánh xạ từ mỗi hàng sang embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))                     # CLS token đặc trưng cho toàn bộ ảnh
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))    # Ma trận Positional Embedding để bổ sung thông tin vị trí không gian cho các token

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.permute(0, 2, 1, 3).contiguous().view(B, H, W * C)    # Chuyển đổi tensor để có dạng (B, H, W * C), mỗi hàng của ảnh trở thành một token
        x = self.proj(x)                                            # Ánh xạ các hàng sang embedding có kích thước embed_dim
        
        cls_tokens = self.cls_token.expand(B, -1, -1)               # Tạo CLS token cho mỗi ảnh trong batch
        x = torch.cat((cls_tokens, x), dim = 1)                     # Nối CLS token vào đầu chuỗi các token của hàng
        x = x + self.pos_embed                                      # Bổ sung thông tin vị trí không gian cho các token bằng cách cộng ma trận Positional Embedding
        return x
    
# MSA tự viết từ các phép toán cơ bản
class MyMSA(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim                          # Chiều embedding của mỗi token
        self.num_heads = num_heads                          # Số lượng đầu (head) trong cơ chế Multi-Head Self-Attention
        self.head_dim = embed_dim // num_heads              # Chiều embedding của mỗi head (head_dim)
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim phải chia hết cho num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)     # Lớp Linear để ánh xạ từ embedding sang query
        self.k_linear = nn.Linear(embed_dim, embed_dim)     # Lớp Linear để ánh xạ từ embedding sang key
        self.v_linear = nn.Linear(embed_dim, embed_dim)     # Lớp Linear để ánh xạ từ embedding sang value
        self.out_linear = nn.Linear(embed_dim, embed_dim)   # Lớp Linear để ánh xạ từ embedding sau khi tính attention trở lại embedding ban đầu

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # Chuyển đổi tensor để có dạng (B, num_heads, N, head_dim) sang ma trận query
        k = self.k_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # Chuyển đổi tensor để có dạng (B, num_heads, N, head_dim) sang ma trận key
        v = self.v_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # Chuyển đổi tensor để có dạng (B, num_heads, N, head_dim) sang ma trận value
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)          # Tính toán ma trận attention scores
        attn_weights = torch.softmax(scores, dim = -1)                                  # Áp dụng softmax để chuẩn hóa ma trận attention scores thành ma trận attention weights
        
        out = torch.matmul(attn_weights, v)                                             # Tính toán ma trận output bằng cách nhân ma trận attention weights với ma trận value
        out = out.transpose(1, 2).contiguous().view(B, N, C)                            # Chuyển đổi tensor để có dạng (B, N, C) bằng cách gộp các head lại với nhau
        return self.out_linear(out)

# MSA dùng PyTorch nn.MultiheadAttention
class PyTorchMSA(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.msa = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first = True)  # Sử dụng lớp MultiheadAttention của PyTorch để thực hiện cơ chế Multi-Head Self-Attention

    def forward(self, x):
        out, _ = self.msa(x, x, x)                                                                          # Thực hiện cơ chế Multi-Head Self-Attention bằng cách sử dụng ma trận query, key và value đều là x
        return out

class ImageTransformerClassifier(nn.Module):

    def __init__(self, tokenizer, attention, embed_dim = 64, num_classes = 10):
        super().__init__()
        self.tokenizer = tokenizer                          # Lớp tokenizer để chuyển đổi ảnh sang chuỗi các token
        self.attention = attention                          # Lớp attention để thực hiện cơ chế Multi-Head Self-Attention
        self.classifier = nn.Linear(embed_dim, num_classes) # Lớp Linear để ánh xạ từ embedding của CLS token sang số lượng lớp (classes) cần phân loại

    def forward(self, x):
        x = self.tokenizer(x)                               # Chuyển đổi ảnh sang chuỗi các token bằng cách sử dụng lớp tokenizer

        if self.attention is not None:
            x = self.attention(x)                           # Thực hiện cơ chế Multi-Head Self-Attention bằng cách sử dụng lớp attention
            x = x[:, 0]                                     # Lấy embedding của CLS token (token đầu tiên) để sử dụng cho việc phân loại
        else:
            x = x[:, 1:].mean(dim = 1)                      # Lấy trung bình embedding của tất cả các token (ngoại trừ CLS token) để sử dụng cho việc phân loại

        return self.classifier(x)