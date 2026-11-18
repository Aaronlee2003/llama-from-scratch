#first vit  text model split world as a token  imaage as as seconary matrix 
#and it appied to transformer   make like a 1D sequence of tokens faltten
#petch imbedding for image 
"""
input((B,3,224,224)
patch size 16 by 16 pixels are  same like word token patcth (224/16=14)*(224/16=14)=196
trick we usesInstead of cropping the image and applying linear pooling (nn.Linear), I use a single line of nn.Conv2d(3, embed_dim, kernel_size=16, stride=16) with kernel size and stride set to $P(16)$ for patching and embedding dimensions
D
D) Finish the projection simultaneously.
(b,3,224,224)conv2d(b,d14,14)-> flatten -> (b,196,d)
cls token 
front of the sequence token add the empty cls token and pass the all layer  one cls token input the nlp head
which embedding s  vit just andd x+=positional_embedding
what is difference transformer  bridirection  lamma must use  causal mask for prediction but vit is bidirectional and does not use causal mask
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 196 (224/16=14)*(224/16=14)=196
        # kernel and stide  given patch size for non-overlapping patching and projection
        #
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        #kernle size and stride are set same using conv2d kernel did tnot overlap
        #using inner product for one patch  and patch partiain and linear projection for embedding
    def forward(self, x):
        x = self.proj(x)  # -> (B, 768, 14, 14)
        x = x.flatten(2)  # # -> (B, 768, 196)
        x = x.transpose(1, 2) # -> (B, 196, 768)
        return x
    
class attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads#all embedding divide  by head size
        
        self.scale = self.head_dim ** -0.5#prevent softamx slope 0
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)#merge 3 matrices for q, k, v in one linear layer
        #split 
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)#indicate q/k/v heads and split head dimenstion# for  matrix multiplication and head dimension alignment
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale#(b,12,197,64)@b,12,197,64)=(b,12,197,197)
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#(B, 12, 197, 64) reshape to (B, 197, 768) for projection usin reshape 12 head
        x = self.proj(x)
        return x
    
class vitblock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 학습 가능한 [CLS] 토큰과 1D 위치 임베딩
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))#empty bowl token  have all image not bais
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))#each position  addd embedding matrix declare learning parameter

        self.blocks = nn.ModuleList([vitblock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                           # (B, 3, 224, 224)->(B, 196, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)     # (B, 1, 768)Copy (expand) the [CLS] token by the current batch size B (1,1,768) -> (B, 1, 768)
        x = torch.cat((cls_tokens, x), dim=1)             # (B, 197, 768) at index 0 add cls token
        x = x + self.pos_embed                            # Broadcasting 더하기 each position  add each patch vector  element-wise

        for block in self.blocks:#12 transformer blocks in the model and exract  all petch features for each block shape is (B, 197, 768) maintianed
            x = block(x)

        x = self.norm(x)#doing normalization on the final output of the transformer blocks
        return self.head(x[:, 0])                         # [CLS] 토큰만 추출하여 분류: (B, 1000)

# 텐서 차원 검증
model = VisionTransformer()
dummy_img = torch.randn(2, 3, 224, 224)
print("Output shape:", model(dummy_img).shape)  # torch.Size([2, 1000])
