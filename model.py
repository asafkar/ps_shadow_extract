import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim=None, dropout = 0.):
        super().__init__()
        outdim = output_dim if output_dim is not None else dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, outdim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
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
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# Mix patch information
class ConvMixTransformer(nn.Module):
    def __init__(self, args, dim, depth, heads, dim_head, mlp_dim, patch_size, dropout = 0., output_dim=None):
        outdim = output_dim if output_dim is not None else dim

        super().__init__()
        self.layers = nn.ModuleList([])
        self.patch_size = patch_size
        self.args = args
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
            ]))
        self.last_layer = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.last_ff = PreNorm(dim, FeedForward(dim, mlp_dim,output_dim=outdim, dropout = dropout))

    def forward(self, x, num_images_in_batch, img_size):
        batch, _, _ = x.shape

        for attn, ff, conv in self.layers:
            x_orig = x
            x = attn(x) + x
            x = ff(x) + x

            x = rearrange(x, '(b s) n f -> b s n f', b=num_images_in_batch)
            x = rearrange(x, 'b (s1 s2) n f -> b n f s1 s2', s1=img_size // self.patch_size)
            x = rearrange(x, 'b n f s1 s2 -> (b n) f s1 s2')
            x = conv(x)
            x = rearrange(x, '(b n) f s1 s2 -> (b s1 s2) n f', b=num_images_in_batch)

            if self.args.rand_layer_skip and self.training:
                if torch.rand(1) < 0.05:
                    x *= 0.01
                    x += x_orig

        x = self.last_layer(x)
        x = self.last_ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth,
                heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64,
                dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)


class ViT_normals(nn.Module):
    def __init__(self, *, args, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'mean',
                channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_size = patch_size
        self.dim = dim

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b s (w w1) (h h1) -> (b w h) s w1 h1', w1=patch_size, h1=patch_size),
            Rearrange('b (s c) h w -> b s (c h w)', c=channels),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = ConvMixTransformer(args, dim, depth, heads, dim_head, mlp_dim, patch_size, dropout)
        self.pool = pool

        self.mlp_head_normals = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 3 * args.patch_size ** 2),
            Rearrange('b (c w h) -> b c w h', w=patch_width, h=patch_width, c=3),
            nn.Conv2d(3, 3, stride=1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, stride=1, kernel_size=3, padding=1)
        )

    def forward(self, img):
        num_img_in_batch, _, _, img_size = img.shape
        x = self.to_patch_embedding(img)  # (b, s*c, w, h) -> (b, s*c, w*h)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, num_img_in_batch, img_size)  # input x: 256, 33, 768, output: 256, 33, 768

        merged_patches = rearrange(x[:, 1:, :], '(b p) n f -> b n p f', b=num_img_in_batch)  # batch, num_imgs, num_patch_in_img, features

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # in: 256, 3, 16, 16, out: 4, 3, 128, 128
        normals = rearrange(self.mlp_head_normals(x), '(b p1 p2) c w1 h1 -> b c (p1 w1) (p2 h1)', c=3,
                p1=img_size // self.patch_size, p2=img_size // self.patch_size)

        normals_z = 1 - (normals[..., 0, :, :].unsqueeze(1) ** 2 + normals[..., 1, :, :].unsqueeze(1) ** 2 + 1e-6).sqrt()
        normals_result = torch.hstack([normals[..., 0, :, :].unsqueeze(1), normals[..., 1, :, :].unsqueeze(1), normals_z])

        return normals_result, merged_patches


class LightEst(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        self.args = args
        self.dim = dim

        self.pre_light_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//16)  # xyz size = 3
        )
        self.light_transformer = Transformer(dim // 16, depth=2, mlp_dim=128, heads=4, dim_head=32, dropout=0.1)

        self.light_head = nn.Sequential(
            nn.LayerNorm(dim // 16),
            nn.Linear(dim // 16, 2)  # theta, phi
        )

    def forward(self, merged_patches):
        num_img_in_batch, n, p, f = merged_patches.shape
        reduced_features = self.pre_light_head(merged_patches.reshape(-1, self.dim))
        reduced_features = rearrange(reduced_features, '(b n p) f-> (b n) p f', b=num_img_in_batch, n=n)
        light = self.light_transformer(reduced_features).mean(dim=1)
        light = rearrange(self.light_head(light), '(b n) c -> b n c', n=n)
        theta = light[..., 0]
        phi = light[..., 1]
        light_x = torch.sin(theta) * torch.cos(phi)
        light_y = torch.sin(theta) * torch.sin(phi)
        light_z = torch.sqrt(1 - light_x ** 2 - light_y ** 2 + 1e-6).clamp(0, 1)  # prevent underflow
        light_out = torch.cat([light_x.unsqueeze(2), light_y.unsqueeze(2), light_z.unsqueeze(2)], dim=2)

        return light_out


class ShadowDecoder(nn.Module):
    def __init__(self, dim, patch_size, args):
        super().__init__()
        self.args = args
        self.patch_size = patch_size
        self.dim = dim

        self.to_shadow_features = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        self.shadow_head = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.shadow_transformer = ConvMixTransformer(args, dim, depth=3, heads=6, dim_head=32, mlp_dim=128,
                                                            patch_size=patch_size, dropout=0.1, output_dim=patch_size**2)

    def forward(self, features):
        # features.shape = num_imgs_in_batch, num_imgs_seq, num_patches_in_img, features
        b, s, p, _ = features.shape
        img_size = int(np.sqrt(p) * self.args.patch_size)

        input_features = self.to_shadow_features(features.reshape(-1, self.dim))
        input_features = rearrange(input_features, '(b n p) f -> (b p) n f', b=b, p=p)

        # input to shadow transformer - (total num of patches, num_imgs, dim_size)
        shadow_features = self.shadow_transformer(input_features, num_images_in_batch=b, img_size=img_size)  # input x: 256, 32, 768, output: 256, 32, 768
        tr_features = rearrange(shadow_features, '(b p) n f -> b n p f', b=b, p=p)
        tr_features = rearrange(tr_features, 'b n p (p1 p2) -> b n p p1 p2', b=b, p=p, p1=self.args.patch_size)
        conv_img = rearrange(tr_features, 'b n (p_h p_w) p1 p2 -> (b n) (p_h p1) (p_w p2)', b=b, p1=self.args.patch_size, p_h=int(np.sqrt(p)))

        shadows = self.shadow_head(conv_img.unsqueeze(1)).squeeze().reshape(b, -1, img_size, img_size)
        # TODO add conv layer, reshape back to images
        
        # shadow_features = rearrange(shadow_features, '(b p) n f -> (b n p) f', b=b, p=p)
        # shadows = self.shadow_head(shadow_features)
        # shadows = rearrange(shadows, '(b n p) f -> b n p f', b=b, p=p)
        # shadows = rearrange(shadows, 'b n (p1 p2) (w1 h1) -> b n (p1 w1) (p2 h1)',
        #     p1=img_size // self.patch_size,
        #     p2=img_size // self.patch_size,
        #     w1=self.patch_size)
        #
        # shadows = torch.sigmoid(shadows)
        return shadows