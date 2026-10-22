import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from torch.utils.checkpoint import checkpoint

# TIMESTEP SHIFTING UTILITY
def time_shift(t: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """
    Applica il timestep shifting (time warping) per rimappare la traiettoria
    del Flow Matching, concentrando la risoluzione del flusso sui dettagli fini.
    """
    return (shift * t) / (1 + (shift - 1) * t)



# TIMESTEP & CLASS EMBEDDING
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * 1000.0, self.frequency_embedding_size)
        return self.mlp(t_freq)


class RoPE2D(nn.Module):
    def __init__(self, head_dim, rope_base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_base = rope_base

        self.register_buffer("rope_row_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_row_sin", torch.empty(0), persistent=False)
        self.register_buffer("rope_col_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_col_sin", torch.empty(0), persistent=False)
        self.register_buffer("cached_height", torch.tensor(-1), persistent=False)
        self.register_buffer("cached_width", torch.tensor(-1), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _get_rope_cache(self, seq_len: int, device: torch.device, height: int, width: int):
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")

        expected_len = height * width
        if seq_len != expected_len:
            raise ValueError(f"seq_len {seq_len} does not match height ({height}) * width ({width}) = {expected_len}")

        if (
            self.rope_row_cos.numel() > 0
            and self.rope_row_cos.device == device
            and self.cached_height == height
            and self.cached_width == width
        ):
            return self.rope_row_cos, self.rope_row_sin, self.rope_col_cos, self.rope_col_sin

        row_ids = torch.arange(height, device=device, dtype=torch.float32)
        col_ids = torch.arange(width, device=device, dtype=torch.float32)
        row_idx, col_idx = torch.meshgrid(row_ids, col_ids, indexing="ij")
        row_idx = row_idx.flatten()
        col_idx = col_idx.flatten()

        if self.head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {self.head_dim}.")

        axis_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_base
            ** (torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32) / axis_dim)
        )

        row_angles = row_idx[:, None] * inv_freq[None, :]
        col_angles = col_idx[:, None] * inv_freq[None, :]

        row_cos = torch.repeat_interleave(torch.cos(row_angles), repeats=2, dim=-1)
        row_sin = torch.repeat_interleave(torch.sin(row_angles), repeats=2, dim=-1)
        col_cos = torch.repeat_interleave(torch.cos(col_angles), repeats=2, dim=-1)
        col_sin = torch.repeat_interleave(torch.sin(col_angles), repeats=2, dim=-1)

        self.rope_row_cos = row_cos
        self.rope_row_sin = row_sin
        self.rope_col_cos = col_cos
        self.rope_col_sin = col_sin
        self.cached_height = torch.tensor(height, device=device)
        self.cached_width = torch.tensor(width, device=device)

        return self.rope_row_cos, self.rope_row_sin, self.rope_col_cos, self.rope_col_sin

    def forward(self, x, height=None, width=None, visual_len=None):
        if x.dim() != 4:
            raise ValueError(f"Expected tensor with shape [B, H, N, D], got {tuple(x.shape)}")

        _, _, seq_len, head_dim = x.shape
        if seq_len == 0:
            return x

        if height is None or width is None:
            grid_size = int(math.sqrt(visual_len if visual_len else seq_len))
            height = width = grid_size

        single_img_len = height * width
        if visual_len is None:
            visual_len = seq_len

        if seq_len > visual_len:
            x_vis = x[..., :visual_len, :]
            x_txt = x[..., visual_len:, :]
            x_vis_rot = self.forward(x_vis, height=height, width=width, visual_len=visual_len)
            return torch.cat([x_vis_rot, x_txt], dim=-2)

        num_images = max(1, visual_len // single_img_len)
        
        row_cos_single, row_sin_single, col_cos_single, col_sin_single = self._get_rope_cache(
            single_img_len, x.device, height=height, width=width
        )

        row_cos = row_cos_single.repeat(num_images, 1)
        row_sin = row_sin_single.repeat(num_images, 1)
        col_cos = col_cos_single.repeat(num_images, 1)
        col_sin = col_sin_single.repeat(num_images, 1)

        if row_cos.shape[0] != seq_len:
            row_cos = row_cos[:seq_len]
            row_sin = row_sin[:seq_len]
            col_cos = col_cos[:seq_len]
            col_sin = col_sin[:seq_len]

        axis_dim = head_dim // 2
        row_cos = row_cos.view(1, 1, seq_len, axis_dim).to(dtype=x.dtype, device=x.device)
        row_sin = row_sin.view(1, 1, seq_len, axis_dim).to(dtype=x.dtype, device=x.device)
        col_cos = col_cos.view(1, 1, seq_len, axis_dim).to(dtype=x.dtype, device=x.device)
        col_sin = col_sin.view(1, 1, seq_len, axis_dim).to(dtype=x.dtype, device=x.device)

        x_row = x[..., :axis_dim]
        x_col = x[..., axis_dim:]

        x_row_rot = x_row * row_cos + self._rotate_half(x_row) * row_sin
        x_col_rot = x_col * col_cos + self._rotate_half(x_col) * col_sin

        return torch.cat([x_row_rot, x_col_rot], dim=-1)



class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, rope_base=10000.0):
        super().__init__()
        #setting all bias to false to match flux2 architecture
    
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.img_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # SwiGLU gate + up (2 * mlp_hidden_dim)
        self.img_linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim * 2, bias=False)
        self.img_linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, bias=False)

        self.txt_linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim * 2, bias=False)
        self.txt_linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, bias=False)

        self.img_q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.img_k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.txt_q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.txt_k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6)

        self.rope = RoPE2D(self.head_dim, rope_base=rope_base)
        
        self.img_adaLN_linear = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.txt_adaLN_linear = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        nn.init.zeros_(self.img_adaLN_linear.weight)
        nn.init.zeros_(self.txt_adaLN_linear.weight)

        self.mlp_hidden_dim = mlp_hidden_dim

    def forward(self, img, txt, c, height=None, width=None):
        B, N_img, _ = img.shape
        _, N_txt, _ = txt.shape

        c_silu = F.silu(c)
        img_shift, img_scale, img_gate = self.img_adaLN_linear(c_silu).chunk(3, dim=-1)
        txt_shift, txt_scale, txt_gate = self.txt_adaLN_linear(c_silu).chunk(3, dim=-1)

        # image flux
        img_normed = self.img_norm(img) * (1 + img_scale.unsqueeze(1)) + img_shift.unsqueeze(1)
        img_fused = self.img_linear1(img_normed)
        img_qkv, img_mlp_gate, img_mlp_up = img_fused.split(
            [self.hidden_size * 3, self.mlp_hidden_dim, self.mlp_hidden_dim], dim=-1
        )
        img_mlp_in = F.silu(img_mlp_gate) * img_mlp_up  # SwiGLU

        img_q, img_k, img_v = (
            img_qkv.reshape(B, N_img, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        img_q = self.img_q_norm(img_q)
        img_k = self.img_k_norm(img_k)
        img_q = self.rope(img_q, height=height, width=width, visual_len=N_img)
        img_k = self.rope(img_k, height=height, width=width, visual_len=N_img)

        # text flux
        txt_normed = self.txt_norm(txt) * (1 + txt_scale.unsqueeze(1)) + txt_shift.unsqueeze(1)
        txt_fused = self.txt_linear1(txt_normed)
        txt_qkv, txt_mlp_gate, txt_mlp_up = txt_fused.split(
            [self.hidden_size * 3, self.mlp_hidden_dim, self.mlp_hidden_dim], dim=-1
        )
        txt_mlp_in = F.silu(txt_mlp_gate) * txt_mlp_up  # SwiGLU

        txt_q, txt_k, txt_v = (
            txt_qkv.reshape(B, N_txt, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        #normalizing q and k before the joint attention 
        txt_q = self.txt_q_norm(txt_q)
        txt_k = self.txt_k_norm(txt_k)

        # joint attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N_txt + N_img, self.hidden_size)

        txt_attn = attn_out[:, :N_txt, :]
        img_attn = attn_out[:, N_txt:, :]

        img_combined = torch.cat([img_attn, img_mlp_in], dim=-1)
        img_out = self.img_linear2(img_combined)
        img = img + img_gate.unsqueeze(1) * img_out

        txt_combined = torch.cat([txt_attn, txt_mlp_in], dim=-1)
        txt_out = self.txt_linear2(txt_combined)
        txt = txt + txt_gate.unsqueeze(1) * txt_out

        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, rope_base=10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim * 2, bias=False)
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, bias=False)
        
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        
        self.rope = RoPE2D(self.head_dim, rope_base=rope_base)
        
        self.adaLN_linear = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        nn.init.zeros_(self.adaLN_linear.weight)

        self.mlp_hidden_dim = mlp_hidden_dim

    def forward(self, x, c, height=None, width=None, visual_len=None):
        B, N_seq, _ = x.shape
        shift, scale, gate = self.adaLN_linear(F.silu(c)).chunk(3, dim=-1)
        
        x_norm = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        fused = self.linear1(x_norm)
        qkv, mlp_gate, mlp_up = fused.split(
            [self.hidden_size * 3, self.mlp_hidden_dim, self.mlp_hidden_dim], dim=-1
        )
        mlp_out = F.silu(mlp_gate) * mlp_up  # SwiGLU
        
        q, k, v = (
            qkv.reshape(B, N_seq, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        
        q = self.q_norm(q)
        q = self.rope(q, height=height, width=width, visual_len=visual_len)
        k = self.k_norm(k)
        k = self.rope(k, height=height, width=width, visual_len=visual_len)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N_seq, self.hidden_size)
        
        combined = torch.cat([attn_out, mlp_out], dim=-1)
        out = self.linear2(combined)
        
        x = x + gate.unsqueeze(1) * out
        return x


# -----------------------------------------------------------------------------
# 3. DIT COMPLETO (BIAS=FALSE)
# -----------------------------------------------------------------------------
class MMDiT(nn.Module):
    def __init__(self, patch_size=2, in_channels=32, hidden_size=512, y_dim_clip=512, y_dim_t5=768, depth_double=2, depth_single=4, num_heads=8, max_cond_images=5, use_gradient_checkpointing=True, shift_factor=1.5):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.shift_factor = shift_factor
        
        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size) 
        
        self.y_clip_embedder = nn.Linear(y_dim_clip, hidden_size, bias=False)
        self.y_t5_embedder = nn.Linear(y_dim_t5, hidden_size, bias=False)

        self.img_tok_embed = nn.Embedding(max_cond_images + 1, hidden_size)

        self.linear_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
        )
        
        self.blocks_double = nn.ModuleList([DoubleStreamBlock(hidden_size, num_heads) for _ in range(depth_double)])
        self.blocks_single = nn.ModuleList([SingleStreamBlock(hidden_size, num_heads) for _ in range(depth_single)])
        
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN_silu = nn.SiLU()
        self.final_adaLN_linear = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=False)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.final_adaLN_linear.weight)
        nn.init.zeros_(self.final_linear.weight)

    def patchify(self, x):
        B, _, H, W = x.shape
        p = self.patch_size
        C = self.in_channels
        h, w = H // p, W // p
        
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x, patch_H, patch_W):
        B = x.shape[0]
        p = self.patch_size
        C = self.in_channels
        x = x.reshape(B, patch_H, patch_W, p, p, C)
        x = torch.einsum("nhwpqc->nchpwq", x).contiguous()
        return x.reshape(B, C, patch_H * p, patch_W * p)

    def forward(self, x_noisy, t, y_clip, y_t5, clip_lens, t5_mask=None, cond_images: Optional[List[torch.Tensor]] = None):
        if cond_images is None:
            cond_images = []

        # time shifting
        if self.training:
            t = time_shift(t, shift=self.shift_factor)

        B, C, H, W = x_noisy.shape
        p = self.patch_size
        patch_H, patch_W = H // p, W // p
        device = x_noisy.device
        

        idx_tensor = torch.zeros((B,), dtype=torch.long, device=device)
        t_0 = self.img_tok_embed(idx_tensor).unsqueeze(1)
        
        x_noisy_patches = self.x_embedder(self.patchify(x_noisy))
        visual_tokens = [x_noisy_patches + t_0]
        
        for idx, img_cond in enumerate(cond_images, start=1):
            cond_idx_tensor = torch.full((B,), idx, dtype=torch.long, device=device)
            t_id = self.img_tok_embed(cond_idx_tensor).unsqueeze(1)
            cond_patches = self.x_embedder(self.patchify(img_cond))
            visual_tokens.append(cond_patches + t_id)
            
        img = torch.cat(visual_tokens, dim=1) 
        total_visual_len = img.shape[1] 
        
        # CLIP and EOS embedding
        y_c_projected = self.y_clip_embedder(y_clip)
        batch_indices = torch.arange(B, device=device)
        y_c_eos = y_c_projected[batch_indices, clip_lens]

        # T5 Masked Mean Pooling
        y_t5_projected = self.y_t5_embedder(y_t5)
        if t5_mask is not None:
            mask = t5_mask.unsqueeze(-1).type_as(y_t5_projected)
            sum_hidden = torch.sum(y_t5_projected * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            y_t5_pooled = sum_hidden / sum_mask
        else:
            y_t5_pooled = y_t5_projected.mean(dim=1)

        # global modulation
        t_emb = self.t_embedder(t)
        c = self.linear_proj(t_emb + y_c_eos + y_t5_pooled)
        y = y_t5_projected

        # Double Stream Blocks 
        for block in self.blocks_double:
            if self.use_gradient_checkpointing and self.training:
                img, y = checkpoint(block, img, y, c, height=patch_H, width=patch_W, use_reentrant=False)
            else:
                img, y = block(img, y, c, height=patch_H, width=patch_W)

        h = torch.cat([img, y], dim=1) 
        
        # Single Stream Blocks
        for block in self.blocks_single:
            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(block, h, c, height=patch_H, width=patch_W, visual_len=total_visual_len, use_reentrant=False)
            else:
                h = block(h, c, height=patch_H, width=patch_W, visual_len=total_visual_len)

        num_noisy_tokens = x_noisy_patches.shape[1]
        x_out = h[:, :num_noisy_tokens, :]

        # unpatchify
        shift, scale = self.final_adaLN_linear(self.final_adaLN_silu(c)).chunk(2, dim=-1)
        x_out = self.final_norm(x_out) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x_out = self.final_linear(x_out)
        
        return self.unpatchify(x_out, patch_H, patch_W)