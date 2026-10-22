import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torch.utils.checkpoint import checkpoint

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int, num_heads: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.norm = nn.GroupNorm(32, in_channels)

        # 1x1 per Q, K, V
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        b, c, h, w = h_.shape

        # (B, num_heads, H*W, head_dim)
        q = self.q(h_).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = self.k(h_).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        v = self.v(h_).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

        # PyTorch FlashAttention / SDPA Kernel NATIVO
        out = F.scaled_dot_product_attention(q, k, v)

        # (B, C, H, W)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.proj_out(out)

        # Residual connection
        return x + out


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 16, hidden_dims: list | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 1024]

        # Stem: 512x512
        self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)

        # Down 1: 512x512 -> 256x256 (64 -> 128)
        self.down1 = nn.Sequential(
            ResBlock(hidden_dims[0]),
            ResBlock(hidden_dims[0]),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1)
        )

        # Down 2: 256x256 -> 128x128 (128 -> 256)
        self.down2 = nn.Sequential(
            ResBlock(hidden_dims[1]),
            ResBlock(hidden_dims[1]),
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=4, stride=2, padding=1)
        )

        # Down 3: 128x128 -> 64x64 (256 -> 512)
        self.down3 = nn.Sequential(
            ResBlock(hidden_dims[2]),
            ResBlock(hidden_dims[2]),
            nn.Conv2d(hidden_dims[2], hidden_dims[3], kernel_size=4, stride=2, padding=1)
        )
        self.down4 = nn.Sequential(
            ResBlock(hidden_dims[3]),
            ResBlock(hidden_dims[3]),
            nn.Conv2d(hidden_dims[3], hidden_dims[4], kernel_size=4, stride=2, padding=1)
        )
        # Bottleneck: 64x64 @ 512 channels con Self-Attention
        self.res_bottleneck1 = ResBlock(hidden_dims[4])
        self.attn_bottleneck = AttnBlock(hidden_dims[4], num_heads=8)
        self.res_bottleneck2 = ResBlock(hidden_dims[4])

        # mean and log-variance (2 * latent_dim)
        self.to_moments = nn.Conv2d(hidden_dims[4], 2 * latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.init_conv(x)
        
        # gradient checkpointing
        if self.training:
            x = checkpoint(self.down1, x, use_reentrant=False)
            x = checkpoint(self.down2, x, use_reentrant=False)
            x = checkpoint(self.down3, x, use_reentrant=False)
            x = checkpoint(self.down4, x, use_reentrant=False)
        else:
            x = self.down1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)

        # Bottleneck Pass
        x = self.res_bottleneck1(x)
        x = self.attn_bottleneck(x)
        x = self.res_bottleneck2(x)

        moments = self.to_moments(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, latent_dim: int = 16, hidden_dims: list | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128, 64]

        self.from_latent = nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, padding=1)

        # Bottleneck: 64x64 @ 512 channels con Self-Attention
        self.res_bottleneck1 = ResBlock(hidden_dims[0])
        self.attn_bottleneck = AttnBlock(hidden_dims[0], num_heads=8)
        self.res_bottleneck2 = ResBlock(hidden_dims[0])

        # Up 1: 32x32 -> 64x64 (512 -> 256)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            ResBlock(hidden_dims[1]),
            ResBlock(hidden_dims[1])
        )

        # Up 2: 64x64 -> 128x128 (256 -> 128)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),
            ResBlock(hidden_dims[2]),
            ResBlock(hidden_dims[2])
        )

        # Up 3: 128x128 -> 256x256 (128 -> 64)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dims[2], hidden_dims[3], kernel_size=3, padding=1),
            ResBlock(hidden_dims[3]),
            ResBlock(hidden_dims[3])
        )
        self.up4 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(hidden_dims[3], hidden_dims[4], kernel_size=3, padding=1),
                    ResBlock(hidden_dims[4]),
                    ResBlock(hidden_dims[4])
                )

        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, hidden_dims[4]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[4], out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.from_latent(z)

        # Bottleneck Pass
        z = self.res_bottleneck1(z)
        z = self.attn_bottleneck(z)
        z = self.res_bottleneck2(z)

        if self.training:
            z = checkpoint(self.up1, z, use_reentrant=False)
            z = checkpoint(self.up2, z, use_reentrant=False)
            z = checkpoint(self.up3, z, use_reentrant=False)
            z = checkpoint(self.up4, z, use_reentrant=False)
        else:
            z = self.up1(z)
            z = self.up2(z)
            z = self.up3(z)
            z = self.up4(z)

        return self.final_conv(z)


class AutoencoderKL(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, hidden_dims=[64, 128, 256, 512, 1024])
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dims=[1024, 512, 256, 128, 64])

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: torch.Tensor):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z)
        return recon, mean, logvar
    
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers: list[nn.Module] = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # Output patch grid
        )

    def forward(self, source_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        x = torch.cat([source_img, target_img], dim=1)
        return self.model(x)

class VAELoss(nn.Module):
    def __init__(self, kl_weight: float = 1e-5, lpips_weight: float = 0.6, adv_weight: float = 0.1):
        super().__init__()
        self.kl_weight = kl_weight
        self.lpips_weight = lpips_weight
        self.adv_weight = adv_weight

        # LPIPS (VGG16) in eval
        self.lpips_fn = lpips.LPIPS(net="vgg").eval()

        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def to(self, *args, **kwargs):
        self.lpips_fn = self.lpips_fn.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(
        self, 
        x: torch.Tensor,         
        recon_x: torch.Tensor,        
        mean: torch.Tensor | None = None, 
        logvar: torch.Tensor | None = None, 
        source_img: torch.Tensor | None = None, 
        discriminator: nn.Module | None = None, 
        for_discriminator: bool = False,
        is_real: bool = True
    ):
        # discriminator
        if for_discriminator:
            assert discriminator is not None and source_img is not None
            if is_real:
                # x is the real image
                preds = discriminator(source_img, x)
                target_label = torch.ones_like(preds)
            else:
                # recon_x is the fake image
                preds = discriminator(source_img, recon_x.detach())
                target_label = torch.zeros_like(preds)
                
            disc_loss = F.mse_loss(preds, target_label) # Least Squares GAN (LSGAN)
            return disc_loss

    
        # L1 Loss
        l1_loss = F.l1_loss(recon_x, x)

        # LPIPS Loss
        x_target = x.detach()
        lpips_loss = self.lpips_fn(recon_x, x_target).mean().to(torch.float32)

        # KL Loss
        assert mean is not None and logvar is not None
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3])
        kl_loss = kl_loss.mean().to(torch.float32)

        # Adversarial Loss
        adv_loss = torch.tensor(0.0, device=x.device)
        if discriminator is not None and source_img is not None and self.adv_weight > 0:
            preds_fake = discriminator(source_img, recon_x)
            # Il generatore vuole che il discriminatore classifichi le immagini finte come reali (1)
            adv_loss = F.mse_loss(preds_fake, torch.ones_like(preds_fake))

        # Total Loss
        total_loss = (
            l1_loss 
            + (self.lpips_weight * lpips_loss) 
            + (self.kl_weight * kl_loss) 
            + (self.adv_weight * adv_loss)
        )

        return total_loss, {
            "total": total_loss.item(),
            "l1": l1_loss.item(),
            "lpips": lpips_loss.item(),
            "kl": kl_loss.item(),
            "adv": adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
        }