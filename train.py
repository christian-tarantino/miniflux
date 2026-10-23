import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import h5py
import numpy as np
from tqdm import tqdm
from typing import Any, Optional, Tuple, List, Dict, cast
from safetensors.torch import load_file
from torchinfo import summary
from model.model import MMDiT
from safetensors.torch import load_file
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if BF16_OK else torch.float16
VAE_WEIGHTS_PATH = "LDM/flux2/flux2-vae.safetensors"

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cuda"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trovato in: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    

    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    start_epoch = checkpoint.get("epoch", 0)
    
    print(f"✅ Checkpoint caricato correttamente da {checkpoint_path} (Ripresa dall'Epoch: {start_epoch})")
    return start_epoch

def decode_if_needed(vae: Optional[nn.Module], x: torch.Tensor) -> torch.Tensor:
    #Esegue la decodifica tramite il VAE solo se il modello VAE è disponibile
    if vae is None:
        return x
    if hasattr(vae, "decode"):
        out = vae(x)
        return out.sample if hasattr(out, "sample") else out
    return vae(x)

def patchify_latents(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, h // 2, w // 2)

def unpatchify_latents(x: torch.Tensor) -> torch.Tensor:
    b, c4, h, w = x.shape
    x = x.reshape(b, c4 // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c4 // 4, h * 2, w * 2)

class LatentDenormalizer:
    def __init__(self, vae: AutoencoderKLFlux2):
        bn = getattr(vae, "bn", None)
        if bn is None or not hasattr(bn, "running_mean") or bn.running_mean is None:
            raise ValueError("Il modulo BatchNorm 'bn' non è valido o non è inizializzato nel VAE.")

        mean = bn.running_mean.detach().float().cpu()
        var = bn.running_var.detach().float().cpu()
        eps = float(getattr(vae.config, "batch_norm_eps", getattr(bn, "eps", 1e-4)))
        
        self.bn_std = torch.sqrt(var + eps).view(1, -1, 1, 1).to(DEVICE)
        self.bn_mean = mean.view(1, -1, 1, 1).to(DEVICE)
        self.shift = float(getattr(vae.config, "shift_factor", 0.0))
        self.scale = float(getattr(vae.config, "scaling_factor", 1.0))

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        x = patchify_latents(z.float())
        x = x / self.scale + self.shift
        x = x * self.bn_std + self.bn_mean
        return unpatchify_latents(x)

def load_flux2_decoder_for_eval(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pesi VAE non trovati: {weights_path}")

    print(f"Caricamento VAE Flux2 da {weights_path}...")
    vae = AutoencoderKLFlux2()
    state_dict = load_file(weights_path, device="cpu")
    
    converted_state_dict = convert_ldm_vae_checkpoint(state_dict, dict(vae.config))
    flux2_key_map = {
        "bn.running_mean": "bn.running_mean",
        "bn.running_var": "bn.running_var",
        "bn.num_batches_tracked": "bn.num_batches_tracked",
        "quant_conv.weight": "encoder.quant_conv.weight",
        "quant_conv.bias": "encoder.quant_conv.bias",
        "post_quant_conv.weight": "decoder.post_quant_conv.weight",
        "post_quant_conv.bias": "decoder.post_quant_conv.bias",
    }
    converted_state_dict.update({key: state_dict[source] for key, source in flux2_key_map.items()})
    vae.load_state_dict(converted_state_dict, strict=True)
    
    denormalizer = LatentDenormalizer(vae)

    vae.to(device=DEVICE) # type: ignore

    #forcing vae to stay in fp32
    vae.to(dtype=torch.float32) # type: ignore
    
    vae.eval()
    vae.requires_grad_(False)
    print("VAE (Decoder + BN) caricato in FP32 e pronto per la ricostruzione.")
    return vae, denormalizer

class MultiBucketH5Dataset(Dataset):
    def __init__(self, h5_paths: List[str], scale_factor: float = 1.0):
        super().__init__()
        self.h5_paths = h5_paths
        self.scale_factor = scale_factor
        
        self.file_handles = None
        self.cumulative_sizes = []
        
        total_samples = 0
        for path in self.h5_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File HDF5 non trovato in '{path}'.")
            with h5py.File(path, 'r', libver='latest') as f:
                latents_ds = cast(h5py.Dataset, f['latents'])
                if latents_ds.ndim != 4 or latents_ds.shape[1] != 32:
                    raise ValueError(
                        f"Latent dataset in '{path}' has shape {latents_ds.shape}; "
                        "expected [N, 32, H, W]."
                    )
                if 'clip_lens' not in f:
                    raise KeyError(f"Dataset 'clip_lens' mancante nel file HDF5 '{path}'.")
                
                size = latents_ds.shape[0]
                total_samples += size
                self.cumulative_sizes.append(total_samples)
                
        self.total_samples = total_samples
        if self.total_samples == 0:
            raise ValueError("Nessun campione trovato nei file HDF5 forniti.")

    def _init_workers(self):
        #Apre i file HDF5 in modo sicuro all'interno di ciascun worker del DataLoader
        if self.file_handles is None:
            self.file_handles = [
                h5py.File(path, 'r', libver='latest') for path in self.h5_paths
            ]

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._init_workers()
        assert self.file_handles is not None

        # Trova a quale file appartiene l'indice globale `idx`
        file_idx = 0
        while idx >= self.cumulative_sizes[file_idx]:
            file_idx += 1
            
        local_idx = idx if file_idx == 0 else idx - self.cumulative_sizes[file_idx - 1]
        f = self.file_handles[file_idx]

        latents_ds = cast(h5py.Dataset, f['latents'])
        clip_ds = cast(h5py.Dataset, f['clip_hidden_states'])
        t5_ds = cast(h5py.Dataset, f['t5_hidden_states'])
        h_ds = cast(h5py.Dataset, f['latent_h'])
        w_ds = cast(h5py.Dataset, f['latent_w'])
        clip_lens_ds = cast(h5py.Dataset, f['clip_lens'])

        latents = torch.from_numpy(latents_ds[local_idx]).float()
        clip_emb = torch.from_numpy(clip_ds[local_idx]).float()
        t5_emb = torch.from_numpy(t5_ds[local_idx]).float()
        
        latent_h = int(h_ds[local_idx])
        latent_w = int(w_ds[local_idx])
        clip_lens = int(clip_lens_ds[local_idx])

        if self.scale_factor != 1.0:
            latents = latents * self.scale_factor

        return {
            "latents": latents,
            "clip_hidden": clip_emb,
            "t5_hidden": t5_emb,
            "latent_h": torch.tensor(latent_h, dtype=torch.long),
            "latent_w": torch.tensor(latent_w, dtype=torch.long),
            "clip_lens": torch.tensor(clip_lens, dtype=torch.long)
        }


def train(
    model, 
    train_loaders: Dict[str, DataLoader], 
    optimizer, 
    device, 
    scaler=None,
    scheduler=None,
    max_grad_norm=1.0,
    p_drop_text: float = 0.10, 
    debug_first_batch=True,
    logit_mean=0.0,
    logit_std=1.0
):
    model.train()
    running_loss = 0.0
    total_samples = 0
    epoch_start = time.time()

    null_t5: Optional[torch.Tensor] = None
    null_clip: Optional[torch.Tensor] = None
    null_clip_lens: Optional[torch.Tensor] = None

    bucket_names = list(train_loaders.keys())
    iterators = {name: iter(loader) for name, loader in train_loaders.items()}

    #for stronger multi-batch training
    total_batches = sum(len(loader) for loader in train_loaders.values())
    
    progress_bar = tqdm(range(total_batches), desc="Training MMDiT (Multi-Bucket)", leave=False)

    for batch_count in progress_bar:
        bucket_name = random.choice(bucket_names)
        
        try:
            batch_data = next(iterators[bucket_name])
        except StopIteration:
            iterators[bucket_name] = iter(train_loaders[bucket_name])
            batch_data = next(iterators[bucket_name])

        if debug_first_batch and batch_count == 0:
            batch_start_time = time.time()

        # data extraction
        x_0 = batch_data["latents"].to(device)
        y_t5 = batch_data["t5_hidden"].to(device)
        y_clip = batch_data["clip_hidden"].to(device)
        clip_lens = batch_data["clip_lens"].to(device)
        batch_size = x_0.size(0)

        if null_t5 is None:
            null_t5 = torch.zeros_like(y_t5[:1])
        if null_clip is None:
            null_clip = torch.zeros_like(y_clip[:1])
        if null_clip_lens is None:
            null_clip_lens = torch.zeros_like(clip_lens[:1])

        # dropout 
        if p_drop_text > 0.0:
            # 1d mask 
            drop_mask = torch.rand(batch_size, device=device) < p_drop_text
            
            # masking for T5 [B, SeqLen, Dim] e CLIP [B, SeqLen, Dim]
            t5_mask_expanded = drop_mask.view(batch_size, *([1] * (y_t5.ndim - 1)))
            clip_mask_expanded = drop_mask.view(batch_size, *([1] * (y_clip.ndim - 1)))
            
            y_t5 = torch.where(t5_mask_expanded, null_t5.expand_as(y_t5), y_t5)
            y_clip = torch.where(clip_mask_expanded, null_clip.expand_as(y_clip), y_clip)
            
            # clip lens use directly 1d
            clip_lens = torch.where(drop_mask, null_clip_lens.expand_as(clip_lens), clip_lens)

        optimizer.zero_grad()

        # Logit-Normal Timestep Sampling
        u = torch.randn(batch_size, device=device) * logit_std + logit_mean
        t = torch.sigmoid(u)

        # Rectified Flow Interpolation standard
        x_1 = torch.randn_like(x_0)
        t_dim = t.view(batch_size, *([1] * (x_0.ndim - 1)))
        
        x_t = (1.0 - t_dim) * x_0 + t_dim * x_1
        target_velocity = x_1 - x_0

        use_cuda = device.type == "cuda"
        dtype_amp = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported() and scaler is None) else torch.float16

        with torch.amp.autocast(device_type=device.type, dtype=dtype_amp, enabled=use_cuda):
            pred_velocity = model(x_t, t, y_clip=y_clip, y_t5=y_t5, clip_lens=clip_lens, cond_images=None)
            loss = F.mse_loss(pred_velocity, target_velocity)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scale_before <= scaler.get_scale()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer_stepped = True

        if scheduler is not None and optimizer_stepped:
            scheduler.step()

        if debug_first_batch and batch_count == 0:
            step_time = time.time() - batch_start_time # type: ignore
            print(f"\n[Debug First Batch] Bucket: {bucket_name} | Tempo batch 0: {step_time:.4f}s | Loss: {loss.item():.4f}")

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_time = time.time() - epoch_start
    samples_per_sec = total_samples / epoch_time if epoch_time > 0 else 0.0
    epoch_loss = running_loss / total_samples

    return epoch_loss, epoch_time, samples_per_sec

@torch.no_grad()
def validate(
    model: nn.Module, 
    val_loaders: Dict[str, DataLoader], 
    device: torch.device, 
    epoch: int,
    cfg_scale: float = 4.0, 
    num_sampling_steps: int = 20,
    num_samples_to_save: int = 3, 
    save_dir: str = "samples",
    vae: Optional[nn.Module] = None,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    scale_factor: float = 1.0,
    denormalizer: Optional[LatentDenormalizer] = None
) -> Tuple[float, float, float]:
    model.eval()
    
    total_running_loss = 0.0
    total_samples_count = 0
    
    use_cuda = device.type == "cuda"
    dtype_amp = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    if use_cuda:
        torch.cuda.synchronize()
    val_start_time = time.time()

    os.makedirs(save_dir, exist_ok=True)
    bucket_samples: Dict[str, dict] = {}

    for bucket_name, loader in val_loaders.items():
        bucket_loss = 0.0
        bucket_samples_count = 0
        
        vis_x0_list = []
        vis_t5_list = []
        vis_clip_list = []
        vis_cliplens_list = []  
        collected_vis_samples = 0
        
        for batch_data in tqdm(loader, desc=f"Validating [{bucket_name}]", leave=False):
            x_0 = batch_data["latents"].to(device)
            y_t5 = batch_data["t5_hidden"].to(device)
            y_clip = batch_data["clip_hidden"].to(device)
            clip_lens = batch_data["clip_lens"].to(device)
            batch_size = x_0.size(0)

            # Logit-Normal Timestep Sampling 
            u = torch.randn(batch_size, device=device) * logit_std + logit_mean
            t = torch.sigmoid(u)
            t_dim = t.view(batch_size, *([1] * (x_0.ndim - 1)))

            x_1 = torch.randn_like(x_0)
            x_t = (1.0 - t_dim) * x_0 + t_dim * x_1
            target_velocity = x_1 - x_0

            with torch.amp.autocast(device_type=device.type, dtype=dtype_amp, enabled=use_cuda):
                pred_velocity = model(x_t, t, y_clip=y_clip, y_t5=y_t5, clip_lens=clip_lens, cond_images=None)
                loss = F.mse_loss(pred_velocity, target_velocity)

            bucket_loss += loss.item() * batch_size
            bucket_samples_count += batch_size

            if collected_vis_samples < num_samples_to_save:
                needed = num_samples_to_save - collected_vis_samples
                take_n = min(batch_size, needed)
                vis_x0_list.append(x_0[:take_n].detach().cpu())
                vis_t5_list.append(y_t5[:take_n].detach().cpu())
                vis_clip_list.append(y_clip[:take_n].detach().cpu())
                vis_cliplens_list.append(clip_lens[:take_n].detach().cpu())
                collected_vis_samples += take_n

        total_running_loss += bucket_loss
        total_samples_count += bucket_samples_count

        if vis_x0_list:
            bucket_samples[bucket_name] = {
                "x0": torch.cat(vis_x0_list, dim=0),
                "t5": torch.cat(vis_t5_list, dim=0),
                "clip": torch.cat(vis_clip_list, dim=0),
                "clip_lens": torch.cat(vis_cliplens_list, dim=0)
            }

    if use_cuda:
        torch.cuda.synchronize()
    val_time = time.time() - val_start_time
    val_loss = total_running_loss / total_samples_count if total_samples_count > 0 else 0.0

    if vae is None:
        print("⚠️ Nessun VAE fornito per la decodifica delle immagini di validazione.")
        if use_cuda:
            torch.cuda.empty_cache()
        return val_loss, val_time, 0.0

    def decode_latents(latents_tensor):
        if scale_factor != 1.0:
            latents_tensor = latents_tensor / scale_factor
        latents_denorm = denormalizer.denormalize(latents_tensor.to(device)) # type: ignore
        decoded = vae.decode(latents_denorm, return_dict=False)[0] # type: ignore
        return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
    
    # 2 CFG: Uncond, Text)
    sample_x0 = sample_t5 = sample_clip = sample_clip_lens = None
    uncond_t5 = uncond_clip = uncond_clip_lens = base_noise = x_ts = None
    all_decoded_branches = gt_images = comparison_rows = final_grid = None

    for bucket_name, data in bucket_samples.items():
        sample_x0 = data["x0"].to(device)
        sample_t5 = data["t5"].to(device)
        sample_clip = data["clip"].to(device)
        sample_clip_lens = data["clip_lens"].to(device)
        actual_num_samples = sample_x0.size(0)

        uncond_t5 = torch.zeros_like(sample_t5)
        uncond_clip = torch.zeros_like(sample_clip)
        uncond_clip_lens = torch.zeros_like(sample_clip_lens) 
        
        sample_shape = sample_x0.shape[1:]

        base_noise = torch.randn((actual_num_samples, *sample_shape), device=device)
        x_ts = [
            base_noise.clone(), # Uncond
            base_noise.clone()  # Text-guided (CFG)
        ]

        dt = 1.0 / num_sampling_steps
        timesteps = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype_amp, enabled=use_cuda):
            for i in range(num_sampling_steps):
                t_curr = timesteps[i]
                t_batch = torch.full((actual_num_samples,), t_curr.item(), device=device, dtype=torch.float32)

                x_t_dual = torch.cat([x_ts[0], x_ts[1]], dim=0)
                t_dual = torch.cat([t_batch, t_batch], dim=0)
                
                # Rami CFG: 0: Uncond, 1: Text
                y_t5_dual = torch.cat([uncond_t5, sample_t5], dim=0)
                y_clip_dual = torch.cat([uncond_clip, sample_clip], dim=0)
                
                clip_lens_dual = torch.cat([uncond_clip_lens, sample_clip_lens], dim=0)

                v_dual = model(x_t_dual, t_dual, y_clip=y_clip_dual, y_t5=y_t5_dual, clip_lens=clip_lens_dual, cond_images=None)
                v_uncond, v_text = v_dual.chunk(2, dim=0)

                v_b0 = v_uncond
                v_b1 = v_uncond + cfg_scale * (v_text - v_uncond)

                x_ts[0] = x_ts[0] - dt * v_b0
                x_ts[1] = x_ts[1] - dt * v_b1

        all_decoded_branches = []
        for x_t_final in x_ts:
            gen_images = decode_latents(x_t_final)
            all_decoded_branches.append(gen_images)

        gt_images = decode_latents(sample_x0)

        comparison_rows = []
        for sample_idx in range(actual_num_samples):
            row_imgs = [
                gt_images[sample_idx].cpu(),
                all_decoded_branches[0][sample_idx].cpu(), # Uncond
                all_decoded_branches[1][sample_idx].cpu()  # Text CFG
            ]
            comparison_rows.append(torch.stack(row_imgs, dim=0))

        final_grid = torch.cat(comparison_rows, dim=0)
        save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}_{bucket_name}_cfg_comparison.png")
        save_image(final_grid, save_path, nrow=3)
        print(f"✅ Salvati {actual_num_samples} campioni per il bucket {bucket_name} in {save_path}")

    if 'bucket_samples' in locals():
        del bucket_samples
    if 'sample_x0' in locals():
        del sample_x0, sample_t5, sample_clip, sample_clip_lens
    if 'uncond_t5' in locals():
        del uncond_t5, uncond_clip, uncond_clip_lens, base_noise, x_ts
    if 'all_decoded_branches' in locals():
        del all_decoded_branches, gt_images, comparison_rows, final_grid

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    sample_time = time.time() - val_start_time

    return val_loss, val_time, sample_time

def main():
    total_start_time = time.perf_counter()

    train_loaders = {
        "512x512": DataLoader(MultiBucketH5Dataset(["LDM/dataset/cc12m_flux_encoded_512x512.h5"]), batch_size=8, shuffle=True, num_workers=2, drop_last=True, pin_memory=True),
        #"512x768": DataLoader(MultiBucketH5Dataset(["LDM/dataset/cc12m_flux_encoded_512x768.h5"]), batch_size=8, shuffle=True, num_workers=2, drop_last=True, pin_memory=True),
        #"768x512": DataLoader(MultiBucketH5Dataset(["LDM/dataset/cc12m_flux_encoded_768x512.h5"]), batch_size=8, shuffle=True, num_workers=2, drop_last=True, pin_memory=True),
    }
    
    val_loaders = {
        "512x512": DataLoader(MultiBucketH5Dataset(["LDM/dataset/cc12m_flux_encoded_512x512_val.h5"]), batch_size=8, shuffle=False, num_workers=2, pin_memory=True),
        #"512x768": DataLoader(MultiBucketH5Dataset(["LDM/dataset/cc12m_flux_encoded_512x768_val.h5"]), batch_size=8, shuffle=False, num_workers=2, pin_memory=True),
        #"768x512": DataLoader(MultiBucketH5Dataset(["LDM/dataset/cc12m_flux_encoded_768x512_val.h5"]), batch_size=8, shuffle=False, num_workers=2, pin_memory=True),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae_decoder, denormalizer = load_flux2_decoder_for_eval(VAE_WEIGHTS_PATH)

    use_cuda = device.type == "cuda"
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    # inizializing MMDiT
    model = MMDiT(
    patch_size=2, 
    in_channels=32, 
    hidden_size=768,         
    y_dim_clip=512, 
    y_dim_t5=768,
    depth_single=30,        
    depth_double=5,      
    num_heads=12,            # (768 / 12 = 64 dim per head)
    max_cond_images=2
).to(device)
    
    num_epochs = 20
    warmup_epochs = 2

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True 
    )

    steps_per_epoch = sum(len(loader) for loader in train_loaders.values())
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    scheduler_main = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=1e-6,
    )

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=max(1, warmup_steps),
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler_main],
        milestones=[warmup_steps],
    )

    dummy_x = torch.randn(8, 32, 64, 64, device=device)
    dummy_t = torch.randn(8, device=device)
    dummy_y_clip = torch.randn(8, 77, 512, device=device)
    dummy_y_t5 = torch.randn(8, 77, 768, device=device)
    dummy_clip_lens = torch.full((8,), 76, dtype=torch.long, device=device)
    
    try:
        summary(model, input_data=[dummy_x, dummy_t, dummy_y_clip, dummy_y_t5, dummy_clip_lens])
    except Exception as e:
        print(f"Summary saltato o non compatibile: {e}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    CHECKPOINT_PATH = "mmdit_final_checkpoints/mmdit_best.pth"

    # Ripresa dal checkpoint (passando anche lo scheduler)
    if os.path.exists(CHECKPOINT_PATH):
        start_epoch = load_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler=scheduler, device="cuda")
    else:
        start_epoch = 0
        print("Nessun checkpoint trovato. Inizio il training da zero.")

    print(f"Usando device: {device}")

    if hasattr(torch, 'compile'):
        print("--> Compilazione MMDiT con torch.compile()...")
        model = cast(nn.Module, torch.compile(model, mode="default", dynamic=True))
    best_val_loss = float('inf')
    scaler = None


    print(f"Usando device: {device}")

    for epoch in range(start_epoch, num_epochs):
        debug = (epoch == start_epoch)

        train_loss, train_time, samples_per_sec = train(
            model=model, 
            train_loaders=train_loaders, 
            optimizer=optimizer, 
            device=device,
            scaler=scaler,
            scheduler=scheduler,
            max_grad_norm=1.0,
            p_drop_text=0.10,
            debug_first_batch=debug
        )
        raw_model = getattr(model, "_orig_mod", model)
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
        assert isinstance(raw_model, nn.Module)

        val_loss, val_time, sample_time = validate(
            model=raw_model,
            val_loaders=val_loaders,
            device=device,
            epoch=epoch,
            num_sampling_steps=20,
            num_samples_to_save=6,
            save_dir="samples_mmdit",
            vae=vae_decoder,
            denormalizer=denormalizer 
        )

        print(
            f"Epoch {epoch+1:03d}/{num_epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Time: {train_time:.2f}s ({samples_per_sec:.0f} samples/s) | "
            f"Val Time: {val_time:.2f}s | "
            f"Sampling Time: {sample_time:.2f}s"
        )




        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("mmdit_final_checkpoints", exist_ok=True)
            best_ckpt_path = os.path.join("mmdit_final_checkpoints", "mmdit_best.pth")
            best_checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(best_checkpoint, best_ckpt_path)
            print(f"🏆 [Best Model] Nuovo record di Val Loss ({best_val_loss:.4f})!")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            os.makedirs("mmdit_final_checkpoints", exist_ok=True)
            ckpt_path = os.path.join("mmdit_final_checkpoints", f"mmdit_epoch_{epoch+1:03d}.pth")
            
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"[Checkpoint] Salvato con successo in: {ckpt_path}")

    total_elapsed = time.perf_counter() - total_start_time
    print(f"Total Run Time: {total_elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()