import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image
import numpy as np
import lpips
from tqdm import tqdm
from torchinfo import summary
from typing import Optional, Any, Tuple, Dict, cast
from bitsandbytes.optim.adamw import AdamW8bit
from model.vae_model import AutoencoderKL, VAELoss, PatchDiscriminator


class UltraEditMMapDataset(Dataset):
    """
    Dataset ad alte prestazioni che legge le coppie di immagini (input e target)
    da due file mmap distinti e il relativo edit_prompt dal JSON.
    """
    def __init__(
        self, 
        input_mmap_path: str, 
        target_mmap_path: str, 
        json_metadata_path: str,
        image_size: int = 512, 
        channels: int = 3,
        is_hwc: bool = False  
    ):
        super().__init__()
        self.input_mmap_path = input_mmap_path
        self.target_mmap_path = target_mmap_path
        self.json_metadata_path = json_metadata_path
        self.image_size = image_size
        self.channels = channels
        self.is_hwc = is_hwc

        if not os.path.exists(self.json_metadata_path):
            raise FileNotFoundError(f"File di metadati JSON non trovato in '{self.json_metadata_path}'.")

        with open(self.json_metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.num_samples: int = metadata["num_samples"]
        self.prompts: list[str] = [item["edit_prompt"] for item in metadata["data"]]
  
        if self.is_hwc:
            self.shape = (self.num_samples, image_size, image_size, channels)
        else:
            self.shape = (self.num_samples, channels, image_size, image_size)
        
        # lazy initialization
        self.input_data: Optional[np.ndarray] = None
        self.target_data: Optional[np.ndarray] = None

    def _init_mmap(self):
        if not os.path.exists(self.input_mmap_path):
            raise FileNotFoundError(f"File mmap input non trovato in '{self.input_mmap_path}'.")
        if not os.path.exists(self.target_mmap_path):
            raise FileNotFoundError(f"File mmap target non trovato in '{self.target_mmap_path}'.")

        self.input_data = np.memmap(self.input_mmap_path, dtype='uint8', mode='r', shape=self.shape)
        self.target_data = np.memmap(self.target_mmap_path, dtype='uint8', mode='r', shape=self.shape)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.input_data is None or self.target_data is None:
            self._init_mmap()

        assert self.input_data is not None and self.target_data is not None

        
        input_arr = self.input_data[idx]
        target_arr = self.target_data[idx]

        input_tensor = torch.from_numpy(input_arr.copy()).to(torch.float32).div_(127.5).sub_(1.0)
        target_tensor = torch.from_numpy(target_arr.copy()).to(torch.float32).div_(127.5).sub_(1.0)

        if self.is_hwc:
            input_tensor = input_tensor.permute(2, 0, 1).contiguous()
            target_tensor = target_tensor.permute(2, 0, 1).contiguous()

        return {
            "source_image": input_tensor,
            "target_image": target_tensor,
            "edit_prompt": self.prompts[idx]
        }


def build_ultraedit_dataloaders(
    data_dir: str = "./ultraedit_mmap_processed",
    batch_size: int = 3,         
    num_workers: int = 4,        
    is_hwc: bool = False,
    max_train_samples: Optional[int] = None,  
    max_val_samples: Optional[int] = None     
) -> Tuple[DataLoader, DataLoader]:

    train_dataset = UltraEditMMapDataset(
        input_mmap_path=os.path.join(data_dir, "train_input_images.mmap"),
        target_mmap_path=os.path.join(data_dir, "train_target_images.mmap"),
        json_metadata_path=os.path.join(data_dir, "train_metadata.json"),
        is_hwc=is_hwc
    )

    val_dataset = UltraEditMMapDataset(
        input_mmap_path=os.path.join(data_dir, "val_input_images.mmap"),
        target_mmap_path=os.path.join(data_dir, "val_target_images.mmap"),
        json_metadata_path=os.path.join(data_dir, "val_metadata.json"),
        is_hwc=is_hwc
    )

    if max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
        
    if max_val_samples is not None:
        val_dataset = Subset(val_dataset, range(min(max_val_samples, len(val_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader



# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train(
    model: nn.Module,                     
    discriminator: nn.Module,            
    train_loader: DataLoader,            
    optimizer: optim.Optimizer,          
    optimizer_disc: optim.Optimizer,      
    criterion: VAELoss,                   
    device: torch.device, 
    scaler: Optional[torch.amp.GradScaler] = None,           
    scaler_disc: Optional[torch.amp.GradScaler] = None,     
    scheduler: Optional[Any] = None,         
    scheduler_disc: Optional[Any] = None,     
    max_grad_norm: float = 1.0,
    debug_first_batch: bool = True
) -> Tuple[Dict[str, float], float, float]:
    
    model.train()
    discriminator.train()
    
    running_loss = 0.0
    running_l1 = 0.0
    running_lpips = 0.0
    running_kl = 0.0
    running_adv = 0.0
    running_disc_loss = 0.0
    running_abs_mean = 0.0
    running_logvar_mean = 0.0
    total_images = 0
    epoch_start = time.time()

    use_cuda = device.type == "cuda"
    dtype_amp = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    for batch_count, batch in enumerate(tqdm(train_loader, desc="Training VAE + GAN (Source + Target)", leave=False)):
        if debug_first_batch and batch_count == 0:
            batch_start_time = time.time()

        source_img = batch["source_image"].to(device, non_blocking=True) # [B, 3, H, W]
        target_img = batch["target_image"].to(device, non_blocking=True) # [B, 3, H, W]
        batch_size_orig = source_img.size(0)

        # [2B, 3, H, W]
        combined_imgs = torch.cat([source_img, target_img], dim=0)
        optimizer_disc.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=dtype_amp, enabled=use_cuda):
            with torch.no_grad():
                recon_target_fake, _, _ = model(target_img)

            loss_d_real = criterion(
                x=target_img, recon_x=None, source_img=source_img, 
                discriminator=discriminator, for_discriminator=True, is_real=True
            )
            loss_d_fake = criterion(
                x=None, recon_x=recon_target_fake, source_img=source_img, 
                discriminator=discriminator, for_discriminator=True, is_real=False
            )
            loss_disc = (loss_d_real + loss_d_fake) * 0.5

        if scaler_disc is not None:
            scaler_disc.scale(loss_disc).backward()
            scaler_disc.unscale_(optimizer_disc)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
            
            scale_before_disc = scaler_disc.get_scale()
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()
            scale_after_disc = scaler_disc.get_scale()
            
            disc_optimizer_stepped = scale_before_disc <= scale_after_disc
        else:
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
            optimizer_disc.step()
            disc_optimizer_stepped = True

        if scheduler_disc is not None and disc_optimizer_stepped:
            scheduler_disc.step()


        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=dtype_amp, enabled=use_cuda):
          
            recon_combined, mean_combined, logvar_combined = model(combined_imgs)
            
            recon_target = recon_combined[batch_size_orig:]

            loss_vae_combined, loss_dict_combined = criterion(
                x=combined_imgs, 
                recon_x=recon_combined, 
                mean=mean_combined, 
                logvar=logvar_combined, 
                source_img=None, 
                discriminator=None, 
                for_discriminator=False
            )

            if discriminator is not None and criterion.adv_weight > 0:
                preds_fake = discriminator(source_img, recon_target)
                adv_loss = F.mse_loss(preds_fake, torch.ones_like(preds_fake))
            else:
                adv_loss = torch.tensor(0.0, device=device)

            loss_total_vae = loss_vae_combined + (criterion.adv_weight * adv_loss)

        # Backward VAE
        if scaler is not None:
            scaler.scale(loss_total_vae).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            
            optimizer_stepped = scale_before <= scale_after
        else:
            loss_total_vae.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer_stepped = True

        if scheduler is not None and optimizer_stepped:
            scheduler.step()

        if debug_first_batch and batch_count == 0:
            step_time = time.time() - batch_start_time # type: ignore
            print(f"\n[Debug First Batch] Tempo batch 0: {step_time:.4f}s | Loss VAE Total: {loss_total_vae.item():.4f} | Loss Disc: {loss_disc.item():.4f}")

        total_batch_images = combined_imgs.size(0) # 2 * B
        
        running_loss += loss_total_vae.item() * total_batch_images
        running_l1 += loss_dict_combined["l1"] * total_batch_images
        running_lpips += loss_dict_combined["lpips"] * total_batch_images
        running_kl += loss_dict_combined["kl"] * total_batch_images
        running_adv += adv_loss.item() * batch_size_orig
        running_disc_loss += loss_disc.item() * batch_size_orig
        
        with torch.no_grad():
            running_abs_mean += mean_combined.abs().mean().item() * total_batch_images
            running_logvar_mean += logvar_combined.mean().item() * total_batch_images

        total_images += total_batch_images

    epoch_time = time.time() - epoch_start
    samples_per_sec = total_images / epoch_time if epoch_time > 0 else 0.0
    
    metrics = {
        "loss": running_loss / total_images,
        "l1": running_l1 / total_images,
        "lpips": running_lpips / total_images,
        "kl": running_kl / total_images,
        "adv": running_adv / (total_images / 2),
        "disc_loss": running_disc_loss / (total_images / 2),
        "abs_mean": running_abs_mean / total_images,
        "logvar_mean": running_logvar_mean / total_images
    }

    return metrics, epoch_time, samples_per_sec


@torch.no_grad()
def validate(
    model: nn.Module, 
    val_loader: DataLoader, 
    criterion: nn.Module,
    device: torch.device, 
    epoch: int,
    num_samples_to_save: int = 8,
    save_dir: str = "reconstructions_120M"
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    total_images = 0 

    use_cuda = device.type == "cuda"
    dtype_amp = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    if use_cuda:
        torch.cuda.synchronize()
    val_start_time = time.time()

    all_val_sources = []
    all_val_targets = []
    all_val_recons = []

    for step, batch in enumerate(tqdm(val_loader, desc="Validating VAE", leave=False)):
        source_img = batch["source_image"].to(device, non_blocking=True) # [B, 3, H, W]
        target_img = batch["target_image"].to(device, non_blocking=True) # [B, 3, H, W]
        batch_size_orig = source_img.size(0)

        # [2B, 3, H, W]
        combined_imgs = torch.cat([source_img, target_img], dim=0)

        with torch.amp.autocast(device_type=device.type, dtype=dtype_amp, enabled=use_cuda):
            recon_combined, mean_combined, logvar_combined = model(combined_imgs)
            
            _, loss_dict = criterion(
                x=combined_imgs, 
                recon_x=recon_combined, 
                mean=mean_combined, 
                logvar=logvar_combined, 
                source_img=None, 
                discriminator=None, 
                for_discriminator=False
            )

        loss_val = loss_dict["total"].item() if hasattr(loss_dict["total"], "item") else loss_dict["total"]
        running_loss += loss_val * combined_imgs.size(0)
        
        total_images += combined_imgs.size(0)
        if len(all_val_sources) < 4:
            all_val_sources.append(source_img.float().cpu())
            all_val_targets.append(target_img.float().cpu())
            recon_target = recon_combined[batch_size_orig:].float().cpu()
            all_val_recons.append(recon_target)

    if use_cuda:
        torch.cuda.synchronize()
    val_time = time.time() - val_start_time
    val_loss = running_loss / total_images if total_images > 0 else 0.0

        #grid
    if len(all_val_sources) > 0:
        os.makedirs(save_dir, exist_ok=True)
        
        sources_cat = torch.cat(all_val_sources, dim=0)
        targets_cat = torch.cat(all_val_targets, dim=0)
        recons_cat = torch.cat(all_val_recons, dim=0)
        
        total_val_samples = sources_cat.size(0)
        actual_samples_to_save = min(num_samples_to_save, total_val_samples)
        
        sel_source = sources_cat[:actual_samples_to_save]
        sel_target = targets_cat[:actual_samples_to_save]
        sel_recon = recons_cat[:actual_samples_to_save]

        # De-normalization
        src_imgs = (sel_source / 2.0 + 0.5).clamp(0.0, 1.0)
        tgt_imgs = (sel_target / 2.0 + 0.5).clamp(0.0, 1.0)
        rcn_imgs = (sel_recon / 2.0 + 0.5).clamp(0.0, 1.0)

        interleaved = []
        for i in range(actual_samples_to_save):
            interleaved.append(src_imgs[i])
            interleaved.append(tgt_imgs[i])
            interleaved.append(rcn_imgs[i])
            
        comparison = torch.stack(interleaved, dim=0)
        save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}_recon.png")
        
        save_image(comparison, save_path, nrow=3)

    return val_loss, val_time



def main():
    total_start_time = time.perf_counter()
    #PARAMETERS
    IMAGE_SIZE = 512
    LATENT_DIM = 16     
    
    BATCH_SIZE = 3
    NUM_EPOCHS = 20
    LR = 1e-4
    LR_DISC = 1e-5  # LR/10
    WEIGHT_DECAY_DISC = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Target KL Loss Weight and Warmup Schedule
    TARGET_KL_WEIGHT = 1e-5
    KL_WARMUP_EPOCHS = 5    
    LPIPS_WEIGHT = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    train_loader, val_loader = build_ultraedit_dataloaders(
        data_dir="./ultraedit_mmap_processed",
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    model: nn.Module = AutoencoderKL(latent_dim=LATENT_DIM).to(device)
    discriminator: nn.Module = PatchDiscriminator(in_channels=6).to(device)

    criterion = VAELoss(kl_weight=TARGET_KL_WEIGHT, lpips_weight=LPIPS_WEIGHT).to(device)

    # Summary
    dummy_x = torch.randn(6, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    summary(model, input_data=[dummy_x])

    optimizer = AdamW8bit(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    optimizer_disc = AdamW8bit(
        discriminator.parameters(),
        lr=LR_DISC,
        weight_decay=WEIGHT_DECAY_DISC,
        betas=(0.5, 0.999), 
        eps=1e-8
    )

    LR_WARMUP_EPOCHS = 2
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = len(train_loader) * LR_WARMUP_EPOCHS
    min_lr_ratio = 1e-6 / LR  

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    min_lr_ratio_disc = 1e-6 / LR_DISC

    def lr_lambda_disc(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio_disc + (1.0 - min_lr_ratio_disc) * cosine_decay

    scheduler_disc = optim.lr_scheduler.LambdaLR(optimizer_disc, lr_lambda=lr_lambda_disc)

    torch.backends.cudnn.benchmark = True
    # Compilazione del modello
    if hasattr(torch, 'compile'):
        print("--> Compilazione VAE con torch.compile()...")
        model = cast(nn.Module, torch.compile(model, mode="default"))
 
    best_val_loss = float('inf')
    start_epoch: int = 0  

    resume_path = "checkpoints_vae_120M/vae_best.pth" #checkpoint

    if os.path.exists(resume_path):
        print(f"Caricamento checkpoint da {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        
        model_to_load = getattr(model, "_orig_mod", model)
        if hasattr(model_to_load, "module"):
            model_to_load = model_to_load.module
        assert isinstance(model_to_load, nn.Module)
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        disc_to_load = getattr(discriminator, "_orig_mod", discriminator)
        if hasattr(disc_to_load, "module"):
            disc_to_load = disc_to_load.module
        assert isinstance(disc_to_load, nn.Module)
        disc_to_load.load_state_dict(checkpoint['discriminator_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
        
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"--> Training ripreso con successo. Si parte dall'epoca {start_epoch + 1} (Best Val Loss precedente: {best_val_loss:.4f}).")
    else:
        print("--> Nessun checkpoint trovato. Partenza da zero.")

    print("\n--> Inizio Training VAE + GAN...")
    total_start_time = time.perf_counter()

    for epoch in range(start_epoch, NUM_EPOCHS):
        debug = (epoch == 0)

        # KL Annealing Warmup
        current_kl_weight = min(
            TARGET_KL_WEIGHT,
            TARGET_KL_WEIGHT * (epoch + 1) / KL_WARMUP_EPOCHS
        )
        criterion.kl_weight = current_kl_weight  

        train_metrics, train_time, samples_per_sec = train(
            model=model,
            discriminator=discriminator,
            train_loader=train_loader,
            optimizer=optimizer,
            optimizer_disc=optimizer_disc,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            scheduler_disc=scheduler_disc,
            max_grad_norm=1.0,
            debug_first_batch=debug
        )

        val_loss, val_time = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            num_samples_to_save=8,
            save_dir="reconstructions_120M"
        )

        
        print(
            f"Epoch {epoch+1:03d}/{NUM_EPOCHS:03d} | KL Weight: {current_kl_weight:.2e} | "
            f"Train Loss: {train_metrics['loss']:.4f} (L1: {train_metrics['l1']:.4f}, LPIPS: {train_metrics['lpips']:.4f}, KL: {train_metrics['kl']:.4f}, Adv: {train_metrics['adv']:.4f}, Disc Loss: {train_metrics['disc_loss']:.4f}) | "
            f"Latent -> Abs Mean: {train_metrics['abs_mean']:.4f}, LogVar Mean: {train_metrics['logvar_mean']:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Time: {train_time:.2f}s ({samples_per_sec:.0f} samples/s)"
            f"Val Time: {val_time:.2f}s ({samples_per_sec:.0f} samples/s)"
        )

        #extracting raw model
        raw_model = getattr(model, "_orig_mod", model)
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
        assert isinstance(raw_model, nn.Module)
            
        raw_disc = getattr(discriminator, "_orig_mod", discriminator)
        if hasattr(raw_disc, "module"):
            raw_disc = raw_disc.module
        assert isinstance(raw_disc, nn.Module)

        os.makedirs("checkpoints_vae_120M", exist_ok=True)

        # Save Best Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = "checkpoints_vae_120M/vae_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'discriminator_state_dict': raw_disc.state_dict(),
                'optimizer_disc_state_dict': optimizer_disc.state_dict(),
                'val_loss': val_loss
            }, best_ckpt_path)
            print(f"--> [BEST MODEL] Nuovo record Val Loss ({val_loss:.4f})! Salvato in {best_ckpt_path}")

        # periodical save
        if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
            ckpt_path = f"checkpoints_vae_120M/vae_epoch_{epoch+1:03d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'discriminator_state_dict': raw_disc.state_dict(),
                'optimizer_disc_state_dict': optimizer_disc.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)
            print(f"--> Checkpoint periodico salvato: {ckpt_path}")

    total_elapsed = time.perf_counter() - total_start_time
    print(f"\nTraining VAE + GAN Completato in {total_elapsed / 60:.2f} minuti.")

if __name__ == "__main__":
    main()