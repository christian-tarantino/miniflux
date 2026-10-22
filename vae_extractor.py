import os
import csv
import io
import json
import contextlib
import h5py
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict, Any, cast
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPTokenizer, CLIPTextModel, T5Tokenizer, T5EncoderModel
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from safetensors.torch import load_file

# ==========================================
# 0. CONFIGURAZIONE DEL LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("preprocessing_async.log"),
        logging.StreamHandler()
    ]
)

# ==========================================
# 1. CONFIGURAZIONE E PATHS
# ==========================================
TSV_METADATA_PATH = "cc12m/cc12m.tsv"
OUTPUT_H5_PREFIX = "cc12m_flux_encoded"
CHECKPOINT_STATE_PATH = "async_preprocessing_checkpoint.json"
VAE_WEIGHTS_PATH = "LDM/flux2/flux2-vae.safetensors"
BATCH_SIZE = 8
MAX_CONCURRENT_DOWNLOADS = 256
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=15, connect=5)
TARGET_SAMPLES = 500000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if BF16_OK else torch.float16
T5_DTYPE = torch.bfloat16 if BF16_OK else torch.float32

T5_MODEL_ID = "google/t5-v1_1-base"
T5_MAX_LENGTH = 128
LATENT_CHANNELS = 32      # VAE flux2
VAE_SCALE = 8             # downsampling 
SHIFT_FACTOR_OVERRIDE: Optional[float] = None
SCALING_FACTOR_OVERRIDE: Optional[float] = None

RESOLUTIONS = [
    (512, 512),
    (512, 768),
    (768, 512)
]

#FILTERS

def is_valid_caption(caption: str) -> bool:
    caption = caption.strip()
    words = caption.split()
    if len(words) < 3 or len(caption) < 15:
        return False
    if "http" in caption or "www." in caption or ".com" in caption:
        return False
    return True

def find_best_bucket(w: int, h: int, resolutions: List[Tuple[int, int]]) -> Tuple[int, int]:
    target_aspect = w / h
    best_bucket = resolutions[0]
    min_diff = float('inf')

    for (bh, bw) in resolutions:
        bucket_aspect = bw / bh
        diff = abs(target_aspect - bucket_aspect)
        if diff < min_diff:
            min_diff = diff
            best_bucket = (bh, bw)

    return best_bucket

def center_crop_resize(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    w, h = img.size
    target_aspect = target_w / target_h
    img_aspect = w / h

    if img_aspect > target_aspect:
        new_w = int(h * target_aspect)
        offset = (w - new_w) // 2
        img = img.crop((offset, 0, offset + new_w, h))
    else:
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        img = img.crop((0, offset, w, offset + new_h))

    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

def process_image_bytes(content: bytes, caption: str, resolutions: List[Tuple[int, int]]) -> Optional[Tuple[torch.Tensor, str, Tuple[int, int]]]:
    try:
        image_bytes = io.BytesIO(content)
        with Image.open(image_bytes) as img:
            img = img.convert("RGB")
            w, h = img.size
            aspect_ratio = w / h

            if min(w, h) < 512 or (w * h) < 262144:
                return None
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                return None

            img_np = np.array(img)
            if img_np.std() < 15.0:
                return None

            best_bucket = find_best_bucket(w, h, resolutions)
            target_h, target_w = best_bucket

            image = center_crop_resize(img, target_h, target_w)

            img_np = np.array(image, dtype=np.float32)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) / 127.5 - 1.0

            return img_tensor, caption, (target_h, target_w)
    except Exception:
        return None

#ASYNC PRODUCER
async def download_worker(session: aiohttp.ClientSession, queue: asyncio.Queue, sem: asyncio.Semaphore, url: str, caption: str, line_idx: int, resolutions: List[Tuple[int, int]], cpu_executor: ThreadPoolExecutor):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    async with sem:
        try:
            async with session.get(url, headers=headers, timeout=HTTP_TIMEOUT) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(cpu_executor, process_image_bytes, content, caption, resolutions)
                    if result is not None:
                        img_tensor, cap, bucket = result
                        await queue.put((line_idx, img_tensor, cap, bucket))
        except Exception:
            pass

async def feed_queue(tsv_path: str, queue: asyncio.Queue, resolutions: List[Tuple[int, int]], start_line_idx: int):
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS, force_close=True)
    cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

            tasks = []
            logging.info(f"Lettura TSV e avvio download asincrono a partire dalla riga {start_line_idx}...")

            with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                for line_idx, row in enumerate(reader):
                    if line_idx < start_line_idx:
                        continue
                    if len(row) < 2:
                        continue
                    url, caption = row[0], row[1]

                    if not is_valid_caption(caption):
                        continue

                    task = asyncio.create_task(download_worker(session, queue, sem, url, caption, line_idx, resolutions, cpu_executor))
                    tasks.append(task)

                    if len(tasks) >= MAX_CONCURRENT_DOWNLOADS * 4:
                        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        tasks = list(pending)
                        await asyncio.sleep(0)

            if tasks:
                await asyncio.gather(*tasks)

            await queue.put(None)
    finally:
        cpu_executor.shutdown(wait=False)


#BATCH PROCESSING GPU PER BUCKET

@torch.no_grad()
def process_and_write_batch(
    bucket: Tuple[int, int],
    imgs_list: List[torch.Tensor],
    captions_list: List[str],
    vae: AutoencoderKLFlux2,
    normalizer: "LatentNormalizer",
    clip_tokenizer: Any,
    clip_text_encoder: nn.Module,
    t5_tokenizer: Any,
    t5_text_encoder: nn.Module,
    datasets_dict: Dict[Tuple[int, int], Dict[str, h5py.Dataset]]
):
    bs = len(imgs_list)
    imgs = torch.stack(imgs_list).to(device=DEVICE, dtype=DTYPE)

    # --- Encoder VAE Flux2 ---
    mean_latents = vae.encode(imgs, return_dict=False)[0].mode()
    if mean_latents.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"Expected {LATENT_CHANNELS} latent channels, got {mean_latents.shape[1]}.")
    mean_latents = normalizer.normalize(mean_latents)

    # --- CLIP Tokenization & Encoding ---
    clip_tokens = clip_tokenizer(captions_list, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    clip_seq_lens = clip_tokens['attention_mask'].sum(dim=1) - 1  # Indice esatto del token EOS per ciascuna caption
    
    clip_tokens = {k: v.to(DEVICE) for k, v in clip_tokens.items()}
    clip_outputs = clip_text_encoder(**clip_tokens, output_hidden_states=True)
    clip_penultimate = clip_outputs.hidden_states[-2]

    # --- T5 Tokenization & Encoding ---
    t5_tokens = t5_tokenizer(captions_list, padding="max_length", truncation=True, max_length=T5_MAX_LENGTH, return_tensors="pt")
    t5_tokens = {k: v.to(DEVICE) for k, v in t5_tokens.items()}
    t5_hidden = t5_text_encoder(**t5_tokens).last_hidden_state

    lat_h, lat_w = mean_latents.shape[2], mean_latents.shape[3]

    ds = datasets_dict[bucket]
    lat_ds = ds['latents']
    clip_ds = ds['clip_hidden_states']
    clip_len_ds = ds['clip_lens']
    t5_ds = ds['t5_hidden_states']
    h_ds = ds['latent_h']
    w_ds = ds['latent_w']

    current_size = lat_ds.shape[0]
    new_size = current_size + bs

    lat_ds.resize((new_size, LATENT_CHANNELS, lat_h, lat_w))
    clip_ds.resize((new_size, 77, 512))
    clip_len_ds.resize((new_size,))
    t5_ds.resize((new_size, T5_MAX_LENGTH, 768))
    h_ds.resize((new_size,))
    w_ds.resize((new_size,))

    lat_numpy = mean_latents.cpu().to(torch.float16).numpy()
    clip_numpy = clip_penultimate.cpu().to(torch.float16).numpy()
    clip_lens_numpy = clip_seq_lens.cpu().to(torch.int32).numpy()
    t5_numpy = t5_hidden.cpu().to(torch.float16).numpy()

    # writing in h5
    lat_ds[current_size:new_size] = lat_numpy
    clip_ds[current_size:new_size] = clip_numpy
    clip_len_ds[current_size:new_size] = clip_lens_numpy
    t5_ds[current_size:new_size] = t5_numpy
    h_ds[current_size:new_size] = lat_h
    w_ds[current_size:new_size] = lat_w


def patchify_latents(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, h // 2, w // 2)

def unpatchify_latents(x: torch.Tensor) -> torch.Tensor:
    b, c4, h, w = x.shape
    x = x.reshape(b, c4 // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c4 // 4, h * 2, w * 2)

class LatentNormalizer:
    def __init__(self, vae: AutoencoderKLFlux2):
        bn = getattr(vae, "bn", None)
        if bn is None:
            raise ValueError("Il VAE non ha un modulo 'bn': impossibile normalizzare i latenti Flux2.")
        mean = bn.running_mean.detach().float().cpu()
        var = bn.running_var.detach().float().cpu()
        if torch.allclose(mean, torch.zeros_like(mean)) and torch.allclose(var, torch.ones_like(var)):
            raise ValueError(
                "Le statistiche del BatchNorm sono ai valori di default (mean=0, var=1): "
                "i pesi 'bn.running_mean/running_var' non sono stati caricati dal checkpoint."
            )
        self.eps = float(getattr(vae.config, "batch_norm_eps", getattr(bn, "eps", 1e-4)))
        self.bn_mean = mean.view(1, -1, 1, 1).to(DEVICE)
        self.bn_var = var.view(1, -1, 1, 1).to(DEVICE)
        self.bn_std = torch.sqrt(self.bn_var + self.eps)

        shift = SHIFT_FACTOR_OVERRIDE if SHIFT_FACTOR_OVERRIDE is not None else getattr(vae.config, "shift_factor", None)
        scale = SCALING_FACTOR_OVERRIDE if SCALING_FACTOR_OVERRIDE is not None else getattr(vae.config, "scaling_factor", None)
        self.shift = 0.0 if shift is None else float(shift)
        self.scale = 1.0 if scale is None else float(scale)
        logging.info(
            f"Normalizzazione latenti: BN su {mean.numel()} canali (eps={self.eps}), "
            f"shift_factor={self.shift}, scaling_factor={self.scale}"
        )

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        x = patchify_latents(z.float())
        x = (x - self.bn_mean) / self.bn_std
        x = (x - self.shift) * self.scale
        return unpatchify_latents(x)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        x = patchify_latents(z.float())
        x = x / self.scale + self.shift
        x = x * self.bn_std + self.bn_mean
        return unpatchify_latents(x)

def load_flux2_vae(weights_path: str) -> Tuple[AutoencoderKLFlux2, "LatentNormalizer"]:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pesi VAE Flux2 non trovati: {weights_path}")

    logging.info(f"Caricamento VAE Flux2 da {weights_path}...")
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
    del state_dict, converted_state_dict
    normalizer = LatentNormalizer(vae)
    vae.to(device=DEVICE, dtype=DTYPE) # type: ignore # type: ignore
    vae.eval()
    vae.requires_grad_(False)
    logging.info("VAE Flux2 caricato con successo.")
    return vae, normalizer

def save_norm_attrs(normalizer: "LatentNormalizer", h5_handles: Dict[Tuple[int, int], h5py.File]):
    for f_h5 in h5_handles.values():
        f_h5.attrs["latents_normalized"] = True
        f_h5.attrs["bn_mean"] = normalizer.bn_mean.flatten().cpu().numpy()
        f_h5.attrs["bn_var"] = normalizer.bn_var.flatten().cpu().numpy()
        f_h5.attrs["bn_eps"] = normalizer.eps
        f_h5.attrs["shift_factor"] = normalizer.shift
        f_h5.attrs["scaling_factor"] = normalizer.scale
        f_h5.attrs["patch_size"] = 2


async def main():
    logging.info(f"Inizializzazione Preprocessing Asincrono Multi-File con Bucket su: {DEVICE}")

    start_line_idx = 0
    if os.path.exists(CHECKPOINT_STATE_PATH):
        with open(CHECKPOINT_STATE_PATH, 'r') as cf:
            chk = json.load(cf)
            start_line_idx = chk.get("last_tsv_line", 0)
        logging.info(f"Ripresa dal checkpoint TSV alla riga: {start_line_idx}")

    vae, normalizer = load_flux2_vae(VAE_WEIGHTS_PATH)

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device=DEVICE, dtype=DTYPE) # type: ignore
    clip_text_encoder.eval()

    t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_ID, use_fast=False)
    t5_text_encoder = T5EncoderModel.from_pretrained(T5_MODEL_ID).to(device=DEVICE, dtype=T5_DTYPE) # type: ignore
    t5_text_encoder.eval()

    h5_handles: Dict[Tuple[int, int], h5py.File] = {}
    datasets_dict = {}

    for (h, w) in RESOLUTIONS:
        file_path = f"{OUTPUT_H5_PREFIX}_{w}x{h}.h5"
        file_mode = 'a' if os.path.exists(file_path) else 'w'
        f_h5 = h5py.File(file_path, file_mode, libver='latest')
        h5_handles[(h, w)] = f_h5

        lat_h_dim = h // VAE_SCALE
        lat_w_dim = w // VAE_SCALE

        if 'latents' not in f_h5:
            latents_ds = f_h5.create_dataset('latents', shape=(0, LATENT_CHANNELS, lat_h_dim, lat_w_dim), maxshape=(None, LATENT_CHANNELS, lat_h_dim, lat_w_dim), dtype='float16', chunks=(BATCH_SIZE, LATENT_CHANNELS, lat_h_dim, lat_w_dim))
            clip_hidden_ds = f_h5.create_dataset('clip_hidden_states', shape=(0, 77, 512), maxshape=(None, 77, 512), dtype='float16', chunks=(BATCH_SIZE, 77, 512))
            clip_len_ds = f_h5.create_dataset('clip_lens', shape=(0,), maxshape=(None,), dtype='int32', chunks=(1024,))
            t5_hidden_ds = f_h5.create_dataset('t5_hidden_states', shape=(0, T5_MAX_LENGTH, 768), maxshape=(None, T5_MAX_LENGTH, 768), dtype='float16', chunks=(BATCH_SIZE, T5_MAX_LENGTH, 768))
            heights_ds = f_h5.create_dataset('latent_h', shape=(0,), maxshape=(None,), dtype='int32', chunks=(1024,))
            widths_ds = f_h5.create_dataset('latent_w', shape=(0,), maxshape=(None,), dtype='int32', chunks=(1024,))
        else:
            if not f_h5.attrs.get("latents_normalized", False):
                raise ValueError(
                    f"'{file_path}' contiene latenti non normalizzati (formato precedente). "
                    f"Spostalo o rinominalo prima di rieseguire, per non mischiare formati diversi."
                )
            latents_ds = cast(h5py.Dataset, f_h5['latents'])
            if latents_ds.ndim != 4 or latents_ds.shape[1] != LATENT_CHANNELS:
                raise ValueError(
                    f"Existing latent dataset in '{file_path}' has shape {latents_ds.shape}; "
                    f"expected [N, {LATENT_CHANNELS}, H, W]. Move the old file before re-encoding."
                )
            clip_hidden_ds = cast(h5py.Dataset, f_h5['clip_hidden_states'])
            clip_len_ds = cast(h5py.Dataset, f_h5['clip_lens'])
            t5_hidden_ds = cast(h5py.Dataset, f_h5['t5_hidden_states'])
            heights_ds = cast(h5py.Dataset, f_h5['latent_h'])
            widths_ds = cast(h5py.Dataset, f_h5['latent_w'])

        datasets_dict[(h, w)] = {
            'latents': latents_ds,
            'clip_hidden_states': clip_hidden_ds,
            'clip_lens': clip_len_ds,
            't5_hidden_states': t5_hidden_ds,
            'latent_h': heights_ds,
            'latent_w': widths_ds
        }

    save_norm_attrs(normalizer, h5_handles)

    total_saved_samples = sum(ds['latents'].shape[0] for ds in datasets_dict.values())
    logging.info(f"File HDF5 Multi-Bucket attivi (Latest Format). Campioni totali già salvati: {total_saved_samples}")

    queue = asyncio.Queue(maxsize=2000)
    producer_task = asyncio.create_task(feed_queue(TSV_METADATA_PATH, queue, RESOLUTIONS, start_line_idx))

    progress_bar = tqdm(total=TARGET_SAMPLES, initial=total_saved_samples, desc="Preprocessing Multi-File")

    bucket_batches = {bucket: {'line_idxs': [], 'imgs': [], 'caps': []} for bucket in RESOLUTIONS}
    max_processed_line = start_line_idx

    def run_batch(bucket, b_dict):
        process_and_write_batch(
            bucket=bucket,
            imgs_list=b_dict['imgs'],
            captions_list=b_dict['caps'],
            vae=vae,
            normalizer=normalizer,
            clip_tokenizer=clip_tokenizer,
            clip_text_encoder=clip_text_encoder,
            t5_tokenizer=t5_tokenizer,
            t5_text_encoder=t5_text_encoder,
            datasets_dict=datasets_dict
        )

    def checkpoint_line() -> int:
        pending = [min(b['line_idxs']) for b in bucket_batches.values() if b['line_idxs']]
        return min(pending + [max_processed_line])

    try:
        while True:
            item = await queue.get()
            if item is None:
                break

            line_idx, img_tensor, caption, bucket_size = item
            b_dict = bucket_batches[bucket_size]

            b_dict['line_idxs'].append(line_idx)
            b_dict['imgs'].append(img_tensor)
            b_dict['caps'].append(caption)

            if line_idx > max_processed_line:
                max_processed_line = line_idx

            if len(b_dict['imgs']) == BATCH_SIZE:
                run_batch(bucket_size, b_dict)

                processed_count = len(b_dict['imgs'])
                total_saved_samples += processed_count
                progress_bar.update(processed_count)

                b_dict['line_idxs'].clear()
                b_dict['imgs'].clear()
                b_dict['caps'].clear()

                with open(CHECKPOINT_STATE_PATH, 'w') as cf:
                    json.dump({"last_tsv_line": checkpoint_line()}, cf)

            if total_saved_samples >= TARGET_SAMPLES:
                logging.info(" Target di 500.000 campioni raggiunto.")
                producer_task.cancel()
                break

        for bucket, b_dict in bucket_batches.items():
            if b_dict['imgs']:
                run_batch(bucket, b_dict)
                n = len(b_dict['imgs'])
                total_saved_samples += n
                progress_bar.update(n)
                b_dict['line_idxs'].clear()
                b_dict['imgs'].clear()
                b_dict['caps'].clear()

    finally:
        progress_bar.close()
        if not producer_task.done():
            producer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await producer_task
        for f_h5 in h5_handles.values():
            f_h5.close()

    if os.path.exists(CHECKPOINT_STATE_PATH):
        os.remove(CHECKPOINT_STATE_PATH)

    logging.info(f" Preprocessing asincrono multi-file completato con successo! Totale campioni salvati: {total_saved_samples}.")

if __name__ == "__main__":
    asyncio.run(main())