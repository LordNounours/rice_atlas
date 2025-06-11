import os
import numpy as np
import torch
import tifffile as tiff
from torch import nn
from tqdm import tqdm
from rice_atlas.model.segformer3d import SegFormer3D  
from scipy.ndimage import gaussian_filter
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess_volume(volume: np.ndarray, threshold: int = 120, sigma: float = 1.0) -> np.ndarray:
    processed_slices = []

    for slice_idx in range(volume.shape[0]):
        image = volume[slice_idx]
        neighbors_image = np.zeros_like(image)
        thresholded_image = image >= threshold

        rows, cols = image.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if thresholded_image[i, j]:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                neighbors_image[ni, nj] = 255

        updated_image = image.copy()
        updated_image[neighbors_image == 255] = 0
        processed_slices.append(updated_image)

    processed_stack = np.array(processed_slices, dtype=np.float32)
    smoothed_stack = gaussian_filter(processed_stack, sigma=sigma)
    smoothed_stack = np.clip(smoothed_stack, 0, 255).astype(np.uint8)

    return smoothed_stack

def load_model(model_path, model_architecture):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier '{model_path}' n'existe pas.")

    model = model_architecture()
    model.segformer_decoder.linear_pred = nn.Conv3d(256, 2, kernel_size=1)

    checkpoint = torch.load(model_path, map_location="cpu")
    pretrained_dict = checkpoint.get("state_dict", checkpoint)
    model_dict = model.state_dict()

    compatible_weights = {k: v for k, v in pretrained_dict.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(compatible_weights)
    model.load_state_dict(model_dict)
    model.to(device).eval()

    print(f"✔️  {len(compatible_weights)} poids chargés.")
    return model

def compute_patch_coords(volume_shape, patch_size=128, stride=64):
    D, H, W = volume_shape
    coords = []

    def get_positions(dim):
        pos = list(range(0, dim - patch_size + 1, stride))
        if pos[-1] + patch_size < dim:
            pos.append(dim - patch_size)
        return pos

    for z in get_positions(D):
        for y in get_positions(H):
            for x in get_positions(W):
                coords.append((z, y, x))
    return coords

def prepare_patches_batch(patches):
    batch = []
    for patch in patches:
        patch = patch.astype(np.float32)
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
        batch.append(tensor)
    return torch.cat(batch, dim=0).to(device)

def predict_patches_batch(model, patches_tensor):
    with torch.no_grad():
        preds = model(patches_tensor)
        probs = torch.softmax(preds, dim=1)
        class1_probs = probs[:, 1]  # [batch_size, D, H, W]
        binary_preds = torch.argmax(probs, dim=1)  # [batch_size, D, H, W]
        torch.cuda.empty_cache()
        return class1_probs.cpu().numpy() ,binary_preds.cpu().numpy()

def segment_volume_root(
    model_path: str,
    volume_path: str,
    output_path: str = None,
    patch_size: int = 128,
    stride: int = 96,
    batch_size: int = 16,
    pretreatment: bool = False,
    tap_center: Tuple[int, int, int] = (0, 0, 0),
):
    print("🔄 Chargement du modèle...")
    model = load_model(model_path, SegFormer3D)

    print(f"🔄 Chargement du volume depuis {volume_path}")
    volume = tiff.imread(volume_path)

    if pretreatment:
        print("🔄 Application du prétraitement...")
        volume = preprocess_volume(volume)

    shape = volume.shape
    probas_volume = np.zeros(shape, dtype=np.float32)
    count_map = np.zeros(shape, dtype=np.float32)

    coords_list = compute_patch_coords(shape, patch_size, stride)

    # Limiter coords_list selon tap_center z
    max_z = int(tap_center[2]) + 100
    filtered_coords = [coord for coord in coords_list if coord[0] <= max_z]

    print(f"🚀 Prédiction sur {len(filtered_coords)} patches (limité par tap_center z={max_z})...")

    buffer, buffer_coords = [], []

    for coord in tqdm(filtered_coords, desc="🔮 Prédiction batchée"):
        z, y, x = coord
        patch = volume[z:z+patch_size, y:y+patch_size, x:x+patch_size]
        buffer.append(patch)
        buffer_coords.append(coord)

        if len(buffer) == batch_size:
            batch_tensor = prepare_patches_batch(buffer)
            batch_probs, _ = predict_patches_batch(model, batch_tensor)

            for patch_prob, (z, y, x) in zip(batch_probs, buffer_coords):
                dz = min(patch_size, shape[0] - z)
                dy = min(patch_size, shape[1] - y)
                dx = min(patch_size, shape[2] - x)

                probas_volume[z:z+dz, y:y+dy, x:x+dx] += patch_prob[:dz, :dy, :dx]
                count_map[z:z+dz, y:y+dy, x:x+dx] += 1

            buffer, buffer_coords = [], []

    if buffer:
        batch_tensor = prepare_patches_batch(buffer)
        batch_probs, _ = predict_patches_batch(model, batch_tensor)

        for patch_prob, (z, y, x) in zip(batch_probs, buffer_coords):
            dz = min(patch_size, shape[0] - z)
            dy = min(patch_size, shape[1] - y)
            dx = min(patch_size, shape[2] - x)

            probas_volume[z:z+dz, y:y+dy, x:x+dx] += patch_prob[:dz, :dy, :dx]
            count_map[z:z+dz, y:y+dy, x:x+dx] += 1

    print("📊 Moyennage des probabilités (zone prédite uniquement)...")

    # ➕ Moyenne uniquement sur la zone réellement prédite
    zs, ys, xs = zip(*filtered_coords)
    zmin, zmax = min(zs), max(zs) + patch_size
    ymin, ymax = min(ys), max(ys) + patch_size
    xmin, xmax = min(xs), max(xs) + patch_size

    sub_count = count_map[zmin:zmax, ymin:ymax, xmin:xmax]
    sub_count[sub_count == 0] = 1

    probas_volume[zmin:zmax, ymin:ymax, xmin:xmax] /= sub_count

    binary_segmentation = (probas_volume >= 0.5).astype(np.uint8)
    print("Valeurs uniques (segmentation binaire) :", np.unique(binary_segmentation))

    if output_path:
        print(f"💾 Sauvegarde de la segmentation binaire dans {output_path}")
        with tiff.TiffWriter(output_path, bigtiff=True) as tif:
            for z in tqdm(range(binary_segmentation.shape[0]), desc="📸 Sauvegarde des slices"):
                tif.write(binary_segmentation[z], contiguous=True)

    return probas_volume, binary_segmentation


def segment_volume_leaf(
    model_path: str,
    volume_path: str,
    output_path: str = None,
    patch_size: int = 128,
    stride: int = 96,
    batch_size: int = 16,
    pretreatment: bool = False,
    tap_center: Tuple[int, int, int] = (0, 0, 0),
):
    print("🔄 Chargement du modèle...")
    model = load_model(model_path, SegFormer3D)

    print(f"🔄 Chargement du volume depuis {volume_path}")
    volume = tiff.imread(volume_path)

    if pretreatment:
        print("🔄 Application du prétraitement...")
        volume = preprocess_volume(volume)

    shape = volume.shape
    probas_volume = np.zeros(shape, dtype=np.float32)
    count_map = np.zeros(shape, dtype=np.float32)

    coords_list = compute_patch_coords(shape, patch_size, stride)

    # Limiter coords_list selon tap_center z
    min_z = int(tap_center[2])
    filtered_coords = [coord for coord in coords_list if coord[0] >= min_z]

    print(f"🚀 Prédiction sur {len(filtered_coords)} patches (limité par tap_center z={min_z})...")

    buffer, buffer_coords = [], []

    for coord in tqdm(filtered_coords, desc="🔮 Prédiction batchée"):
        z, y, x = coord
        patch = volume[z:z+patch_size, y:y+patch_size, x:x+patch_size]
        buffer.append(patch)
        buffer_coords.append(coord)

        if len(buffer) == batch_size:
            batch_tensor = prepare_patches_batch(buffer)
            batch_probs, _ = predict_patches_batch(model, batch_tensor)

            for patch_prob, (z, y, x) in zip(batch_probs, buffer_coords):
                dz = min(patch_size, shape[0] - z)
                dy = min(patch_size, shape[1] - y)
                dx = min(patch_size, shape[2] - x)

                probas_volume[z:z+dz, y:y+dy, x:x+dx] += patch_prob[:dz, :dy, :dx]
                count_map[z:z+dz, y:y+dy, x:x+dx] += 1

            buffer, buffer_coords = [], []

    if buffer:
        batch_tensor = prepare_patches_batch(buffer)
        batch_probs, _ = predict_patches_batch(model, batch_tensor)

        for patch_prob, (z, y, x) in zip(batch_probs, buffer_coords):
            dz = min(patch_size, shape[0] - z)
            dy = min(patch_size, shape[1] - y)
            dx = min(patch_size, shape[2] - x)

            probas_volume[z:z+dz, y:y+dy, x:x+dx] += patch_prob[:dz, :dy, :dx]
            count_map[z:z+dz, y:y+dy, x:x+dx] += 1

    print("📊 Moyennage des probabilités (zone prédite uniquement)...")

    # ➕ Moyenne uniquement sur la zone réellement prédite
    zs, ys, xs = zip(*filtered_coords)
    zmin, zmax = min(zs), max(zs) + patch_size
    ymin, ymax = min(ys), max(ys) + patch_size
    xmin, xmax = min(xs), max(xs) + patch_size

    sub_count = count_map[zmin:zmax, ymin:ymax, xmin:xmax]
    sub_count[sub_count == 0] = 1

    probas_volume[zmin:zmax, ymin:ymax, xmin:xmax] /= sub_count

    binary_segmentation = (probas_volume >= 0.5).astype(np.uint8)
    print("Valeurs uniques (segmentation binaire) :", np.unique(binary_segmentation))

    if output_path:
        print(f"💾 Sauvegarde de la segmentation binaire dans {output_path}")
        with tiff.TiffWriter(output_path, bigtiff=True) as tif:
            for z in tqdm(range(binary_segmentation.shape[0]), desc="📸 Sauvegarde des slices"):
                tif.write(binary_segmentation[z], contiguous=True)

    return probas_volume, binary_segmentation


