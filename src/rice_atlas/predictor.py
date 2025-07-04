import os
import numpy as np
import torch
import tifffile as tiff
from torch import nn
from tqdm import tqdm
from rice_atlas.model.segformer3d import SegFormer3D  
from scipy.ndimage import gaussian_filter,binary_dilation
from typing import Tuple
from rice_atlas import denoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess_large_volume(volume, threshold=120, sigma=1.0):
    """
    Applique un prétraitement slice par slice à un volume 3D numpy,
    et retourne le volume traité en mémoire.

    Parameters:
        volume (np.ndarray): Volume 3D (z, y, x).
        threshold (int): Seuil pour la suppression des voisins.
        sigma (float): Paramètre du filtre gaussien.

    Returns:
        np.ndarray: Volume prétraité (uint8).
    """
    if volume.ndim != 3:
        raise ValueError("Le volume doit être un tableau 3D (z, y, x)")

    processed_slices = []

    for idx in tqdm(range(volume.shape[0]), desc="Prétraitement des slices"):
        image = volume[idx]

        if image.ndim > 2:
            raise ValueError("Chaque slice doit être en niveaux de gris")

        neighbors = denoise.suppress_neighbors(image.astype(np.uint8), threshold)
        updated = image.copy()
        updated[neighbors == 255] = 0
        filtered = gaussian_filter(updated, sigma=sigma)

        final = np.clip(filtered, 0, 255).astype(np.uint8)
        processed_slices.append(final)

    return np.stack(processed_slices)


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
    volume: np.ndarray,
    output_path: str = None,
    patch_size: int = 128,
    stride: int = 112,
    batch_size: int = 16,
    pretreatment: bool = False,
    tap_center: Tuple[int, int, int] = (0, 0, 0),
):
    print("🔄 Chargement du modèle...")
    model = load_model(model_path, SegFormer3D)

    if pretreatment:
        print("🔄 Application du prétraitement...")
        volume = preprocess_large_volume(volume)

    shape = volume.shape


    coords_list = compute_patch_coords(shape, patch_size, stride)

    # Limiter coords_list selon tap_center z
    max_z = int(tap_center[2]) + 100
    filtered_coords = [coord for coord in coords_list if coord[0] <= max_z]

    coords_array = np.array(filtered_coords)
    zmin, ymin, xmin = coords_array.min(axis=0)
    zmax = coords_array[:, 0].max() + patch_size
    ymax = coords_array[:, 1].max() + patch_size
    xmax = coords_array[:, 2].max() + patch_size

    subshape = (zmax - zmin, ymax - ymin, xmax - xmin)
    probas_volume = np.zeros(subshape, dtype=np.float32)
    count_map = np.zeros(subshape, dtype=np.float32)
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

                probas_volume[z - zmin:z - zmin + dz,
                            y - ymin:y - ymin + dy,
                            x - xmin:x - xmin + dx] += patch_prob[:dz, :dy, :dx]
                count_map[z - zmin:z - zmin + dz,
                        y - ymin:y - ymin + dy,
                        x - xmin:x - xmin + dx] += 1

            buffer, buffer_coords = [], []

    if buffer:
        batch_tensor = prepare_patches_batch(buffer)
        batch_probs, _ = predict_patches_batch(model, batch_tensor)

        for patch_prob, (z, y, x) in zip(batch_probs, buffer_coords):
            dz = min(patch_size, shape[0] - z)
            dy = min(patch_size, shape[1] - y)
            dx = min(patch_size, shape[2] - x)

            probas_volume[z - zmin:z - zmin + dz,
                y - ymin:y - ymin + dy,
                x - xmin:x - xmin + dx] += patch_prob[:dz, :dy, :dx]
            count_map[z - zmin:z - zmin + dz,
                y - ymin:y - ymin + dy,
                x - xmin:x - xmin + dx] += 1


    print("📊 Moyennage des probabilités par blocs...")
    average_predictions_in_chunks(probas_volume, count_map, block_size=32)

    print("Fin de moyennage")
    binary_segmentation = binarize_to_memmap(probas_volume, threshold=0.5)

    print("Valeurs uniques (segmentation binaire) :", np.unique(binary_segmentation))

    if output_path:
        print(f"💾 Sauvegarde de la segmentation binaire dans {output_path}")
        with tiff.TiffWriter(output_path, bigtiff=True) as tif:
            for z in tqdm(range(binary_segmentation.shape[0]), desc="📸 Sauvegarde des slices"):
                tif.write(binary_segmentation[z], contiguous=True)

    return probas_volume, binary_segmentation , (zmin,ymin,xmin)


def average_predictions_in_chunks(probas_volume, count_map, block_size=32):
    z_max, y_max, x_max = probas_volume.shape

    for z in range(0, z_max, block_size):
        for y in range(0, y_max, block_size):
            for x in range(0, x_max, block_size):
                z_end = min(z + block_size, z_max)
                y_end = min(y + block_size, y_max)
                x_end = min(x + block_size, x_max)

                # Extraire blocs
                p_block = probas_volume[z:z_end, y:y_end, x:x_end]
                c_block = count_map[z:z_end, y:y_end, x:x_end]

                # Éviter division par 0
                nonzero_mask = c_block > 0

                # Diviser uniquement où count > 0
                p_block[nonzero_mask] = (
                    p_block[nonzero_mask] / c_block[nonzero_mask]
                )

                # Remettre dans le volume
                probas_volume[z:z_end, y:y_end, x:x_end] = p_block


def binarize_to_memmap(
    probas_volume: np.ndarray,
    threshold: float = 0.5,
    mmap_path: str = "temp/binary_segmentation.dat",
    flush_every: int = 256,  # nombre de slices avant flush
) -> np.memmap:
    os.makedirs(os.path.dirname(mmap_path), exist_ok=True)

    shape = probas_volume.shape
    binary_seg = np.memmap(mmap_path, dtype=np.uint8, mode="w+", shape=shape)

    for z in tqdm(range(shape[0]), desc="Binarisation memmap slice par slice"):
        np.greater_equal(probas_volume[z], threshold, out=probas_volume[z])  # in-place
        binary_seg[z] = probas_volume[z].astype(np.uint8)

        # Flush périodique (évite de trop buffer, mais ne flush pas trop souvent)
        if flush_every and (z + 1) % flush_every == 0:
            binary_seg.flush()

    binary_seg.flush()  # dernier flush
    return binary_seg

def segment_volume_leaf(
    model_path: str,
    volume_path: str,
    output_path: str = None,
    patch_size: int = 128,
    stride: int = 112,
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
        volume = preprocess_large_volume(volume)

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


