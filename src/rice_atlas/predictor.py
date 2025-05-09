import os
import numpy as np
import torch
import tifffile as tiff
from torch import nn
from tqdm import tqdm
from rice_atlas.model.segformer3d import SegFormer3D  
from scipy.ndimage import gaussian_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess_volume(volume: np.ndarray, threshold: int = 120, sigma: float = 1.0) -> np.ndarray:
    """
    Applique un prétraitement pour supprimer les bords proches des zones brillantes, 
    puis un lissage gaussien 3D.

    Parameters:
        volume (np.ndarray): Le volume d'entrée [D, H, W].
        threshold (int): Seuil pour les pixels brillants. Par défaut 120.
        sigma (float): Écart type du filtre gaussien. Par défaut 1.0.

    Returns:
        np.ndarray: Le volume prétraité et lissé.
    """
    # Étape 1 : suppression des voisins des pixels brillants
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

    # Étape 2 : lissage gaussien 3D
    smoothed_stack = gaussian_filter(processed_stack, sigma=sigma)

    # (Optionnel) Reconvertir en uint8 si nécessaire
    smoothed_stack = np.clip(smoothed_stack, 0, 255)
    smoothed_stack = smoothed_stack.astype(np.uint8)

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


def extract_patches_with_stride(volume, patch_size=128, stride=64):
    D, H, W = volume.shape
    patches, coords = [], []

    def get_positions(dim):
        pos = list(range(0, dim - patch_size + 1, stride))
        if pos[-1] + patch_size < dim:
            pos.append(dim - patch_size)
        return pos

    for z in get_positions(D):
        for y in get_positions(H):
            for x in get_positions(W):
                patches.append(volume[z:z+patch_size, y:y+patch_size, x:x+patch_size])
                coords.append((z, y, x))

    return patches, coords


def prepare_patches_batch(patches):
    batch = []
    for patch in patches:
        patch = patch.astype(np.float32)
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        batch.append(tensor)
    return torch.cat(batch, dim=0).to(device)


def predict_patches_batch(model, batch_tensor):
    with torch.no_grad():
        preds = model(batch_tensor)
        torch.cuda.empty_cache()
        return torch.argmax(preds, dim=1).cpu().numpy()  # [B, D, H, W]


def segment_volume(model_path: str, volume_path: str, output_path: str = None,
                   patch_size: int = 128, stride: int = 96, batch_size: int = 16 ,pretreatment : bool = False) -> np.ndarray:
    print("🔄 Chargement du modèle...")
    model = load_model(model_path, SegFormer3D)

    print(f"🔄 Chargement du volume depuis {volume_path}")
    volume = tiff.imread(volume_path)  # [D, H, W]

    # Appliquer le prétraitement si pretreatment est activé
    if pretreatment:
        print("🔄 Application du prétraitement...")
        volume = preprocess_volume(volume)  # Appliquer le prétraitement sur le volume

    shape = volume.shape
    segmented = np.zeros(shape, dtype=np.uint8)

    patches, coords = extract_patches_with_stride(volume, patch_size, stride)

    print("🚀 Prédiction...")
    for i in tqdm(range(0, len(patches), batch_size), desc="Prédiction batchée"):
        batch = patches[i:i+batch_size]
        batch_tensor = prepare_patches_batch(batch)
        preds = predict_patches_batch(model, batch_tensor)

        for pred, (z, y, x) in zip(preds, coords[i:i+batch_size]):
            dz = min(patch_size, shape[0] - z)
            dy = min(patch_size, shape[1] - y)
            dx = min(patch_size, shape[2] - x)
            segmented[z:z+dz, y:y+dy, x:x+dx] = pred[:dz, :dy, :dx]

    if output_path:
        segmented_to_save = (segmented * 255).astype(np.uint8)
        print(f"💾 Sauvegarde du volume segmenté dans {output_path}")
        tiff.imwrite(output_path, segmented_to_save)

    return segmented
