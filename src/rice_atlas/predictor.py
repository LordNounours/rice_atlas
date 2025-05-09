import os
import numpy as np
import torch
import tifffile as tiff
from torch import nn
from tqdm import tqdm
from rice_atlas.model.segformer3d import SegFormer3D  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                   patch_size: int = 128, stride: int = 96, batch_size: int = 16) -> np.ndarray:
    print("🔄 Chargement du modèle...")
    model = load_model(model_path, SegFormer3D)

    print(f"🔄 Chargement du volume depuis {volume_path}")
    volume = tiff.imread(volume_path)  # [D, H, W]
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
        print(f"💾 Sauvegarde du volume segmenté dans {output_path}")
        tiff.imwrite(output_path, segmented)

    return segmented
