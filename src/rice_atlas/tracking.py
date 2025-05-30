import tifffile as tiff
import numpy as np
import heapq
import math
import os
from scipy.ndimage import binary_erosion
from skimage.filters import threshold_otsu
from collections import deque
from skimage.measure import label, regionprops
from skimage.morphology import ball
import random
from tqdm import tqdm
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

VOLUME_PATH = "/tmp/volume_proba_red.npy"
MASK_PATH = "/tmp/border_mask.npy"

def save_arrays(volume_proba_red, border_mask):
    np.save(VOLUME_PATH, volume_proba_red)
    np.save(MASK_PATH, border_mask)

def load_memmap_arrays():
    volume = np.load(VOLUME_PATH, mmap_mode='r')
    mask = np.load(MASK_PATH, mmap_mode='r')
    return volume, mask

def path_for_one_root(start,center):
    print(f"Start : {start}")
    volume, border_mask = load_memmap_arrays()
    end, came_from = astar_to_border_with_map(volume, start, border_mask, center)
    if end is not None:
        path = reconstruct_path(came_from, end, start)
        print(f"Longueur de chemin : {len(path)}")
        return (start, end, path)
    return None


def dijkstra_parallel(starts, center,max_workers=4):
    all_paths = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(path_for_one_root, start,center) for start in starts]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Calcul parallèle des chemins"):
            result = future.result()
            if result is not None:
                all_paths.append(result)
    return all_paths


def apply_morpho_opening_3d(binary_volume, radius=2, iterations=1):
    """
    Applique une ouverture morphologique (érosion + dilatation) pour casser les connexions fines (tubules).
    
    Parameters:
        binary_volume : ndarray
            Volume binaire (uint8 ou bool)
        radius : int
            Rayon de la structuration sphérique
        iterations : int
            Nombre de fois que l'érosion et dilatation sont appliquées
    
    Returns:
        ndarray : volume après ouverture
    """
    struct = ball(radius)

    eroded = binary_volume.copy().astype(bool)
    for _ in range(iterations):
        eroded = binary_erosion(eroded, structure=struct)

    opened = eroded
    #for _ in range(iterations):
    #opened = binary_dilation(opened, structure=struct)

    return (opened.astype(np.uint8) * 255)

def extract_local_volume(volume, center, size=250):
    """Extrait un sous-volume cubique autour de la graine"""
    x, y, z = center
    half = size // 2
    zmin, zmax = max(0, z - half), min(volume.shape[0], z + half)
    ymin, ymax = max(0, y - half), min(volume.shape[1], y + half)
    xmin, xmax = max(0, x - half), min(volume.shape[2], x + half)

    subvol = volume[zmin:zmax, ymin:ymax, xmin:xmax]
    offset = (zmin, ymin, xmin)
    return subvol, offset

def keep_largest_component_per_slice(volume):
    """
    Conserve uniquement la plus grande composante connexe dans chaque slice 2D.
    """
    result = np.zeros_like(volume, dtype=np.uint8)
    for z in range(volume.shape[0]):
        slice_ = volume[z] > 0  # booléen plus rapide à traiter
        if not slice_.any():
            continue  # Pas de composantes sur cette slice

        labeled = label(slice_, connectivity=1)  
        if labeled.max() == 0:
            continue

        # Trouver la plus grande composante
        regions = regionprops(labeled)
        largest_region = max(regions, key=lambda r: r.area)
        mask = (labeled == largest_region.label)
        result[z][mask] = 255

    return result

def segment_structure(volume, seed_point, output_path="structure_segmentee_global.tif", box_size=250):
    # --- Extraire un sous-volume autour de la graine ---
    subvol, offset = extract_local_volume(volume, seed_point, size=box_size)
    # --- Binarisation Otsu ---
    thresh = threshold_otsu(subvol)
    print(f"[i] Seuil Otsu : {thresh}")
    binary = (subvol > thresh).astype(np.uint8) * 255
    #tiff.imwrite("step1_binary_threshold.tif", binary)

    # --- Morphologie ---
    binary_eroded = apply_morpho_opening_3d(binary, radius=2, iterations=4)
    #tiff.imwrite("step2_opening_applied.tif", binary_eroded)

    # --- Filtrage composante principale par slice ---
    filtered = keep_largest_component_per_slice(binary_eroded)

    # --- Créer un masque de même taille que le volume d'origine ---
    full_mask = np.zeros_like(volume, dtype=np.uint8)

    z0, y0, x0 = offset
    z1, y1, x1 = z0 + filtered.shape[0], y0 + filtered.shape[1], x0 + filtered.shape[2]
    full_mask[z0:z1, y0:y1, x0:x1] = filtered

    #tiff.imwrite(output_path, full_mask)
    #print(f"[✓] Masque segmenté inséré dans le volume global : {output_path}")

    return full_mask


def get_extremities_from_volume(volume, zmax, z_window, min_size, low_corner, high_corner):
    """
    Trouve les extrémités des composantes dans un volume numpy, en filtrant
    sur zmax, taille min, et coordonnées XY dans un rectangle donné.
    
    Parameters:
        volume : ndarray (3D)
            Volume binaire (uint8 ou bool) déjà chargé en mémoire.
        zmax : int
            Limite supérieure sur l’axe Z à considérer.
        z_window : int
            Fenêtre de recherche sur Z (entre zmax-z_window et zmax).
        min_size : int
            Taille minimale d’une composante pour être conservée.
        low_corner : tuple (x, y)
            Coin bas (X, Y) du rectangle de filtrage.
        high_corner : tuple (x, y)
            Coin haut (X, Y) du rectangle de filtrage.
    
    Returns:
        z_min_points : list of tuples (z, y, x)
            Points minimum en Z des composantes filtrées.
        discarded_mask : ndarray
            Masque des voxels rejetés.
    """
    volume_cropped = volume[:zmax]
    print(f"Sous-volume utilisé : {volume_cropped.shape}")

    if volume_cropped.dtype != np.uint8:
        volume_cropped = (volume_cropped >= 0.5).astype(np.uint8)

    labels, num = label(volume_cropped, connectivity=3, return_num=True)
    print(f"{num} composantes détectées dans le sous-volume")

    discarded_mask = np.zeros_like(volume_cropped, dtype=np.uint8)

    z_start = zmax - z_window
    z_end = zmax

    print("Filtrage des composantes...")

    z_min_points = []
    filtered_count = 0
    discarded_count = 0

    for region in regionprops(labels):
        coords = region.coords
        # Vérification : tous les voxels doivent être dans le foreground
        for c in coords:
            z, y, x = c
            if volume_cropped[z, y, x] != 1:
                raise ValueError(f"Voxel {c} hors foreground dans la région {region.label}")
        z_coords = coords[:, 0]

        if region.area < min_size:
            discarded_mask[tuple(coords.T)] = 1
            discarded_count += 1
            continue

        z_max_region = np.max(z_coords)
        zm, ym, xm = coords[z_coords == z_max_region][0]

        if z_start <= z_max_region <= z_end and low_corner[0] < xm < high_corner[0] and high_corner[1] < ym < low_corner[1]:
            z_min = np.min(z_coords)
            z_min_point = coords[z_coords == z_min][0]
            z_min_points.append(tuple(z_min_point))
            filtered_count += 1
        else:
            discarded_mask[tuple(coords.T)] = 1
            discarded_count += 1

    print(f"Nombre final de composantes retenues : {filtered_count}")
    print(f"Nombre de composantes rejetées : {discarded_count}")
    z_min_points = sorted(z_min_points, key=lambda coord: coord[0])
    print("Liste des points (z_min, y, x) :", z_min_points)

    gc.collect()

    return z_min_points, discarded_mask


# --- Génération des directions 3D ---
directions = [(dz, dy, dx) for dz in [ 0,1]
                          for dy in [-1, 0, 1]
                          for dx in [-1, 0, 1]
                          if not (dz == dy == dx == 0)]

def get_connectivity_weight(dz, dy, dx):
    non_zero = sum(1 for d in (dz, dy, dx) if d != 0)
    if non_zero == 1:
        return 1.0         # 6-connectivité
    elif non_zero == 2:
        return math.sqrt(2)  # 18-connectivité
    else:
        return math.sqrt(3)  # 26-connectivité


def proba_aware_heuristic_euclidean(a, b, mean_proba=0.85, base_weight=1.0):
    """
    Heuristique pondérée par une probabilité moyenne, utilisant la distance euclidienne.
    
    Paramètres :
        a (tuple or array): Coordonnées du point A.
        b (tuple or array): Coordonnées du point B.
        mean_proba (float): Probabilité moyenne (entre 0 et 1).
        base_weight (float): Poids de base appliqué à la distance.
    
    Retourne :
        float: Valeur heuristique.
    """
    d = np.linalg.norm(np.array(a) - np.array(b))  # distance euclidienne
    return d * base_weight * (1 - mean_proba)


def astar_to_border_with_map(volume, start, border_mask, goal_point, heuristic_weight=0.3):
    shape = volume.shape
    mean_proba = np.mean(volume)  # ou fixe comme 0.85 si tu préfères

    distances = np.full(shape, np.inf, dtype=np.float32)
    came_from = {}
    visited = np.zeros(shape, dtype=bool)

    heap = []
    distances[start] = 0.0

    h = proba_aware_heuristic_euclidean(goal_point, start, mean_proba)

    heapq.heappush(heap, (h * heuristic_weight, 0.0, start))  # (f = g + h, g, node)

    while heap:
        f_cost, g_cost, current = heapq.heappop(heap)
        z, y, x = current

        if visited[z, y, x]:
            continue
        visited[z, y, x] = True

        if border_mask[z, y, x]:
            print(f"[✓] Bord atteint à {current} avec coût {g_cost:.2f} en partant de {start}")
            return current, came_from

        for dz, dy, dx in directions:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                if visited[nz, ny, nx]:
                    continue

                weight = get_connectivity_weight(dz, dy, dx)
                proba = volume[nz, ny, nx]
                move_cost = weight * (1 - proba)
                new_g = g_cost + move_cost

                if new_g < distances[nz, ny, nx]:
                    distances[nz, ny, nx] = new_g
                    came_from[(nz, ny, nx)] = (z, y, x)

                    h = proba_aware_heuristic_euclidean(goal_point, (nz, ny, nx), mean_proba)

                    heapq.heappush(heap, (new_g + heuristic_weight * h, new_g, (nz, ny, nx)))

    print("[!] Aucun bord atteint.")
    return None, came_from


def reconstruct_path(came_from, end, start):
    path = []
    current = end
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []
    path.append(start)
    path.reverse()
    return path

def export_paths_colored_mask(shape, paths, output_path="chemins_colores.tif"):
    """
    Crée un volume couleur (RGB) avec chaque chemin dans une couleur différente.
    """
    color_mask = np.zeros(shape + (3,), dtype=np.uint8)

    # Générer des couleurs aléatoires pour chaque chemin
    colors = []
    for _ in paths:
        color = tuple(random.choices(range(50, 256), k=3))  # Évite trop sombre
        colors.append(color)

    for color, (_, _, path) in zip(colors, paths):
        for (z, y, x) in path:
            color_mask[z, y, x] = color

    #tiff.imwrite(output_path, color_mask)
    print(f"[✓] Chemins colorés exportés dans : {output_path}")

def get_border_center(border_mask):
    coords = np.argwhere(border_mask)
    center = coords.mean(axis=0)
    return tuple(center)

def run_tracking_pipeline(
    input_path_volume,
    seed_point,
    low_corner,
    high_corner,
    output_dir=".",
    zmax=2000,
    z_window=250,
    min_size=1000,
    box_size=250,
    max_workers=4,
    probas_volume=None,
    segmented=None
):
    print("[i] Début du pipeline de tracking...")

    # Si on n’a pas les volumes en mémoire, on lit depuis les fichiers
    if segmented is None or probas_volume is None:
        raise ValueError("Il faut passer 'probas_volume' ET 'segmented' directement, pas les chemins.")
    print(f"zmax avant get extremitis {zmax}")
    # start_points et discarded_mask sont extraits depuis segmented
    start_points, discarded_mask = get_extremities_from_volume(segmented, zmax, z_window, min_size, low_corner, high_corner)

    # volume original chargé depuis input_path_volume (si besoin)
    with tiff.TiffFile(input_path_volume) as tif:
        volume = tif.asarray(out='memmap')


    shape = volume.shape
    segmented_mask = segment_structure(volume,seed_point)
    border_mask = segmented_mask - binary_erosion(segmented_mask.astype(bool)).astype(np.uint8) * 255
    border_mask = border_mask.astype(bool)
    

    # découpage des probas volume si besoin
    volume_proba_red = probas_volume[:zmax+144]

    save_arrays(volume_proba_red, border_mask)

    del probas_volume
    del volume
    del segmented
    gc.collect()
    center = get_border_center(border_mask)
    all_paths = dijkstra_parallel(start_points,center, max_workers=max_workers)

    #export_paths_colored_mask(shape, all_paths, os.path.join(output_dir, "chemins_colores.tif"))
    #tiff.imwrite(os.path.join(output_dir, "composantes_non_utilisées.tif"), discarded_mask.astype(np.uint8))


    print("[✓] Pipeline de tracking terminé.")

    return all_paths


