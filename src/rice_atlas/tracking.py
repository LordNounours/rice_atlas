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

VOLUME_PATH = "/tmp/volume_proba_red.npy"
MASK_PATH = "/tmp/border_mask.npy"

def save_arrays(volume_proba_red, border_mask):
    np.save(VOLUME_PATH, volume_proba_red)
    np.save(MASK_PATH, border_mask)

def load_memmap_arrays():
    volume = np.load(VOLUME_PATH, mmap_mode='r')
    mask = np.load(MASK_PATH, mmap_mode='r')
    return volume, mask

def path_for_one_root(start):
    print(f"Start : {start}")
    volume, border_mask = load_memmap_arrays()
    center = get_border_center(border_mask)
    end, came_from = astar_to_border_with_map(volume, start, border_mask, center)
    if end is not None:
        path = reconstruct_path(came_from, end, start)
        print(f"Longueur de chemin : {len(path)}")
        return (start, end, path)
    return None


def dijkstra_parallel(starts, max_workers=4):
    all_paths = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(path_for_one_root, start) for start in starts]

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

def extract_local_volume(volume, center, size=500):
    """Extrait un sous-volume cubique autour de la graine"""
    z, y, x = center
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

def segment_structure(volume, seed_point, output_path="structure_segmentee_global.tif", box_size=500):
    # --- Extraire un sous-volume autour de la graine ---
    subvol, offset = extract_local_volume(volume, seed_point, size=box_size)

    # --- Binarisation Otsu ---
    thresh = threshold_otsu(subvol)
    print(f"[i] Seuil Otsu : {thresh}")
    binary = (subvol > thresh).astype(np.uint8) * 255
    tiff.imwrite("step1_binary_threshold.tif", binary)

    # --- Morphologie ---
    binary_eroded = apply_morpho_opening_3d(binary, radius=2, iterations=4)
    tiff.imwrite("step2_opening_applied.tif", binary_eroded)

    # --- Filtrage composante principale par slice ---
    filtered = keep_largest_component_per_slice(binary_eroded)

    # --- Créer un masque de même taille que le volume d'origine ---
    full_mask = np.zeros_like(volume, dtype=np.uint8)

    z0, y0, x0 = offset
    z1, y1, x1 = z0 + filtered.shape[0], y0 + filtered.shape[1], x0 + filtered.shape[2]
    full_mask[z0:z1, y0:y1, x0:x1] = filtered

    tiff.imwrite(output_path, full_mask)
    print(f"[✓] Masque segmenté inséré dans le volume global : {output_path}")

    return full_mask


def get_extremities(input_tiff, zmax, z_window, min_size, low_corner, high_corner):
    with tiff.TiffFile(input_tiff) as tif:
        volume = tif.asarray(out='memmap')
        print(f"Volume chargé : {volume.shape}")

    volume_cropped = volume[:zmax+1]
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
        for c in coords:
            z, y, x = c
            if volume_cropped[z, y, x] != 1:
                raise ValueError(f"Voxel {c} hors foreground dans la région {region.label}")
        z_coords = coords[:, 0]

        if region.area < min_size:
            discarded_mask[tuple(coords.T)] = 1
            discarded_count += 1
            continue

        z_max = np.max(z_coords)
        zm, ym, xm = coords[z_coords == z_max][0]

        if z_start <= z_max <= z_end and low_corner[0] < xm < high_corner[0] and high_corner[1] < ym < low_corner[1]:
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

    del volume
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


def export_path_mask(shape, path, output_path="chemin_mask.tif"):
    mask = np.zeros(shape, dtype=np.uint8)
    for z, y, x in path:
        mask[z, y, x] = 255
    tiff.imwrite(output_path, mask)
    print(f"[✓] Masque binaire du chemin exporté : {output_path}")


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

def compute_h_max(shape, goal_point):
    z_size, y_size, x_size = shape
    corners = [
        (0, 0, 0),
        (0, 0, x_size - 1),
        (0, y_size - 1, 0),
        (0, y_size - 1, x_size - 1),
        (z_size - 1, 0, 0),
        (z_size - 1, 0, x_size - 1),
        (z_size - 1, y_size - 1, 0),
        (z_size - 1, y_size - 1, x_size - 1),
    ]
    return max(manhattan(corner, goal_point) for corner in corners)

def proba_aware_heuristic(a, b, mean_proba=0.85, base_weight=1.0):
    d = manhattan(a, b)
    return d * base_weight * (1 - mean_proba)

def astar_to_border_with_map(volume, start, border_mask, goal_point, heuristic_weight=0.45):
    shape = volume.shape
    mean_proba = np.mean(volume)  # ou fixe comme 0.85 si tu préfères

    distances = np.full(shape, np.inf, dtype=np.float32)
    came_from = {}
    visited = np.zeros(shape, dtype=bool)

    heap = []
    distances[start] = 0.0

    h = proba_aware_heuristic(goal_point, start, mean_proba)

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

                    h = proba_aware_heuristic(goal_point, (nz, ny, nx), mean_proba)

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

def dijkstra_to_border_multiple_starts(volume, starts, border_mask,center):
    """
    Trouver un chemin du start point vers border_mask pour plusieurs points start.
    Retourne une liste de (start, end, path).
    """
    all_paths = []
    for start in tqdm(starts, desc="Calcul des chemins"):
        print(f"Start {start}")
        end, came_from = astar_to_border_with_map(volume, start, border_mask,center)
        if end is not None:
            path = reconstruct_path(came_from, end, start)
            all_paths.append((start, end, path))
        else:
            print(f"[!] Aucun chemin pour le point de départ {start}")
    return all_paths


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

    tiff.imwrite(output_path, color_mask)
    print(f"[✓] Chemins colorés exportés dans : {output_path}")

def get_border_center(border_mask):
    coords = np.argwhere(border_mask)
    center = coords.mean(axis=0)
    return tuple(center)

def compute_distance_map_to_center(shape, center):
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = center
    dist_map = np.abs(zz - cz) + np.abs(yy - cy) + np.abs(xx - cx)
    return dist_map.astype(np.uint16)



# --- MAIN ---
if __name__ == "__main__":
    
    input_path_seg = "/home/thomas/Bureau/Code/seg_volume_linked.tiff"
    low_corner=(505,739)
    high_corner=(930,220)
    start_points,discarded_mask = get_extremities(input_path_seg,2000,250,1000,low_corner,high_corner)
    #start_points=start_points[:10]
    
    input_path_volume = "/home/thomas/Bureau/01_full_plant_v2_SlicesY-1_FULL_8bit2_sandless_smooth.tif"
    with tiff.TiffFile(input_path_volume) as tif:
        volume = tif.asarray(out='memmap')

    shape = volume.shape
    segmented_mask = segment_structure(volume,(1956, 481, 686))
    border_mask = segmented_mask - binary_erosion(segmented_mask.astype(bool)).astype(np.uint8) * 255
    border_mask = border_mask.astype(bool)

    
    center = get_border_center(border_mask)
    
    volume_proba = tiff.imread("/home/thomas/Bureau/Code/heatmap_volume2.tiff")
    volume_proba_red=volume_proba[:2100]
    save_arrays(volume_proba_red, border_mask)

    del volume_proba
    gc.collect()

    del volume,segmented_mask
    gc.collect()
    all_paths = dijkstra_parallel(start_points, max_workers=4)



    export_paths_colored_mask(shape, all_paths, "chemins_colores2.tif")

    tiff.imwrite("composantes_non_utilisées2.tif", discarded_mask.astype(np.uint8))
    print("[✓] Fichier TIFF écrit : composantes_non_utilisées2.tif")
    if os.path.exists(VOLUME_PATH):
        
        print(f"[✓] Fichier supprimé : {VOLUME_PATH}")

    if os.path.exists(MASK_PATH):
        os.remove(MASK_PATH)
        print(f"[✓] Fichier supprimé : {MASK_PATH}")

