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
import cc3d
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
    volume_cropped = volume[:zmax]
    print(f"Sous-volume utilisé : {volume_cropped.shape}")
    

    """if volume_cropped.dtype != np.uint8:
        volume_cropped = (volume_cropped >= 0.5).astype(np.uint8)"""
    labels = cc3d.connected_components(volume_cropped, connectivity=26)
    num = labels.max()
    #labels, num = label(volume_cropped, connectivity=3, return_num=True)
    print(f"{num} composantes détectées dans le sous-volume")

    z_start = zmax - z_window
    z_end = zmax

    

    z_min_points = []
    filtered_count = 0
    discarded_count = 0

    for region in regionprops(labels,cache=False):
        if region.area < min_size:
            discarded_count += 1
            continue

        # Utiliser bbox pour éviter coords
        z0, y0, x0, z1, y1, x1 = region.bbox
        z_max_region = z1 - 1  # Max inclusif

        # Trouver un point au z_max (plus rapide qu’analyser tous coords)
        slice_z = labels[z_max_region, y0:y1, x0:x1]
        mask_at_zmax = slice_z == region.label
        if not np.any(mask_at_zmax):
            continue
        indices = np.argwhere(mask_at_zmax)
        ym, xm = indices[0] + [y0, x0]

        # Vérifier fenêtre Z et rectangle XY
        if z_start <= z_max_region <= z_end and low_corner[0] < xm < high_corner[0] and high_corner[1] < ym < low_corner[1]:
            # Trouver point à z_min
            z_min = z0
            slice_zmin = labels[z_min, y0:y1, x0:x1]
            mask_at_zmin = slice_zmin == region.label
            indices_min = np.argwhere(mask_at_zmin)
            if indices_min.size > 0:
                ymin, xmin = indices_min[0] + [y0, x0]
                z_min_points.append((z_min, ymin, xmin))
                filtered_count += 1

    print(f"Nombre final de composantes retenues : {filtered_count}")
    print(f"Nombre de composantes rejetées : {discarded_count}")
    z_min_points = sorted(z_min_points, key=lambda coord: coord[0])
    print("Liste des points (z_min, y, x) :", z_min_points)

    gc.collect()
    return z_min_points

import gc
import numpy as np
import cc3d
from skimage.measure import regionprops

def get_extremities_from_volume_leaf(volume, zmin, volume_z_max, z_window, min_size, low_corner, high_corner):
    # On découpe le volume à partir de zmin
    volume_cropped = volume[zmin:]
    print(f"Sous-volume utilisé : {volume_cropped.shape}")
    
    # Détection des composantes connexes (dans le repère local croppé)
    labels = cc3d.connected_components(volume_cropped, connectivity=26)
    num = labels.max()
    print(f"{num} composantes détectées dans le sous-volume")

    # Bornes en Z (globales)
    z_start = zmin + z_window
    z_end = volume_z_max
    print(f"z start : {z_start}")
    print(f"z end : {z_end}")

    z_min_points = []
    filtered_count = 0
    discarded_count = 0

    # Parcours des régions détectées
    for region in regionprops(labels, cache=False):
        # Filtrage par taille si nécessaire
        # if region.area < min_size:
        #     discarded_count += 1
        #     continue

        # Bounding box en indices locaux (dans labels)
        z0_local, y0, x0, z1_local, y1, x1 = region.bbox
        z_max_local = z1_local - 1
        z_min_local = z0_local

        # Conversion en indices globaux
        z_max_global = z_max_local + zmin
        z_min_global = z_min_local + zmin

        # Slice au z_max local
        slice_z = labels[z_max_local, y0:y1, x0:x1]
        mask_at_zmax = slice_z == region.label
        if not np.any(mask_at_zmax):
            continue

        indices = np.argwhere(mask_at_zmax)
        ym, xm = indices[0] + [y0, x0]
        print(z_max_global, ym, xm)

        # Vérification des critères de filtrage
        if z_start <= z_max_global <= z_end and low_corner[0] < xm < high_corner[0] and high_corner[1] < ym < low_corner[1]:
            # Slice au z_min local
            slice_zmin = labels[z_min_local, y0:y1, x0:x1]
            mask_at_zmin = slice_zmin == region.label
            indices_min = np.argwhere(mask_at_zmin)
            if indices_min.size > 0:
                ymin, xmin = indices_min[0] + [y0, x0]

                # On stocke le point avec coordonnées globales
                z_min_points.append((z_min_global, ymin, xmin))
                filtered_count += 1

    print(f"Nombre final de composantes retenues : {filtered_count}")
    print(f"Nombre de composantes rejetées : {discarded_count}")

    # Tri par profondeur
    z_min_points = sorted(z_min_points, key=lambda coord: coord[0])
    print("Liste des points (z_min, y, x) :", z_min_points)

    gc.collect()
    return z_min_points




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
    mean_proba = np.mean(volume)  

    distances = {start: 0.0}
    came_from = {}
    visited = set()

    heap = []
    h = proba_aware_heuristic_euclidean(goal_point, start, mean_proba)
    heapq.heappush(heap, (h * heuristic_weight, 0.0, start))  # (f = g + h, g, node)

    while heap:
        f_cost, g_cost, current = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)

        z, y, x = current
        if border_mask[z, y, x]:
            print(f"[✓] Bord atteint à {current} avec coût {g_cost:.2f} en partant de {start}")
            return current, came_from

        for dz, dy, dx in directions:
            nz, ny, nx = z + dz, y + dy, x + dx
            if not (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]):
                continue
            neighbor = (nz, ny, nx)
            if neighbor in visited:
                continue

            weight = get_connectivity_weight(dz, dy, dx)
            proba = volume[nz, ny, nx]
            move_cost = weight * (1 - proba)
            new_g = g_cost + move_cost

            if new_g < distances.get(neighbor, np.inf):
                distances[neighbor] = new_g
                came_from[neighbor] = current
                h = proba_aware_heuristic_euclidean(goal_point, neighbor, mean_proba)
                heapq.heappush(heap, (new_g + heuristic_weight * h, new_g, neighbor))

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
    volume,
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
    start_points = get_extremities_from_volume(segmented, zmax, z_window, min_size, low_corner, high_corner)
    segmented_mask = segment_structure(volume,seed_point)
    border_mask = segmented_mask - binary_erosion(segmented_mask.astype(bool)).astype(np.uint8) * 255
    border_mask = border_mask.astype(bool)
    

    # découpage des probas volume si besoin
    volume_proba_red = probas_volume[:zmax+100]
    
    save_arrays(volume_proba_red, border_mask)

    del probas_volume
    del volume
    del segmented
    gc.collect()
    center = get_border_center(border_mask)
    all_paths = dijkstra_parallel(start_points,center, max_workers=max_workers)


    print("[✓] Pipeline de tracking terminé.")

    return all_paths


def run_tracking_pipeline_leaf(
    volume,
    seed_point,
    low_corner,
    high_corner,
    output_dir=".",
    zmin=2000,
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
    print(f"zmax avant get extremitis {zmin}")
    # start_points et discarded_mask sont extraits depuis segmented
    start_points = get_extremities_from_volume_leaf(segmented, zmin,volume.shape[0], z_window, zmin, low_corner, high_corner)
    segmented_mask = segment_structure(volume,seed_point)
    border_mask = segmented_mask - binary_erosion(segmented_mask.astype(bool)).astype(np.uint8) * 255
    border_mask = border_mask.astype(bool)
    
    proba_class1 = probas_volume[1]
    proba_class2 = probas_volume[2]
    combined_proba = np.maximum(proba_class1, proba_class2)

    # découpage des probas volume si besoin
    volume_proba_red = combined_proba[zmin-100:]

    save_arrays(volume_proba_red, border_mask)

    del probas_volume
    del volume
    del segmented
    gc.collect()
    center = get_border_center(border_mask)
    all_paths = dijkstra_parallel(start_points,center, max_workers=max_workers)


    print("[✓] Pipeline de tracking terminé.")

    return all_paths


