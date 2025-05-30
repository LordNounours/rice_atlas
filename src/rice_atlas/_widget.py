from typing import TYPE_CHECKING
import numpy as np
from magicgui import magic_factory
from qtpy.QtWidgets import (
    QPushButton, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, QSpinBox, QLabel
)
from rice_atlas.predictor import segment_volume
from rice_atlas.tracking import run_tracking_pipeline
from tifffile import imwrite,imread
import multiprocessing
import random 
import csv
from pathlib import Path
from scipy.ndimage import center_of_mass,binary_erosion

if TYPE_CHECKING:
    import napari

# Dictionnaire global pour stocker la référence des boutons/widgets
save_button_ref = {}

@magic_factory(
    model_path={"widget_type": "FileEdit", "label": "Chemin du modèle", "mode": "r"},
    volume_path={"widget_type": "FileEdit", "label": "Volume à segmenter", "mode": "r"},
    output_path={"widget_type": "FileEdit", "label": "Fichier de sortie (optionnel)", "nullable": True, "mode": "w"},
    patch_size={"label": "Taille du patch", "min": 16, "max": 256, "step": 16},
    stride={"label": "Stride", "min": 8, "max": 256, "step": 8},
    batch_size={"label": "Taille du batch", "min": 1, "max": 64, "step": 1},
    tap_x={"label": "Centre X", "min": 0, "max": 1200, "step": 1},
    tap_y={"label": "Centre Y", "min": 0, "max": 1200, "step": 1},
    tap_z={"label": "Centre Z", "min": 0, "max": 18000, "step": 1},
    pretreatment={"label": "Prétraitement", "widget_type": "CheckBox", "value": False},
)
def segment_volume_widget(
    model_path: str,
    volume_path: str,
    output_path: str = None,
    patch_size: int = 128,
    stride: int = 96,
    batch_size: int = 16,
    pretreatment: bool = False,
    tap_x: int = 0,
    tap_y: int = 0,
    tap_z: int = 0,
    viewer: "napari.viewer.Viewer" = None,
) -> None:
    tap_center = (tap_x, tap_y, tap_z)
    print(f"📍 Centre du plateau sélectionné : {tap_center}")

    z_max=tap_z+100
    print(f"z max apres recup tap center : {z_max}" )

    # 🔍 Lancer la segmentation
    probas_volume, segmented = segment_volume(
        model_path=model_path,
        volume_path=volume_path,
        output_path=output_path,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        pretreatment=pretreatment,
        tap_center=tap_center,
    )

    if viewer is not None:
        viewer.add_image(probas_volume, name="Probabilités classe 1", colormap="gray")
        viewer.add_labels(segmented, name="Segmentation")

    # 💾 Fonctions de sauvegarde
    def save_probas():
        save_path, _ = QFileDialog.getSaveFileName(
            caption="Enregistrer les probabilités",
            filter="Fichiers TIFF (*.tiff *.tif)",
        )
        if save_path:
            imwrite(save_path, probas_volume.astype(np.float32))
            print(f"✅ Probabilités sauvegardées à : {save_path}")

    def save_segmented():
        save_path, _ = QFileDialog.getSaveFileName(
            caption="Enregistrer la segmentation binaire",
            filter="Fichiers TIFF (*.tiff *.tif)",
        )
        if save_path:
            imwrite(save_path, segmented.astype(np.uint8))
            print(f"✅ Segmentation binaire sauvegardée à : {save_path}")

    # ♻️ Supprimer anciens boutons si existants
    for btn_key in ["save_button_proba", "save_button_segmented", "save_button_tracking", "corners_container", "btn_run_tracking"]:
        if btn_key in save_button_ref:
            widget = save_button_ref[btn_key]
            if hasattr(widget, "deleteLater"):
                widget.deleteLater()
            else:
                viewer.window.remove_dock_widget(widget)
            save_button_ref.pop(btn_key)

    # 🧱 Créer un widget conteneur (vertical layout)
    button_container = QWidget()
    layout = QVBoxLayout()
    button_container.setLayout(layout)

    # ➕ Bouton probas
    button_proba = QPushButton("Sauvegarder les probabilités")
    button_proba.clicked.connect(save_probas)
    layout.addWidget(button_proba)
    save_button_ref["save_button_proba"] = button_proba

    # ➕ Bouton segmentation
    button_seg = QPushButton("Sauvegarder la segmentation binaire")
    button_seg.clicked.connect(save_segmented)
    layout.addWidget(button_seg)
    save_button_ref["save_button_segmented"] = button_seg

    # Création conteneur pour coins
    corners_container = QWidget()
    corners_layout = QVBoxLayout()
    corners_container.setLayout(corners_layout)

    # Layout pour coin bas gauche
    low_corner_row = QHBoxLayout()
    low_label = QLabel("Coin bas gauche :")
    low_inputs = QHBoxLayout()
    low_corner_x = QSpinBox()
    low_corner_x.setRange(0, 1200)
    low_corner_y = QSpinBox()
    low_corner_y.setRange(0, 1200)
    low_inputs.addWidget(QLabel("X"))
    low_inputs.addWidget(low_corner_x)
    low_inputs.addWidget(QLabel("Y"))
    low_inputs.addWidget(low_corner_y)
    low_corner_row.addWidget(low_label)
    low_corner_row.addLayout(low_inputs)
    corners_layout.addLayout(low_corner_row)

    # Layout pour coin haut droit
    high_corner_row = QHBoxLayout()
    high_label = QLabel("Coin haut droit :")
    high_inputs = QHBoxLayout()
    high_corner_x = QSpinBox()
    high_corner_x.setRange(0, 1200)
    high_corner_y = QSpinBox()
    high_corner_y.setRange(0, 1200)
    high_inputs.addWidget(QLabel("X"))
    high_inputs.addWidget(high_corner_x)
    high_inputs.addWidget(QLabel("Y"))
    high_inputs.addWidget(high_corner_y)
    high_corner_row.addWidget(high_label)
    high_corner_row.addLayout(high_inputs)
    corners_layout.addLayout(high_corner_row)


    # Cette fonction sera appelée après segmentation pour afficher les coins et bouton tracking
    def after_segmentation():
        layout.addWidget(corners_container)
        corners_container.show()
        save_button_ref["corners_container"] = corners_container
        def recentre_point_local_volume(volume, z, y, x, win_size=64, threshold=10):
            half = win_size // 2

            # Clamp fenêtre dans le plan 2D (y,x) à la tranche z
            y_min = max(0, y - half)
            y_max = min(volume.shape[1], y + half)
            x_min = max(0, x - half)
            x_max = min(volume.shape[2], x + half)

            patch = volume[z, y_min:y_max, x_min:x_max]

            mask = patch > threshold
            if np.sum(mask) == 0:
                # Pas de structure détectée, on ne décale pas
                return y, x

            com_y_patch, com_x_patch = center_of_mass(patch * mask)

            # Centre de masse dans les coordonnées globales (volume)
            com_y = y_min + com_y_patch
            com_x = x_min + com_x_patch

            # Nouveau point recalé (borné dans les dimensions)
            new_y = int(round(com_y))
            new_x = int(round(com_x))

            new_y = max(0, min(volume.shape[1] - 1, new_y))
            new_x = max(0, min(volume.shape[2] - 1, new_x))

            return new_y, new_x
        
        
        def run_tracking_with_corners():
            low_corner = (low_corner_x.value(), low_corner_y.value())
            high_corner = (high_corner_x.value(), high_corner_y.value())
            print(f"Utilisation des coins : low {low_corner}, high {high_corner}")
            print("🚀 Lancement du tracking...")
            print(f"Zmax avant run tracking pipeline {z_max}")
            all_paths = run_tracking_pipeline(volume_path, tap_center, low_corner, high_corner, zmax=z_max,
                                            probas_volume=probas_volume, segmented=segmented)
            print("✅ Tracking terminé.")
            volume = imread(volume_path)  # On charge ici pour éviter de le recharger plus tard
            slice_size = 16
            all_paths_recal = []
            n_iter = 8  # Nombre d’itérations de recentrage

            for seed, score, path in all_paths:
                new_path = []
                for z, y, x in path:
                    for _ in range(n_iter):
                        y, x = recentre_point_local_volume(volume, z, y, x, win_size=slice_size, threshold=10)
                    new_path.append((z, y, x))
                all_paths_recal.append((seed, score, new_path))

            print("🧠 Chemins recalés")


            # Conversion paths -> volume labelisé RGB pour napari
            shape = probas_volume.shape
            color_mask = np.zeros(shape + (3,), dtype=np.uint8)
            for idx, (_, _, path) in enumerate(all_paths):
                color = tuple(random.choices(range(50, 256), k=3))
                for (z, y, x) in path:
                    color_mask[z, y, x] = color

            # Ajouter dans napari
            viewer.add_image(color_mask, name="Chemins colorés", rgb=True)

            shape = probas_volume.shape
            color_mask = np.zeros(shape + (3,), dtype=np.uint8)
            for idx, (_, _, path) in enumerate(all_paths_recal):
                color = tuple(random.choices(range(50, 256), k=3))
                for (z, y, x) in path:
                    color_mask[z, y, x] = color

            # Ajouter dans napari
            viewer.add_image(color_mask, name="Chemins colorés_recalés", rgb=True)

            def save_colored_paths():
                save_path, _ = QFileDialog.getSaveFileName(
                    caption="Enregistrer les chemins colorés",
                    filter="Fichiers TIFF (*.tiff *.tif)",
                )
                if save_path:
                    imwrite(save_path, color_mask.astype(np.uint8))
                    print(f"✅ Chemins colorés sauvegardés à : {save_path}")
            def save_paths_to_csv():
                dir_path = QFileDialog.getExistingDirectory(
                    caption="Choisir un dossier pour enregistrer les chemins CSV"
                )
                if dir_path:
                    from pathlib import Path
                    import csv

                    for idx, (_, _, path) in enumerate(all_paths):
                        csv_path = Path(dir_path) / f"{idx}.csv"
                        with open(csv_path, mode="w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["X", "Y", "Z"])
                            for z, y, x in path:
                                writer.writerow([x, y, z])  # X, Y, Z

                    print(f"✅ {len(all_paths)} chemins enregistrés dans {dir_path}")

            def extract_root_slices():
                dir_path = QFileDialog.getExistingDirectory(
                    caption="Choisir un dossier pour enregistrer les slices racines"
                )
                if not dir_path:
                    return
                volume = imread(volume_path)  # ✅ On lit bien un TIFF ici

                slice_size = 64  # taille des slices extraites (64x64)
                half = slice_size // 2

                for idx, (_, _, path) in enumerate(all_paths):
                    root_dir = Path(dir_path) / f"root_{idx}"
                    root_dir.mkdir(parents=True, exist_ok=True)

                    for z, y, x in path:
                        if (
                            y - half < 0 or y + half >= volume.shape[1]
                            or x - half < 0 or x + half >= volume.shape[2]
                            or z < 0 or z >= volume.shape[0]
                        ):
                            continue

                        slice_ = volume[z, y - half : y + half, x - half : x + half]
                        imwrite(str(root_dir / f"{z}.tif"), slice_.astype(volume.dtype))

                print(f"✅ Slices extraites dans : {dir_path}")

            
            btn_save_colored = QPushButton("Sauvegarder chemins colorés")
            btn_save_colored.clicked.connect(save_colored_paths)
            layout.addWidget(btn_save_colored)
            save_button_ref["save_button_tracking"] = btn_save_colored

            btn_save_csv = QPushButton("Sauvegarder chemins en CSV")
            btn_save_csv.clicked.connect(save_paths_to_csv)
            layout.addWidget(btn_save_csv)
            save_button_ref["save_button_csv"] = btn_save_csv

            btn_extract_slices = QPushButton("Extraire slices racines")
            btn_extract_slices.clicked.connect(extract_root_slices)
            layout.addWidget(btn_extract_slices)
            save_button_ref["btn_extract_slices"] = btn_extract_slices

            




        btn_run_tracking = QPushButton("Lancer tracking avec coins")
        btn_run_tracking.clicked.connect(run_tracking_with_corners)
        layout.addWidget(btn_run_tracking)
        save_button_ref["btn_run_tracking"] = btn_run_tracking

    after_segmentation()

    # ➕ Ajouter le widget conteneur à l’UI napari
    viewer.window.add_dock_widget(button_container, name="📥 Exporter les résultats")


#multiprocessing.freeze_support()
