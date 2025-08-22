from typing import TYPE_CHECKING
import numpy as np
from magicgui import magic_factory
from qtpy.QtWidgets import (
    QPushButton, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, QSpinBox, QLabel,QLineEdit
)
from rice_atlas.predictor import segment_volume_root,segment_volume_leaf
from rice_atlas.tracking import run_tracking_pipeline,run_tracking_pipeline_leaf
from tifffile import imwrite, imread
import random
import csv
from pathlib import Path
from scipy.ndimage import center_of_mass
from tifffile import TiffWriter

from skimage.draw import disk
from PyQt5.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox
if TYPE_CHECKING:
    import napari


import shutil
import atexit

TMP_DIR = Path("/tmp/napari/")  

save_button_ref = {}
segmentation_dock_ref = {}
previous_mouse_callbacks_racines = []
path_names = {}

def clean_tmp():
    """Nettoie les fichiers et dossiers temporaires (TIFF + memmaps)."""
    # 1. Supprimer ton dossier principal TMP_DIR (pour les TIFFs)
    try:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR)
            print(f"🧹 Dossier temporaire {TMP_DIR} supprimé")
    except Exception as e:
        print(f"⚠️ Erreur suppression {TMP_DIR}: {e}")

    # 2. Supprimer les dossiers de type /tmp/tmp* (memmaps via mkdtemp)
    tmp_root = Path("/tmp")
    for d in tmp_root.glob("tmp*"):
        try:
            if d.is_dir():
                shutil.rmtree(d)
                print(f"🧹 Dossier mmap {d} supprimé")
        except Exception as e:
            print(f"⚠️ Erreur suppression {d}: {e}")

# S'assure que le dossier est supprimé quand Python se termine
atexit.register(clean_tmp)


def make_widget():
    widget = load_volume_widget()    
    return widget



@magic_factory(call_button="Charger un volume")
def load_volume_widget(viewer: "napari.viewer.Viewer" = None):
    path, _ = QFileDialog.getOpenFileName(
        None, "Choisir un volume TIFF", "", "Fichiers TIFF (*.tif *.tiff)"
    )
    if path and Path(path).exists():
        volume = imread(path)

        # Supprime la couche "Volume" si elle existe déjà
        existing_layer = next((layer for layer in viewer.layers if layer.name == "Volume"), None)
        if existing_layer:
            viewer.layers.remove(existing_layer)

        volume_layer = viewer.add_image(volume, name="Volume", colormap="gray",blending='additive')

        # Supprime le dock "🔬 Segmenter volume" s'il existe déjà
        if "segment_dock" in segmentation_dock_ref:
            try:
                viewer.window.remove_dock_widget(segmentation_dock_ref["segment_dock"])
            except Exception as e:
                print(f"Erreur en retirant l'ancien dock : {e}")
            segmentation_dock_ref.pop("segment_dock", None)

        # Crée et ajoute un nouveau widget de segmentation
        seg_widget_factory = build_segment_volume_widget(volume.shape)
        seg_widget = seg_widget_factory()
        seg_widget.loaded_volume = volume 
        QTimer.singleShot(0, lambda: getattr(seg_widget, "click_mode", None) and seg_widget.click_mode.native.setVisible(False))

        seg_widget.volume_path.value = str(path)
        dock = viewer.window.add_dock_widget(seg_widget, name="🔬 Segmenter volume", area="right")

        # Garde une référence au dock ajouté
        segmentation_dock_ref["segment_dock"] = dock

        def on_mouse_click(layer, event):
            if event.type == 'mouse_press' and event.button == 1:
                pos = layer.world_to_data(event.position)
                z, y, x = map(int, pos)
                print(f"📍 Clic sur Volume : z={z}, y={y}, x={x}")

                mode = seg_widget.click_mode.value  # récupère le mode clic dans le widget
                print(f"💡 Mode clic actuel au moment du clic: {mode}")
                if mode == "Centre":
                    seg_widget.tap_x.value = x
                    seg_widget.tap_y.value = y
                    seg_widget.tap_z.value = z
                    print(f"Centre mis à jour : x={x}, y={y}, z={z}")
                elif mode == "Coin bas gauche":
                    print("saclic")
                    seg_widget.low_corner_x.setValue(x)
                    seg_widget.low_corner_y.setValue(y)
                    print(f"Coin bas gauche mis à jour : x={x}, y={y}")
                elif mode == "Coin haut droit":
                    seg_widget.high_corner_x.setValue(x)
                    seg_widget.high_corner_y.setValue(y)
                    print(f"Coin haut droit mis à jour : x={x}, y={y}")


        volume_layer.mouse_drag_callbacks.append(on_mouse_click)

save_button_ref = {}
def build_segment_volume_widget(volume_shape):
    max_z, max_y, max_x = volume_shape
    
    @magic_factory(
        model_root_path={"widget_type": "FileEdit", "label": "Chemin du modèle pour racines", "mode": "r"},
        model_leaf_path={"widget_type": "FileEdit", "label": "Chemin du modèle pour feuilles", "mode": "r"},
        volume_path={"widget_type": "FileEdit", "label": "Volume à segmenter", "mode": "r"},
        output_path={"widget_type": "FileEdit", "label": "Fichier de sortie (optionnel)", "nullable": True, "mode": "w"},
        patch_size={"label": "Taille du patch", "min": 16, "max": 256, "step": 16},
        stride={"label": "Stride", "min": 8, "max": 256, "step": 8},
        batch_size={"label": "Taille du batch", "min": 1, "max": 64, "step": 1},
        tap_x={"label": "Centre X", "min": 0, "max": max_x - 1, "step": 1},
        tap_y={"label": "Centre Y", "min": 0, "max": max_y, "step": 1},
        tap_z={"label": "Centre Z", "min": 0, "max": max_z, "step": 1},
        pretreatment={"label": "Prétraitement", "widget_type": "CheckBox", "value": False},
        click_mode={"label": "Mode clic", "widget_type": "ComboBox", "choices": ["Centre", "Coin bas gauche", "Coin haut droit"], "value": "Centre"},
    )
    def segment_volume_widget(
        model_root_path: str,
        model_leaf_path: str,
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
        click_mode: str = "Centre",
    ) -> None:
        volume = segment_volume_widget.loaded_volume
        tap_center = (tap_x, tap_y, tap_z)
        print(f"Viewer :  {viewer}")
        print(f"📍 Centre du plateau sélectionné : {tap_center}")

        z_max = tap_z + 100
        z_min= tap_z - 100
        print(f"z max apres recup tap center : {z_max}")
        leaf_seg=False
        root_seg=False
        # Segmentation feuille
        if model_leaf_path and Path(model_leaf_path).is_file():
            # On récupère les chemins vers les memmaps + origine patchs
            probas_volume_path, segmentation_path, (zmin_leaf, ymin_leaf, xmin_leaf) = segment_volume_leaf(
                model_path=model_leaf_path,
                volume=volume,
                output_path=output_path,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
                pretreatment=pretreatment,
                tap_center=tap_center,
            )
            leaf_seg = True
        else:
            print(f"[Info] Aucun modèle feuille fourni ou chemin invalide : {model_leaf_path}")

        if leaf_seg:
            print("📁 Chargement des memmaps des probabilités feuilles...")
            # Charge le memmap directement à partir du chemin (lecture disque, pas RAM)
            probas_volume_leaf = np.memmap(probas_volume_path, dtype=np.float32, mode='r+')
            segmentation_leaf = np.memmap(segmentation_path, dtype=np.uint8, mode='r+')

            # Ici tu dois reconstruire la forme du volume (le chemin .dat ne contient pas la forme, mais on la connaît)
            # Souvent, ta fonction segment_volume_leaf utilise la forme originale, donc il faut la passer ou la récupérer
            # Pour simplifier, on suppose que la forme est connue : par exemple
            shape = volume.shape
            num_classes = 3

            # Reshape memmap (mémoire virtuelle, pas de chargement)
            probas_volume_leaf = probas_volume_leaf.reshape((num_classes, *shape))
            segmentation_leaf = segmentation_leaf.reshape(shape)

            # Sauvegarde TIFF des probabilités (avec conversion float16 pour gain place)
            output_dir = Path(output_path or "/tmp/napari/")
            output_dir.mkdir(parents=True, exist_ok=True)
            leaf_proba_path = output_dir / "probas_leaf_big.tif"

            with TiffWriter(str(leaf_proba_path), bigtiff=True) as tif:
                for i in range(probas_volume_leaf.shape[1]):  # shape = (C, Z, Y, X) donc slices = dim 1
                    tif.write(np.array(probas_volume_leaf[:, i], dtype=np.float16), contiguous=True)
                    print(f"✅ Sauvegarde tranche {i + 1}/{probas_volume_leaf.shape[1]}")

            print(f"📁 Probabilités feuilles sauvegardées à {leaf_proba_path}")

        # Segmentation racine
        if model_root_path and Path(model_root_path).is_file():
            probas_volume_path_root, segmented_path_root, (zmin_root, ymin_root, xmin_root) = segment_volume_root(
                model_path=model_root_path,
                volume=volume,
                output_path=output_path,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
                pretreatment=pretreatment,
                tap_center=tap_center,
            )
            root_seg=True
        else:
            print(f"[Info] Aucun modèle racine fourni ou chemin invalide : {model_root_path}")

        if root_seg :
            print("📁 Chargement des memmaps des probabilités racines...")
            # Charge le memmap directement à partir du chemin (lecture disque, pas RAM)
            probas_volume = np.memmap(probas_volume_path_root, dtype=np.float32, mode='r+')
            segmentation = np.memmap(segmented_path_root, dtype=np.uint8, mode='r+')

            # Ici tu dois reconstruire la forme du volume (le chemin .dat ne contient pas la forme, mais on la connaît)
            # Souvent, ta fonction segment_volume_leaf utilise la forme originale, donc il faut la passer ou la récupérer
            # Pour simplifier, on suppose que la forme est connue : par exemple
            shape = volume.shape

            # Reshape memmap (mémoire virtuelle, pas de chargement)
            probas_volume = probas_volume.reshape(shape)  # Z, Y, X
            segmentation = segmentation.reshape(shape)

            # Sauvegarde TIFF des probabilités (avec conversion float16 pour gain place)
            output_dir = Path(output_path or "/tmp/napari")
            output_dir.mkdir(parents=True, exist_ok=True)
            root_proba_path = output_dir / "probas_root_big.tif"

            with TiffWriter(str(root_proba_path), bigtiff=True) as tif:
                for z in range(probas_volume.shape[0]):  # axe Z
                    tif.write(probas_volume[z].astype(np.float16), contiguous=True)
                    print(f"✅ Sauvegarde tranche {z + 1}/{probas_volume.shape[0]}")

            print(f"📁 Probabilités racines sauvegardées à {root_proba_path}")

        desired_order = ["Volume","Segmentation racine","Segmentation feuille"]

        # Afficher la probabilité et la segmentation des feuilles si disponibles
        if  leaf_seg:
            viewer.add_labels(segmentation_leaf, name="Segmentation feuille")
            desired_order.append("Segmentation feuille")

        # Afficher la probabilité et la segmentation des racines si disponibles
        if  root_seg:
            viewer.add_labels(segmentation, name="Segmentation racine")
            desired_order.append("Segmentation racine")

        # S'assurer que le volume est toujours actif et premier
        viewer.layers.selection.active = viewer.layers["Volume"]

        # Réorganiser les couches
        def reorder_layers():
            for target_index, name in enumerate(reversed(desired_order)):
                for current_index, layer in enumerate(viewer.layers):
                    if layer.name == name:
                        viewer.layers.move(current_index, target_index)
                        break

        reorder_layers()

        def update_selection_rectangle():
            if not viewer:
                return

            low_x = low_corner_x.value()
            low_y = low_corner_y.value()
            high_x = high_corner_x.value()
            high_y = high_corner_y.value()

            z_min = 0
            z_max = volume.shape[0] - 1  # profondeur

            # Supprimer ancienne couche
            for layer in viewer.layers:
                if layer.name == "Zone sélectionnée":
                    viewer.layers.remove(layer)
                    break

            rectangles = []
            for z in range(z_min, z_max + 1):
                rectangle = [
                    [z, low_y, low_x],
                    [z, low_y, high_x],
                    [z, high_y, high_x],
                    [z, high_y, low_x],
                ]
                rectangles.append(rectangle)

            viewer.add_shapes(
                rectangles,  # ✅ un rectangle par tranche z
                shape_type='rectangle',
                edge_color='red',
                edge_width=2,
                face_color=[1, 0, 0, 0.2],
                name='Zone sélectionnée',
                opacity=0.5,
            )
            viewer.layers.selection.active = viewer.layers["Volume"]



        

        # Nettoyer anciens boutons/docks liés à la sauvegarde
        for btn_key in ["save_button_proba", "save_button_segmented", "save_button_tracking", "corners_container", "btn_run_tracking", "save_button_csv", "btn_extract_slices"]:
            if btn_key in save_button_ref:
                widget = save_button_ref[btn_key]
                if hasattr(widget, "deleteLater"):
                    widget.deleteLater()
                else:
                    viewer.window.remove_dock_widget(widget)
                save_button_ref.pop(btn_key)

        button_container = QWidget()
        layout = QVBoxLayout()
        button_container.setLayout(layout)

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
                imwrite(save_path, segmentation.astype(np.uint8))
                print(f"✅ Segmentation binaire sauvegardée à : {save_path}")

        def save_probas_leaf():
            save_path, _ = QFileDialog.getSaveFileName(
                caption="Enregistrer les probabilités des feuilles",
                filter="Fichiers TIFF (*.tiff *.tif)",
            )
            if save_path:
                imwrite(save_path, probas_volume_leaf.astype(np.float32))
                print(f"✅ Probabilités des feuilles sauvegardées à : {save_path}")

        def save_segmented_leaf():
            save_path, _ = QFileDialog.getSaveFileName(
                caption="Enregistrer la segmentation binaire des feuilles",
                filter="Fichiers TIFF (*.tiff *.tif)",
            )
            if save_path:
                imwrite(save_path, segmententation_leaf.astype(np.uint8))
                print(f"✅ Segmentation binaire des feuilles sauvegardée à : {save_path}")

        if root_seg:
            button_proba = QPushButton("Sauvegarder les probabilités")
            button_proba.clicked.connect(save_probas)
            layout.addWidget(button_proba)
            save_button_ref["save_button_proba"] = button_proba

            button_seg = QPushButton("Sauvegarder la segmentation binaire")
            button_seg.clicked.connect(save_segmented)
            layout.addWidget(button_seg)
            save_button_ref["save_button_segmented"] = button_seg

        if leaf_seg: 
            button_proba_leaf = QPushButton("Sauvegarder les probabilités des feuilles")
            button_proba_leaf.clicked.connect(save_probas_leaf)
            layout.addWidget(button_proba_leaf)
            save_button_ref["save_button_proba_leaf"] = button_proba_leaf

            button_seg_leaf = QPushButton("Sauvegarder la segmentation binaire des feuilles")
            button_seg_leaf.clicked.connect(save_segmented_leaf)
            layout.addWidget(button_seg_leaf)
            save_button_ref["save_button_segmented_leaf"] = button_seg_leaf

        corners_container = QWidget()
        corners_layout = QVBoxLayout()
        corners_container.setLayout(corners_layout)

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

        low_corner_x.valueChanged.connect(update_selection_rectangle)
        low_corner_y.valueChanged.connect(update_selection_rectangle)
        high_corner_x.valueChanged.connect(update_selection_rectangle)
        high_corner_y.valueChanged.connect(update_selection_rectangle)

        
        segment_volume_widget.low_corner_x = low_corner_x
        segment_volume_widget.low_corner_y = low_corner_y
        segment_volume_widget.high_corner_x = high_corner_x
        segment_volume_widget.high_corner_y = high_corner_y

        def after_segmentation():
            segment_volume_widget.click_mode.value = "Coin bas gauche"
            segment_volume_widget.click_mode.native.setVisible(True)
            layout.addWidget(corners_container)
            corners_container.show()
            save_button_ref["corners_container"] = corners_container

            def recentre_point_local_volume(volume, z, y, x, win_size=64):
                half = win_size // 2
                y_min = max(0, y - half)
                y_max = min(volume.shape[1], y + half)
                x_min = max(0, x - half)
                x_max = min(volume.shape[2], x + half)

                patch = volume[z, y_min:y_max, x_min:x_max]

                if np.sum(patch) == 0:
                    return y, x

                com = center_of_mass(patch)

                if np.any(np.isnan(com)):
                    return y, x  # Fallback si le COM échoue

                com_y_patch, com_x_patch = com
                com_y = y_min + com_y_patch
                com_x = x_min + com_x_patch

                new_y = max(0, min(volume.shape[1] - 1, int(round(com_y))))
                new_x = max(0, min(volume.shape[2] - 1, int(round(com_x))))

                return new_y, new_x


            def run_tracking_with_corners():
                low_corner = (low_corner_x.value(), low_corner_y.value())
                high_corner = (high_corner_x.value(), high_corner_y.value())
                print(f"Utilisation des coins : low {low_corner}, high {high_corner}")
                print("🚀 Lancement du tracking...")
                print(f"Zmax avant run tracking pipeline {z_max}")
                if(root_seg) :
                    all_paths_root  = run_tracking_pipeline(
                        volume, tap_center, low_corner, high_corner, zmax=z_max,
                        probas_volume=probas_volume, segmented=segmentation
                    )
                if(leaf_seg) : 
                    all_paths_leaves = run_tracking_pipeline_leaf(
                        volume, tap_center, low_corner, high_corner, zmin=z_min,
                        probas_volume=probas_volume_leaf, segmented=segmentation_leaf
                    )
                print("✅ Tracking terminé.")
                slice_size = 16
                all_paths_recal_root = []
                all_paths_recal_leaf = []
                n_iter = 20
                if(root_seg) :
                    for seed, score, path in all_paths_root:
                        new_path = []
                        for z, y, x in path:
                            for _ in range(n_iter):
                                y, x = recentre_point_local_volume(probas_volume, z, y, x, slice_size)
                            new_path.append((z, y, x))
                        all_paths_recal_root.append((seed, score, new_path))
                if(leaf_seg) :
                    for seed, score, path in all_paths_leaves:
                        new_path = []
                        for z, y, x in path:
                            for _ in range(n_iter):
                                y, x = recentre_point_local_volume(probas_volume_leaf, z, y, x, slice_size)
                            new_path.append((z, y, x))
                        all_paths_recal_leaf.append((seed, score, new_path))

                
                print("🧠 Chemins recalés")

                def make_color_mask(paths, shape, radius=3):
                    """
                    Crée un masque coloré avec des disques autour des points donnés.

                    :param paths: liste de tuples (non utilisé, non utilisé, path), où path est une liste de (z, y, x)
                    :param shape: shape du volume (z, y, x)
                    :param radius: rayon du disque en pixels (dans le plan yx)
                    :return: masque RGB (z, y, x, 3)
                    """
                    color_mask = np.zeros(shape + (3,), dtype=np.uint8)
                    
                    for idx, (_, _, path) in enumerate(paths):
                        color = tuple(random.choices(range(50, 256), k=3))
                        for z, y, x in path:
                            if 0 <= z < shape[0]:
                                rr, cc = disk((y, x), radius, shape=shape[1:])  # disk dans le plan (y, x)
                                color_mask[z, rr, cc] = color  # applique la couleur dans ce plan

                    return color_mask
                                
                if "Zone sélectionnée" in viewer.layers:
                    viewer.layers.remove("Zone sélectionnée")
                if root_seg : 
                    viewer.add_image(make_color_mask(all_paths_recal_root,segmentation.shape), name="Chemins colorés recalés racines")
                if leaf_seg :
                    viewer.add_image(make_color_mask(all_paths_recal_leaf,segmentation_leaf.shape), name="Chemins colorés recalés feuilles")

                from collections import defaultdict
                manual_points_stack_root = defaultdict(list)
                anual_points_stack_leaf = defaultdict(list)
                def reorder_layers_afterseg():
                    desired_order = ["Volume","Chemins colorés recalés racines","Chemins colorés recalés feuilles", "Segmentation racines", "Segmentation feuilles"]

                    for target_index, name in enumerate(reversed(desired_order)):
                        for current_index, layer in enumerate(viewer.layers):
                            if layer.name == name:
                                viewer.layers.move(current_index, target_index)
                                break

                reorder_layers_afterseg()

                highlight_layer_name = "Chemin sélectionné"

                def update_highlighted_path_root(index):
                    if highlight_layer_name in viewer.layers:
                        viewer.layers.remove(highlight_layer_name)
                    _, _, path = all_paths_recal_root[index]
                    highlight = np.zeros(probas_volume.shape, dtype=np.uint8)
                    
                    for z, y, x in path:
                        for dy in [-2, -1, 0, 1, 2]:
                            for dx in [-2, -1, 0, 1, 2]:
                                yy = y + dy
                                xx = x + dx
                                if 0 <= yy < highlight.shape[1] and 0 <= xx < highlight.shape[2]:
                                    highlight[z, yy, xx] = 1

                    viewer.add_labels(highlight, name=highlight_layer_name, opacity=1.0)
                    label_layer = viewer.layers[highlight_layer_name]

                def update_highlighted_path_leaf(index):
                    if highlight_layer_name in viewer.layers:
                        viewer.layers.remove(highlight_layer_name)
                    _, _, path = all_paths_recal_leaf[index]
                    highlight = np.zeros(probas_volume.shape, dtype=np.uint8)
                    
                    for z, y, x in path:
                        for dy in [-2, -1, 0, 1, 2]:
                            for dx in [-2, -1, 0, 1, 2]:
                                yy = y + dy
                                xx = x + dx
                                if 0 <= yy < highlight.shape[1] and 0 <= xx < highlight.shape[2]:
                                    highlight[z, yy, xx] = 1

                    viewer.add_labels(highlight, name=highlight_layer_name, opacity=1.0)
                    label_layer = viewer.layers[highlight_layer_name]
                
                current_path_index_root = [0]
                current_path_index_leaf = [0]
                def on_path_selected_root(index):
                    current_path_index_root[0] = index
                    update_highlighted_path_root(index)

                def on_path_selected_leaf(index):
                    current_path_index_leaf[0] = index
                    update_highlighted_path_leaf(index)

                from qtpy.QtWidgets import QComboBox
                path_selector_root = QComboBox()
                path_selector_root.currentIndexChanged.connect(on_path_selected_root)
                name_editor_root = QLineEdit()

                path_selector_leaf = QComboBox()
                path_selector_leaf.currentIndexChanged.connect(on_path_selected_leaf)
                name_editor_leaf = QLineEdit()

                def update_name_editor_root(index):
                    name = path_names.get(index, f"Chemin {index}")
                    name_editor_root.setText(name)

                def update_name_editor_leaf(index):
                    name = path_names.get(index, f"Chemin {index}")
                    name_editor_leaf.setText(name)

                
                path_selector_root.currentIndexChanged.connect(update_name_editor_root)
                path_selector_leaf.currentIndexChanged.connect(update_name_editor_leaf)
                def rename_path_root():
                    index = path_selector_root.currentIndex()
                    new_name = name_editor_root.text().strip()
                    if new_name:
                        path_names[index] = new_name
                        path_selector_root.setItemText(index, new_name)
                        print(f"✏️ Chemin {index} renommé en : {new_name}")

                def rename_path_leaf():
                    index = path_selector_leaf.currentIndex()
                    new_name = name_editor_leaf.text().strip()
                    if new_name:
                        path_names[index] = new_name
                        path_selector_leaf.setItemText(index, new_name)
                        print(f"✏️ Chemin {index} renommé en : {new_name}")

                name_editor_root.editingFinished.connect(rename_path_root)   
                layout.addWidget(name_editor_root)
                save_button_ref["name_editor_root"] = name_editor_root
                adding_mode_root = [False]
                creating_new_path_mode_root = [False]
                new_path_points_root = []
                btn_new_path_root = QPushButton("🆕 Créer un nouveau chemin racine")
                layout.addWidget(btn_new_path_root)

                name_editor_leaf.editingFinished.connect(rename_path_leaf)   
                layout.addWidget(name_editor_leaf)
                save_button_ref["name_editor_leaf"] = name_editor_leaf
                adding_mode_leaf = [False]
                creating_new_path_mode_leaf = [False]
                new_path_points_leaf = []
                btn_new_path_leaf = QPushButton("🆕 Créer un nouveau chemin feuille")
                layout.addWidget(btn_new_path_leaf)


                def on_click_create_new_path_point(layer, event):
                    if not creating_new_path_mode_root[0]:
                        return
                    if event.type == 'mouse_press' and event.button == 1:
                        pos = layer.world_to_data(event.position)
                        z, y, x = map(int, pos)
                        pt = (z, y, x)
                        new_path_points_root.append(pt)
                        print(f"🆕 ➕ Point ajouté à nouveau chemin : {pt}")

                def on_click_create_new_path_point_leaf(layer, event):
                    if not creating_new_path_mode_leaf[0]:
                        return
                    if event.type == 'mouse_press' and event.button == 1:
                        pos = layer.world_to_data(event.position)
                        z, y, x = map(int, pos)
                        pt = (z, y, x)
                        new_path_points_leaf.append(pt)
                        print(f"🆕 ➕ Point ajouté à nouveau chemin : {pt}")

                def toggle_create_new_path_mode():
                    creating_new_path_mode_root[0] = not creating_new_path_mode_root[0]
                    state = "activé" if creating_new_path_mode_root[0] else "désactivé"
                    print(f"🆕 Mode création de chemin {state}")
                    new_path_points_root.clear()

                    volume_layer = viewer.layers["Volume"]
                    if creating_new_path_mode_root[0]:
                        # Sauver et désactiver les autres callbacks
                        global previous_mouse_callbacks_racines
                        previous_mouse_callbacks_racines = list(volume_layer.mouse_drag_callbacks)
                        volume_layer.mouse_drag_callbacks.clear()
                        volume_layer.mouse_drag_callbacks.append(on_click_create_new_path_point)
                    else:
                        # Restaure les anciens
                        volume_layer.mouse_drag_callbacks.clear()
                        for cb in previous_mouse_callbacks_racines:
                            volume_layer.mouse_drag_callbacks.append(cb)
                        previous_mouse_callbacks_racines.clear()

                def toggle_create_new_path_mode_leaf():
                    creating_new_path_mode_leaf[0] = not creating_new_path_mode_leaf[0]
                    state = "activé" if creating_new_path_mode_leaf[0] else "désactivé"
                    print(f"🆕 Mode création de chemin {state}")
                    new_path_points_leaf.clear()

                    volume_layer = viewer.layers["Volume"]
                    if creating_new_path_mode_leaf[0]:
                        # Sauver et désactiver les autres callbacks
                        global previous_mouse_callbacks_feuilles
                        previous_mouse_callbacks_feuilles = list(volume_layer.mouse_drag_callbacks)
                        volume_layer.mouse_drag_callbacks.clear()
                        volume_layer.mouse_drag_callbacks.append(on_click_create_new_path_point_leaf)
                    else:
                        # Restaure les anciens
                        volume_layer.mouse_drag_callbacks.clear()
                        for cb in previous_mouse_callbacks_feuilles:
                            volume_layer.mouse_drag_callbacks.append(cb)
                        previous_mouse_callbacks_feuilles.clear()

                btn_new_path_root.clicked.connect(toggle_create_new_path_mode)
                btn_validate_new_path = QPushButton("✅ Valider le nouveau chemin")
                layout.addWidget(btn_validate_new_path)

                btn_new_path_leaf.clicked.connect(toggle_create_new_path_mode_leaf)
                btn_validate_new_path_leaf = QPushButton("✅ Valider le nouveau chemin")
                layout.addWidget(btn_validate_new_path_leaf)

                def validate_new_path():
                    if len(new_path_points_root) < 2:
                        print("⚠️ Il faut au moins deux points pour créer un chemin.")
                        return

                    sorted_points = sorted(new_path_points_root)
                    interpolated = interpolate_points(sorted_points)

                    new_index = len(all_paths_recal_root)
                    all_paths_recal_root.append(("inconnu", "inconnu", interpolated))
                    manual_points_stack_root[new_index] = [interpolated]

                    print(f"✅ Nouveau chemin #{new_index} créé avec {len(interpolated)} points.")

                    # Mise à jour du sélecteur
                    path_selector_root.addItem(f"Chemin {new_index}", new_index)

                    # Optionnel : sélectionner et afficher
                    current_path_index_root[0] = new_index
                    update_highlighted_path_root(new_index)

                    if "Chemins colorés recalés racines" in viewer.layers:
                        viewer.layers.remove("Chemins colorés recalés racines")
                    viewer.add_image(make_color_mask(all_paths_recal_root,segmentation.shape), name="Chemins colorés recalés racines")
                    reorder_layers_add()
                    viewer.layers.selection.active = viewer.layers["Volume"]

                    # Reset
                    new_path_points_root.clear()
                    toggle_create_new_path_mode()  # désactive le mode création
                    path_names[new_index] = f"Chemin {new_index}"
                    path_selector_root.addItem(path_names[new_index], new_index)

                def validate_new_path_leaf():
                    if len(new_path_points_leaf) < 2:
                        print("⚠️ Il faut au moins deux points pour créer un chemin.")
                        return

                    sorted_points = sorted(new_path_points_leaf)
                    interpolated = interpolate_points(sorted_points)

                    new_index = len(all_paths_recal_leaf)
                    all_paths_recal_leaf.append(("inconnu", "inconnu", interpolated))
                    manual_points_stack_leaf[new_index] = [interpolated]

                    print(f"✅ Nouveau chemin #{new_index} créé avec {len(interpolated)} points.")

                    # Mise à jour du sélecteur
                    path_selector_root.addItem(f"Chemin {new_index}", new_index)

                    # Optionnel : sélectionner et afficher
                    current_path_index_root[0] = new_index
                    update_highlighted_path_leaf(new_index)

                    if "Chemins colorés recalés feuilles" in viewer.layers:
                        viewer.layers.remove("Chemins colorés recalés feuilles")
                    viewer.add_image(make_color_mask(all_paths_recal_leaf,segmentation.shape), name="Chemins colorés recalés feuilles")
                    reorder_layers_add()
                    viewer.layers.selection.active = viewer.layers["Volume"]

                    # Reset
                    new_path_points_leaf.clear()
                    toggle_create_new_path_mode_leaf()  # désactive le mode création
                    path_names[new_index] = f"Chemin {new_index}"
                    path_selector_leaf.addItem(path_names[new_index], new_index)

                btn_validate_new_path.clicked.connect(validate_new_path)
                btn_validate_new_path_leaf.clicked.connect(validate_new_path_leaf)

                btn_add_points = QPushButton("➕ Ajouter des points à ce chemin de racines")
                layout.addWidget(btn_add_points)
                previous_mouse_callbacks_racines = []

                btn_add_points_feuilles = QPushButton("➕ Ajouter des points à ce chemin de feuilles")
                layout.addWidget(btn_add_points_feuilles)
                previous_mouse_callbacks_feuilles = []

                def toggle_add_mode():
                    viewer.layers.selection.active = viewer.layers["Volume"]
                    adding_mode_root[0] = not adding_mode_root[0]
                    state = "activé" if adding_mode_root[0] else "désactivé"
                    print(f"🖱️ Mode ajout de points {state}")
                    
                    volume_layer = viewer.layers["Volume"]
                    
                    if adding_mode_root[0]:
                        # Sauvegarder les callbacks existants et les retirer
                        global previous_mouse_callbacks_racines
                        previous_mouse_callbacks_racines = list(volume_layer.mouse_drag_callbacks)
                        volume_layer.mouse_drag_callbacks.clear()
                        volume_layer.mouse_drag_callbacks.append(on_click_add_point)
                    else:
                        # Désactiver ajout de point et restaurer les anciens
                        volume_layer.mouse_drag_callbacks.clear()
                        for cb in previous_mouse_callbacks_racines:
                            volume_layer.mouse_drag_callbacks.append(cb)
                        previous_mouse_callbacks_racines.clear()

                def toggle_add_mode_feuilles():
                    viewer.layers.selection.active = viewer.layers["Volume"]
                    adding_mode_leaf[0] = not adding_mode_leaf[0]
                    state = "activé" if adding_mode_leaf[0] else "désactivé"
                    print(f"🖱️ Mode ajout de points {state}")
                    
                    volume_layer = viewer.layers["Volume"]
                    
                    if adding_mode_leaf[0]:
                        # Sauvegarder les callbacks existants et les retirer
                        global previous_mouse_callbacks_racines
                        previous_mouse_callbacks_racines = list(volume_layer.mouse_drag_callbacks)
                        volume_layer.mouse_drag_callbacks.clear()
                        volume_layer.mouse_drag_callbacks.append(on_click_add_point)
                    else:
                        # Désactiver ajout de point et restaurer les anciens
                        volume_layer.mouse_drag_callbacks.clear()
                        for cb in previous_mouse_callbacks_racines:
                            volume_layer.mouse_drag_callbacks.append(cb)
                        previous_mouse_callbacks_racines.clear()

                btn_add_points.clicked.connect(toggle_add_mode)
                btn_add_points_feuilles.clicked.connect(toggle_add_mode_feuilles)
                def reorder_layers_add():
                    desired_order = ["Volume","Chemin sélectionné","Chemins colorés recalés racines","Chemins colorés recalés feuilles", "Segmentation racines", "Segmentation feuilles"]

                    for target_index, name in enumerate(reversed(desired_order)):
                        for current_index, layer in enumerate(viewer.layers):
                            if layer.name == name:
                                viewer.layers.move(current_index, target_index)
                                break

                def interpolate_points(path_points):
                    # path_points doit être trié par z croissant
                    interpolated = []
                    for i in range(len(path_points)-1):
                        z0, y0, x0 = path_points[i]
                        z1, y1, x1 = path_points[i+1]
                        interpolated.append((z0, y0, x0))
                        dz = z1 - z0
                        if dz > 1:
                            for z in range(z0 + 1, z1):
                                alpha = (z - z0) / dz
                                y = int(round((1 - alpha) * y0 + alpha * y1))
                                x = int(round((1 - alpha) * x0 + alpha * x1))
                                interpolated.append((z, y, x))
                    interpolated.append(path_points[-1])
                    return interpolated
                
                def on_click_add_point(layer, event):
                    if not adding_mode_root[0]:
                        return
                    if event.type == 'mouse_press' and event.button == 1:
                        pos = layer.world_to_data(event.position)
                        z, y, x = map(int, pos)
                        index = current_path_index_root[0]

                        existing_zs = [pt[0] for pt in all_paths_recal_root[index][2]]
                        if z in existing_zs:
                            print(f"⚠️ Le point avec z={z} existe déjà dans ce chemin, ajout ignoré.")
                            return

                        new_point = (z, y, x)
                        print(f"➕ Ajout point {new_point} au chemin {index}")

                        # Ajouter temporairement le point
                        all_paths_recal_root[index][2].append(new_point)
                        sorted_points = sorted(all_paths_recal_root[index][2])
                        interpolated = interpolate_points(sorted_points)

                        # Déterminer les points effectivement ajoutés par interpolation
                        interpolated_set = set(interpolated)
                        original_set = set(all_paths_recal_root[index][2])
                        added_by_interp = list(interpolated_set - original_set)

                        # Mémoriser tous les points ajoutés pour ce clic (le point + interpolation)
                        full_added = [new_point] + added_by_interp
                        manual_points_stack_root[index].append(full_added)

                        # Mise à jour du chemin
                        all_paths_recal_root[index] = (
                            all_paths_recal_root[index][0],
                            all_paths_recal_root[index][1],
                            interpolated
                        )

                        update_highlighted_path_root(index)
                        if "Chemins colorés recalés racines" in viewer.layers:
                            viewer.layers.remove("Chemins colorés recalés racines")
                            viewer.add_image(make_color_mask(all_paths_recal_root,segmentation.shape), name="Chemins colorés recalés racines")
                            reorder_layers_add()
                            viewer.layers.selection.active = viewer.layers["Volume"]

                def on_click_add_point_leaf(layer, event):
                    if not adding_mode_root[0]:
                        return
                    if event.type == 'mouse_press' and event.button == 1:
                        pos = layer.world_to_data(event.position)
                        z, y, x = map(int, pos)
                        index = current_path_index_leaf[0]

                        existing_zs = [pt[0] for pt in all_paths_recal_leaf[index][2]]
                        if z in existing_zs:
                            print(f"⚠️ Le point avec z={z} existe déjà dans ce chemin, ajout ignoré.")
                            return

                        new_point = (z, y, x)
                        print(f"➕ Ajout point {new_point} au chemin {index}")

                        # Ajouter temporairement le point
                        all_paths_recal_leaf[index][2].append(new_point)
                        sorted_points = sorted(all_paths_recal_leaf[index][2])
                        interpolated = interpolate_points(sorted_points)

                        # Déterminer les points effectivement ajoutés par interpolation
                        interpolated_set = set(interpolated)
                        original_set = set(all_paths_recal_leaf[index][2])
                        added_by_interp = list(interpolated_set - original_set)

                        # Mémoriser tous les points ajoutés pour ce clic (le point + interpolation)
                        full_added = [new_point] + added_by_interp
                        manual_points_stack_leaf[index].append(full_added)

                        # Mise à jour du chemin
                        all_paths_recal_leaf[index] = (
                            all_paths_recal_leaf[index][0],
                            all_paths_recal_leaf[index][1],
                            interpolated
                        )

                        update_highlighted_path_root(index)
                        if "Chemins colorés recalés feuilles" in viewer.layers:
                            viewer.layers.remove("Chemins colorés recalés feuilles")
                            viewer.add_image(make_color_mask(all_paths_recal_leaf,segmentation.shape), name="Chemins colorés recalés feuilles")
                            reorder_layers_add()
                            viewer.layers.selection.active = viewer.layers["Volume"]

                btn_delete_path = QPushButton("🗑️ Supprimer ce chemin racine")
                layout.addWidget(btn_delete_path)

                btn_delete_path_leaf = QPushButton("🗑️ Supprimer ce chemin feuille")
                layout.addWidget(btn_delete_path_leaf)

                def delete_current_path():
                    index = current_path_index_root[0]
                    
                    if index < 0 or index >= len(all_paths_recal_root):
                        print("❌ Index de chemin invalide.")
                        return

                    confirm = QMessageBox.question(
                        None,
                        "Confirmer la suppression",
                        f"Voulez-vous vraiment supprimer le chemin #{index} ?",
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if confirm == QMessageBox.No:
                        return

                    print(f"🗑️ Suppression du chemin #{index}")
                    
                    # Supprimer le chemin
                    all_paths_recal_root.pop(index)

                    # Mise à jour du combo
                    path_selector_root.clear()
                    for i in range(len(all_paths_recal_root)):
                        name = path_names.get(i, f"Chemin {i}")
                        path_selector_root.addItem(name, i)

                    if len(all_paths_recal_root) > 0:
                        current_path_index_root[0] = 0
                        update_highlighted_path_root(0)
                    else:
                        current_path_index_root[0] = -1
                        if "Chemin sélectionné" in viewer.layers:
                            viewer.layers.remove("Chemin sélectionné")

                    if "Chemins colorés recalés racines" in viewer.layers:
                        viewer.layers.remove("Chemins colorés recalés racines")
                    if len(all_paths_recal_root) > 0:
                        viewer.add_image(make_color_mask(all_paths_recal_root,segmentation.shape), name="Chemins colorés recalés racines")
                    reorder_layers_add()
                    viewer.layers.selection.active = viewer.layers["Volume"]
                    path_names.pop(index, None)

                def delete_current_path_leaf():
                    index = current_path_index_leaf[0]
                    
                    if index < 0 or index >= len(all_paths_recal_leaf):
                        print("❌ Index de chemin invalide.")
                        return

                    confirm = QMessageBox.question(
                        None,
                        "Confirmer la suppression",
                        f"Voulez-vous vraiment supprimer le chemin #{index} ?",
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if confirm == QMessageBox.No:
                        return

                    print(f"🗑️ Suppression du chemin #{index}")
                    
                    # Supprimer le chemin
                    all_paths_recal_leaf.pop(index)

                    # Mise à jour du combo
                    path_selector_leaf.clear()
                    for i in range(len(all_paths_recal_leaf)):
                        name = path_names.get(i, f"Chemin {i}")
                        path_selector_leaf.addItem(name, i)

                    if len(all_paths_recal_leaf) > 0:
                        current_path_index_leaf[0] = 0
                        update_highlighted_path_leaf(0)
                    else:
                        current_path_index_leaf[0] = -1
                        if "Chemin sélectionné" in viewer.layers:
                            viewer.layers.remove("Chemin sélectionné")

                    if "Chemins colorés recalés feuilles" in viewer.layers:
                        viewer.layers.remove("Chemins colorés recalés feuilles")
                    if len(all_paths_recal_leaf) > 0:
                        viewer.add_image(make_color_mask(all_paths_recal_leaf,segmentation.shape), name="Chemins colorés recalés feuilles")
                    reorder_layers_add()
                    viewer.layers.selection.active = viewer.layers["Volume"]
                    path_names.pop(index, None)

                btn_delete_path.clicked.connect(delete_current_path)
                btn_delete_path_leaf.clicked.connect(delete_current_path_leaf)



                def undo_last_manual_point():
                    index = current_path_index_root[0]

                    if not manual_points_stack_root[index]:
                        print("⚠️ Aucun point manuel à annuler.")
                        return

                    removed_points = manual_points_stack_root[index].pop()
                    print(f"↩️ Annulation des points manuels {removed_points} du chemin {index}")

                    # Supprimer ces points du chemin
                    remaining = [
                        pt for pt in all_paths_recal_root[index][2]
                        if pt not in removed_points
                    ]

                    all_paths_recal_root[index] = (
                        all_paths_recal_root[index][0],
                        all_paths_recal_root[index][1],
                        sorted(remaining)
                    )

                    update_highlighted_path_root(index)
                    if "Chemins colorés recalés racines" in viewer.layers:
                        viewer.layers.remove("Chemins colorés recalés racines")
                        viewer.add_image(make_color_mask(all_paths_recal_root,segmentation.shape), name="Chemins colorés recalés racines")
                        reorder_layers_add()
                        viewer.layers.selection.active = viewer.layers["Volume"]

                def undo_last_manual_point_leaf():
                    index = current_path_index_leaf[0]

                    if not manual_points_stack_leaf[index]:
                        print("⚠️ Aucun point manuel à annuler.")
                        return

                    removed_points = manual_points_stack_leaf[index].pop()
                    print(f"↩️ Annulation des points manuels {removed_points} du chemin {index}")

                    # Supprimer ces points du chemin
                    remaining = [
                        pt for pt in all_paths_recal_leaf[index][2]
                        if pt not in removed_points
                    ]

                    all_paths_recal_leaf[index] = (
                        all_paths_recal_leaf[index][0],
                        all_paths_recal_leaf[index][1],
                        sorted(remaining)
                    )

                    update_highlighted_path_leaf(index)
                    if "Chemins colorés recalés feuilles" in viewer.layers:
                        viewer.layers.remove("Chemins colorés recalés feuilles")
                        viewer.add_image(make_color_mask(all_paths_recal_leaf,segmentation.shape), name="Chemins colorés recalés feuilles")
                        reorder_layers_add()
                        viewer.layers.selection.active = viewer.layers["Volume"]

                btn_undo_point = QPushButton("↩️ Annuler dernier point racine")
                btn_undo_point.clicked.connect(undo_last_manual_point)
                layout.addWidget(btn_undo_point)
                viewer.layers["Volume"].mouse_drag_callbacks.append(on_click_add_point)

                btn_undo_point_leaf = QPushButton("↩️ Annuler dernier point feuille")
                btn_undo_point_leaf.clicked.connect(undo_last_manual_point_leaf)
                layout.addWidget(btn_undo_point_leaf)
                viewer.layers["Volume"].mouse_drag_callbacks.append(on_click_add_point_leaf)
                
                for i in range(len(all_paths_recal_root)):
                    path_selector_root.addItem(f"Chemin {i}", i)
                path_selector_root.currentIndexChanged.connect(update_highlighted_path_root)
                layout.addWidget(path_selector_root)

                save_button_ref["path_selector"] = path_selector_root

                # On affiche le premier chemin par défaut
                update_highlighted_path_root(0)

                viewer.layers["Segmentation racine"].visible = False
                viewer.layers.selection.active = viewer.layers["Volume"]

                for i in range(len(all_paths_recal_leaf)):
                    path_selector_leaf.addItem(f"Chemin {i}", i)
                path_selector_leaf.currentIndexChanged.connect(update_highlighted_path_leaf)
                layout.addWidget(path_selector_leaf)

                save_button_ref["path_selector_leaf"] = path_selector_leaf

                # On affiche le premier chemin par défaut
                update_highlighted_path_leaf(0)


                def save_updated_segmentation() :
                    all_paths_recal_root
                    segmentation


                def save_colored_paths():
                    save_path, _ = QFileDialog.getSaveFileName(caption="Enregistrer les chemins colorés", filter="*.tif")
                    if save_path:
                        imwrite(save_path, make_color_mask(all_paths_recal_root,segmentation.shape).astype(np.uint8))
                        print(f"✅ Chemins colorés sauvegardés à : {save_path}")

                def save_paths_to_csv():
                    dir_path = QFileDialog.getExistingDirectory(caption="Choisir un dossier pour les CSV")
                    if dir_path:
                        for idx, (_, _, path) in enumerate(all_paths_recal_root):
                            safe_name = path_names.get(idx, f"Chemin_{idx}").replace(" ", "_")
                            with open(Path(dir_path) / f"{safe_name}.csv", "w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(["X", "Y", "Z"])
                                for z, y, x in path:
                                    writer.writerow([x, y, z])
                        print(f"✅ {len(all_paths_recal_root)} chemins enregistrés")
                
                def trilinear_interpolation(volume, coords):
                    """
                    Interpolation trilineaire dans un volume 3D.

                    Args:
                        volume (np.ndarray): volume 3D (Z, Y, X)
                        coords (np.ndarray): coordonnées flottantes (3, H, W) sous forme (z, y, x)

                    Returns:
                        np.ndarray: image interpolée de taille (H, W)
                    """
                    z, y, x = coords

                    z0 = np.floor(z).astype(int)
                    z1 = z0 + 1
                    y0 = np.floor(y).astype(int)
                    y1 = y0 + 1
                    x0 = np.floor(x).astype(int)
                    x1 = x0 + 1

                    # Clip aux bornes pour ne pas sortir du volume
                    z0 = np.clip(z0, 0, volume.shape[0] - 1)
                    z1 = np.clip(z1, 0, volume.shape[0] - 1)
                    y0 = np.clip(y0, 0, volume.shape[1] - 1)
                    y1 = np.clip(y1, 0, volume.shape[1] - 1)
                    x0 = np.clip(x0, 0, volume.shape[2] - 1)
                    x1 = np.clip(x1, 0, volume.shape[2] - 1)

                    # Poids d'interpolation
                    dz = z - z0
                    dy = y - y0
                    dx = x - x0

                    # Interpolation trilineaire
                    c000 = volume[z0, y0, x0]
                    c001 = volume[z0, y0, x1]
                    c010 = volume[z0, y1, x0]
                    c011 = volume[z0, y1, x1]
                    c100 = volume[z1, y0, x0]
                    c101 = volume[z1, y0, x1]
                    c110 = volume[z1, y1, x0]
                    c111 = volume[z1, y1, x1]

                    c00 = c000 * (1 - dx) + c001 * dx
                    c01 = c010 * (1 - dx) + c011 * dx
                    c10 = c100 * (1 - dx) + c101 * dx
                    c11 = c110 * (1 - dx) + c111 * dx

                    c0 = c00 * (1 - dy) + c01 * dy
                    c1 = c10 * (1 - dy) + c11 * dy

                    c = c0 * (1 - dz) + c1 * dz

                    return c
                
                from vedo import utils

                def interpolate_with_local_frames(path_points, step=1.0):
                    """
                    Interpole la courbe avec un pas régulier en distance curviligne,
                    et calcule une base locale (orthonormée) pour chaque point interpolé.

                    Args:
                        path_points (list of (z, y, x)): points de la courbe.
                        step (float): distance entre les points interpolés.

                    Returns:
                        list of dicts with keys:
                            - point: (z, y, x)
                            - axis0: vecteur orthogonal 1 (ex: "horizontal")
                            - axis1: vecteur orthogonal 2 (ex: "vertical")
                            - axis2: vecteur directeur (tangente)
                    """

                    if len(path_points) < 2:
                        return []

                    # Convertir en np.array en (x, y, z) pour faciliter calculs
                    path_xyz = [np.array([x, y, z]) for z, y, x in path_points]

                    # Calcul des distances cumulées le long de la courbe
                    cumulative_d = [0.0]
                    for i in range(1, len(path_xyz)):
                        d = np.linalg.norm(path_xyz[i] - path_xyz[i - 1])
                        cumulative_d.append(cumulative_d[-1] + d)

                    total_length = cumulative_d[-1]
                    sample_distances = np.arange(0, total_length + step, step)

                    xs = [pt[0] for pt in path_xyz]
                    ys = [pt[1] for pt in path_xyz]
                    zs = [pt[2] for pt in path_xyz]

                    x_interp = np.interp(sample_distances, cumulative_d, xs)
                    y_interp = np.interp(sample_distances, cumulative_d, ys)
                    z_interp = np.interp(sample_distances, cumulative_d, zs)

                    points_interp = [np.array([x, y, z]) for x, y, z in zip(x_interp, y_interp, z_interp)]

                    results = []
                    for i in range(len(points_interp)):
                        p = points_interp[i]

                        # Tangente locale (axis2)
                        if i == 0:
                            v = points_interp[i + 1] - p
                        elif i == len(points_interp) - 1:
                            v = p - points_interp[i - 1]
                        else:
                            v = points_interp[i + 1] - points_interp[i - 1]

                        axis2 = utils.versor(v)

                        # Choix d'un vecteur de référence pour la base locale
                        ref = np.array([1, 0, 0]) if abs(axis2[0]) < min(abs(axis2[1]), abs(axis2[2])) \
                            else np.array([0, 1, 0]) if abs(axis2[1]) < abs(axis2[2]) else np.array([0, 0, 1])

                        axis0 = utils.versor(np.cross(ref, axis2))
                        axis1 = utils.versor(np.cross(axis2, axis0))

                        # Retour à (z, y, x) en int pour point
                        point_int = tuple(map(int, (p[2], p[1], p[0])))

                        results.append({
                            'point': point_int,
                            'axis0': axis0,
                            'axis1': axis1,
                            'axis2': axis2
                        })

                    return results
                def extract_root_slices():
                    dir_path = QFileDialog.getExistingDirectory(caption="Dossier des slices racines")
                    if not dir_path:
                        return

                    half = 64  # moitié taille de la coupe en pixels
                    for idx, (_, _, path) in enumerate(all_paths_recal_root):
                        root_dir = Path(dir_path) / f"root_{idx}"
                        root_dir.mkdir(parents=True, exist_ok=True)

                        # Calcul de la base locale sur le chemin (à adapter si besoin)
                        frames = interpolate_with_local_frames(path, step=1.0)

                        for frame in frames:
                            z, y, x = frame['point']
                            axis0, axis1, axis2 = frame['axis0'], frame['axis1'], frame['axis2']

                            # Vérifier les bornes (volume.shape = (Z, Y, X))
                            if not (0 <= z <volume.shape[0]):
                                continue

                            # Construire un plan 2D dans le volume autour de (z,y,x) orthogonal à axis2
                            # Exemple : créer une grille 2D autour de (0,0) dans le repère local (axis0, axis1)
                            yy, xx = np.meshgrid(np.arange(-half, half), np.arange(-half, half), indexing='ij')

                            # Points du patch dans le repère global
                            coords = (np.array([x, y, z]) +
                                    xx[..., None] * axis0 +
                                    yy[..., None] * axis1)  # forme (patch_size, patch_size, 3)

                            # coords sont en (X, Y, Z) flottants — il faut les réorganiser et interpoler dans volume

                            # Réorganiser coords pour accès volume : volume[z, y, x]
                            sample_coords = np.stack([coords[..., 2], coords[..., 1], coords[..., 0]], axis=0)  # (3, H, W)

                            # Interpolation trilineaire pour extraire la slice orientée
                            slice_ = trilinear_interpolation(volume, sample_coords)

                            # Sauvegarder la slice
                            imwrite(str(root_dir / f"{z}_{y}_{x}.tif"), slice_.astype(volume.dtype))

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

        after_segmentation()  # <-- On appelle ici pour construire le layout AVANT d'ajouter le dock

        # Ajoute le dock AVANT que la fonction segment_volume_widget se termine
        viewer.window.add_dock_widget(button_container, name="📥 Exporter les résultats")

    return segment_volume_widget
