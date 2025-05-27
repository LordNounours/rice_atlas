from typing import TYPE_CHECKING
import numpy as np
from magicgui import magic_factory
from qtpy.QtWidgets import QPushButton, QFileDialog, QWidget, QVBoxLayout
from rice_atlas.predictor import segment_volume
from tifffile import imwrite, imread
import os

if TYPE_CHECKING:
    import napari

# Dictionnaire global pour stocker la référence des boutons
save_button_ref = {}

@magic_factory(
    model_path={"widget_type": "FileEdit", "label": "Chemin du modèle", "mode": "r"},
    volume_path={"widget_type": "FileEdit", "label": "Volume à segmenter", "mode": "r"},
    output_path={"widget_type": "FileEdit", "label": "Fichier de sortie (optionnel)", "nullable": True, "mode": "w"},
    patch_size={"label": "Taille du patch", "min": 16, "max": 256, "step": 16},
    stride={"label": "Stride", "min": 8, "max": 256, "step": 8},
    batch_size={"label": "Taille du batch", "min": 1, "max": 64, "step": 1},
    tap_x={"label": "Centre X", "min": 0, "max": 18000, "step": 1},
    tap_y={"label": "Centre Y", "min": 0, "max": 1200, "step": 1},
    tap_z={"label": "Centre Z", "min": 0, "max": 1200, "step": 1},
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
    for btn_key in ["save_button_proba", "save_button_segmented"]:
        if btn_key in save_button_ref:
            viewer.window.remove_dock_widget(save_button_ref[btn_key])

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

    # ➕ Ajouter le widget conteneur à l’UI napari
    viewer.window.add_dock_widget(button_container, name="📥 Exporter les résultats")

    # 🔁 Rafraîchissement de l'affichage du volume original
    def on_volume_path_change(new_path):
        if viewer is not None and new_path and os.path.exists(new_path):
            print("📂 Volume original changé")
            for layer in list(viewer.layers):
                if layer.name == "Volume original":
                    viewer.layers.remove(layer)
            volume = imread(new_path)
            viewer.add_image(volume, name="Volume original")

    segment_volume_widget.volume_path.changed.connect(on_volume_path_change)
