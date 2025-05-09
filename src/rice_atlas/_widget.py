from typing import TYPE_CHECKING
import numpy as np
from magicgui import magic_factory
from qtpy.QtWidgets import QPushButton, QFileDialog
from rice_atlas.predictor import segment_volume
from tifffile import imwrite

if TYPE_CHECKING:
    import napari

# Dictionnaire global pour stocker la référence du bouton de sauvegarde
save_button_ref = {}

@magic_factory(
    model_path={"widget_type": "FileEdit", "label": "Chemin du modèle", "mode": "r"},
    volume_path={"widget_type": "FileEdit", "label": "Volume à segmenter", "mode": "r"},
    output_path={"widget_type": "FileEdit", "label": "Fichier de sortie (optionnel)", "nullable": True, "mode": "w"},
    patch_size={"label": "Taille du patch", "min": 16, "max": 256, "step": 16},
    stride={"label": "Stride", "min": 8, "max": 256, "step": 8},
    batch_size={"label": "Taille du batch", "min": 1, "max": 64, "step": 1},
    pretreatment={"label": " Prétraitement ", "widget_type": "CheckBox", "value": False},  
)
def segment_volume_widget(
    model_path: str,
    volume_path: str,
    output_path: str = None,
    patch_size: int = 128,
    stride: int = 96,
    batch_size: int = 16,
    pretreatment = False,
    viewer: "napari.viewer.Viewer" = None,
) -> None:
    """Segment a 3D TIFF volume using a 3D SegFormer model and display result."""
    # Exécuter la segmentation
    segmented = segment_volume(
        model_path=model_path,
        volume_path=volume_path,
        output_path=output_path,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        pretreatment=pretreatment
    )

    if viewer is not None:
        # Afficher le volume segmenté dans napari
        viewer.add_labels(segmented, name="Segmentation")

    # Fonction de sauvegarde
    def save_predicted_volume():
        # Ouvrir l'explorateur de fichiers pour choisir le chemin de sauvegarde
        save_path, _ = QFileDialog.getSaveFileName(
            caption="Enregistrer la prédiction",
            filter="Fichiers TIFF (*.tiff *.tif)",
        )
        
        if save_path:
            # Sauvegarder la prédiction sous forme de fichier TIFF
            segmented_to_save = (segmented * 255).astype(np.uint8)
            imwrite(save_path, segmented_to_save)
            print(f"Prédiction sauvegardée à : {save_path}")

    # Si un bouton de sauvegarde existe déjà, on le remplace
    if "save_button" in save_button_ref:
        # Retirer l'ancien bouton de sauvegarde
        viewer.window.remove_dock_widget(save_button_ref["save_button"])

    # Créer un nouveau bouton de sauvegarde
    save_button = QPushButton("Sauvegarder")
    save_button.clicked.connect(save_predicted_volume)
    
    # Ajouter le bouton à l'interface de ton widget
    viewer.window.add_dock_widget(save_button)

    # Mettre à jour la référence du bouton
    save_button_ref["save_button"] = save_button
