"""
Déroulé virtuel manuel d'un nombre quelconque de racines.

Lancement :
    export QT_QPA_PLATFORM=xcb   # si nécessaire
    python scripts/unroll_roots_manual.py

Mode d'emploi visuel directement dans le panneau de droite ("Déroulé").
"""

from pathlib import Path
import json
import numpy as np
import napari
import tifffile as tiff
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox,
    QDoubleSpinBox, QFileDialog, QMessageBox, QListWidget, QListWidgetItem,
    QGroupBox, QFormLayout,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor

from rice_atlas.reconstruction_of_atlas.slice_extraction import (
    extract_and_save_slices,
)

# ---------------------------------------------------------------------------
# PARAMÈTRES PAR DÉFAUT (modifiables ensuite dans l'interface)
# ---------------------------------------------------------------------------
VOLUME_PATH = Path("/media/rfernandez/Crucial X9/SunRice/Hollow/Test01/Raw/Test01.tif")
OUTPUT_DIR  = Path("/media/rfernandez/Crucial X9/SunRice/Hollow/Test01/Unrolled")
HALF_SIZE   = 64
STEP        = 1.0

PALETTE = [
    "#e6194B", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def load_volume(path: Path) -> np.ndarray:
    print(f"📂 Chargement : {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    vol = tiff.imread(str(path))
    print(f"   shape={vol.shape}, dtype={vol.dtype}")
    return vol


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    volume = load_volume(VOLUME_PATH)

    viewer = napari.Viewer()
    viewer.add_image(volume, name="Volume", colormap="gray")

    root_layers: list = []

    # =====================================================================
    # PANNEAU
    # =====================================================================
    panel = QWidget()
    panel.setMinimumWidth(360)
    root_layout = QVBoxLayout(panel)

    # Instructions ---------------------------------------------------------
    help_box = QGroupBox("Mode d'emploi")
    help_lay = QVBoxLayout(help_box)
    help_label = QLabel(
        "<ol>"
        "<li><b>+ Ajouter une racine</b> crée un calque de points.</li>"
        "<li>Sélectionnez la racine dans la liste : son calque devient actif "
        "et l'outil <i>Add points</i> est armé automatiquement.</li>"
        "<li>Naviguez en Z (curseur du bas) et <b>cliquez au centre</b> "
        "de la racine. Commencez par la <b>pointe</b>.</li>"
        "<li>Répétez pour chaque racine (ajoutez-en autant que vous voulez).</li>"
        "<li>Cliquez <b>🌀 DÉROULER TOUT</b>.</li>"
        "</ol>"
        "<i>Pour supprimer un point : passer en mode <b>Select</b> (icône "
        "flèche du calque), cliquer le point, touche <b>Suppr</b>.</i>"
    )
    help_label.setWordWrap(True)
    help_label.setTextFormat(Qt.RichText)
    help_lay.addWidget(help_label)
    root_layout.addWidget(help_box)

    # Liste des racines ----------------------------------------------------
    list_box = QGroupBox("Racines")
    list_lay = QVBoxLayout(list_box)
    list_widget = QListWidget()
    list_widget.setSelectionMode(QListWidget.SingleSelection)
    list_lay.addWidget(list_widget)

    btns_row = QHBoxLayout()
    btn_add = QPushButton("➕ Ajouter une racine")
    btn_del = QPushButton("🗑️ Supprimer")
    btns_row.addWidget(btn_add)
    btns_row.addWidget(btn_del)
    list_lay.addLayout(btns_row)

    status_label = QLabel("0 racine.")
    status_label.setStyleSheet("color: #888;")
    list_lay.addWidget(status_label)
    root_layout.addWidget(list_box)

    # Paramètres -----------------------------------------------------------
    param_box = QGroupBox("Paramètres du déroulé")
    form = QFormLayout(param_box)
    sp_half = QSpinBox(); sp_half.setRange(8, 512); sp_half.setValue(HALF_SIZE)
    sp_step = QDoubleSpinBox(); sp_step.setRange(0.1, 20.0); sp_step.setSingleStep(0.5); sp_step.setValue(STEP)
    form.addRow("Demi-largeur (vx)", sp_half)
    form.addRow("Pas le long du chemin (vx)", sp_step)
    root_layout.addWidget(param_box)

    # I/O points -----------------------------------------------------------
    io_box = QGroupBox("Points de contrôle")
    io_lay = QHBoxLayout(io_box)
    btn_save_pts = QPushButton("💾 Sauver")
    btn_load_pts = QPushButton("📂 Charger")
    io_lay.addWidget(btn_save_pts)
    io_lay.addWidget(btn_load_pts)
    root_layout.addWidget(io_box)

    # Action ---------------------------------------------------------------
    btn_run = QPushButton("🌀  DÉROULER TOUT")
    btn_run.setMinimumHeight(48)
    btn_run.setStyleSheet("font-weight: bold; font-size: 14px;")
    root_layout.addWidget(btn_run)

    root_layout.addStretch()

    # =====================================================================
    # LOGIQUE
    # =====================================================================
    def color_brush(hex_color: str) -> QBrush:
        return QBrush(QColor(hex_color))

    def refresh_status():
        n = len(root_layers)
        total = sum(len(np.asarray(l.data)) for l in root_layers)
        status_label.setText(f"{n} racine(s) — {total} point(s) au total.")

    def update_list_item(i: int):
        if 0 <= i < len(root_layers):
            n_pts = len(np.asarray(root_layers[i].data))
            list_widget.item(i).setText(f"{root_layers[i].name}  —  {n_pts} pt")
            refresh_status()

    def add_root():
        idx = len(root_layers)
        color = PALETTE[idx % len(PALETTE)]
        name = f"Racine {idx + 1}"
        layer = viewer.add_points(
            np.empty((0, 3)),
            name=name,
            size=8,
            face_color=color,
            border_color=color,
            ndim=3,
            symbol="o",
        )
        try:
            layer.mode = "add"
        except Exception:
            pass

        # Mise à jour automatique du compteur quand on ajoute/enlève des points
        def _on_data_change(event, i=idx):
            update_list_item(i)
        layer.events.data.connect(_on_data_change)

        root_layers.append(layer)
        item = QListWidgetItem(f"{name}  —  0 pt")
        item.setForeground(color_brush(color))
        list_widget.addItem(item)
        list_widget.setCurrentRow(idx)
        refresh_status()

    def remove_selected_root():
        row = list_widget.currentRow()
        if row < 0 or row >= len(root_layers):
            return
        if QMessageBox.question(
            None, "Supprimer ?",
            f"Supprimer {root_layers[row].name} et ses points ?",
            QMessageBox.Yes | QMessageBox.No
        ) != QMessageBox.Yes:
            return
        layer = root_layers.pop(row)
        try:
            viewer.layers.remove(layer)
        except Exception:
            pass
        list_widget.takeItem(row)
        # Renommer les suivants
        for i, l in enumerate(root_layers):
            try:
                l.name = f"Racine {i+1}"
            except Exception:
                pass
            list_widget.item(i).setText(
                f"{l.name}  —  {len(np.asarray(l.data))} pt"
            )
        refresh_status()

    def on_selection_changed():
        row = list_widget.currentRow()
        if 0 <= row < len(root_layers):
            layer = root_layers[row]
            viewer.layers.selection.active = layer
            try:
                layer.mode = "add"
            except Exception:
                pass

    def points_to_centerline(points_zyx: np.ndarray) -> np.ndarray:
        if len(points_zyx) < 2:
            return np.empty((0, 3))
        idx = np.argsort(points_zyx[:, 0])
        return points_zyx[idx].astype(np.float64)

    def run_unroll():
        if not root_layers:
            QMessageBox.warning(None, "Rien à dérouler", "Ajoutez au moins une racine.")
            return
        half = sp_half.value()
        step = sp_step.value()
        produced, skipped = [], []
        for i, layer in enumerate(root_layers):
            pts = np.asarray(layer.data)
            if len(pts) < 2:
                skipped.append(f"{layer.name} (< 2 points)")
                continue
            centerline = points_to_centerline(pts)
            organ_id = f"racine_{i+1:02d}"
            print(f"\n▶️ {organ_id} ({len(centerline)} points)")
            try:
                meta = extract_and_save_slices(
                    volume=volume,
                    centerline=centerline,
                    output_dir=OUTPUT_DIR,
                    organ_id=organ_id,
                    organ_type="root",
                    half_size=half,
                    step=step,
                    save_3d_stack=True,
                    save_individual=False,
                    save_metadata=True,
                )
            except Exception as e:
                print(f"❌ {organ_id} : {e}")
                skipped.append(f"{layer.name} (erreur : {e})")
                continue
            print(f"   {meta.num_slices} coupes, longueur {meta.centerline_length:.1f} vx")
            stack_path = OUTPUT_DIR / f"{organ_id}_straightened.tif"
            if stack_path.exists():
                stack = tiff.imread(str(stack_path))
                viewer.add_image(stack, name=f"Déroulé {organ_id}")
                produced.append(str(stack_path))

        msg = ""
        if produced:
            msg += "Fichiers générés :\n" + "\n".join(produced)
        if skipped:
            msg += ("\n\n" if msg else "") + "Ignorées :\n" + "\n".join(skipped)
        QMessageBox.information(None, "Déroulé terminé", msg or "(rien)")

    def save_points():
        path, _ = QFileDialog.getSaveFileName(
            None, "Sauver les points",
            str(OUTPUT_DIR / "control_points.json"), "JSON (*.json)"
        )
        if not path:
            return
        data = {l.name: np.asarray(l.data).tolist() for l in root_layers}
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"💾 {path}")

    def load_points():
        path, _ = QFileDialog.getOpenFileName(
            None, "Charger les points", str(OUTPUT_DIR), "JSON (*.json)"
        )
        if not path:
            return
        data = json.loads(Path(path).read_text())
        for l in list(root_layers):
            try:
                viewer.layers.remove(l)
            except Exception:
                pass
        root_layers.clear()
        list_widget.clear()
        for _name, pts in data.items():
            add_root()
            if len(pts):
                root_layers[-1].data = np.array(pts)
        for i in range(len(root_layers)):
            update_list_item(i)
        print(f"📂 {path}")

    btn_add.clicked.connect(add_root)
    btn_del.clicked.connect(remove_selected_root)
    list_widget.currentRowChanged.connect(lambda _: on_selection_changed())
    btn_run.clicked.connect(run_unroll)
    btn_save_pts.clicked.connect(save_points)
    btn_load_pts.clicked.connect(load_points)

    viewer.window.add_dock_widget(panel, name="Déroulé", area="right")
    add_root()  # commencer avec une racine prête à recevoir des points

    napari.run()


if __name__ == "__main__":
    main()
