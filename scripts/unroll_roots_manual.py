"""
Déroulé virtuel manuel d'un nombre quelconque de racines.

Lancement :
    export QT_QPA_PLATFORM=xcb   # si nécessaire sous Linux
    python scripts/unroll_roots_manual.py

Le script démarre napari VIDE. Le premier bouton du panneau de droite
permet d'ouvrir un volume TIFF. Le dossier de sortie est automatiquement
placé à côté de l'image ouverte (sous-dossier "_unrolled"), mais il peut
être changé à tout moment via un second bouton.
"""

from pathlib import Path
import json
import numpy as np
import napari
import tifffile as tiff
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox,
    QDoubleSpinBox, QFileDialog, QMessageBox, QListWidget, QListWidgetItem,
    QGroupBox, QFormLayout, QLineEdit,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor

from rice_atlas.reconstruction_of_atlas.slice_extraction import (
    extract_and_save_slices,
)

HALF_SIZE = 64
STEP      = 1.0

PALETTE = [
    "#e6194B", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def main():
    viewer = napari.Viewer()

    # État mutable partagé entre les callbacks
    state = {
        "volume": None,
        "volume_path": None,
        "output_dir": None,
    }
    root_layers: list = []

    # =========================================================================
    # PANNEAU
    # =========================================================================
    panel = QWidget()
    panel.setMinimumWidth(370)
    root_layout = QVBoxLayout(panel)

    # --- 1. Ouverture du volume -------------------------------------------
    open_box = QGroupBox("1 · Volume")
    open_lay = QVBoxLayout(open_box)

    btn_open = QPushButton("📂  Ouvrir un volume TIFF…")
    btn_open.setMinimumHeight(36)
    open_lay.addWidget(btn_open)

    lbl_volume = QLabel("Aucun volume chargé.")
    lbl_volume.setWordWrap(True)
    lbl_volume.setStyleSheet("color: #888; font-size: 11px;")
    open_lay.addWidget(lbl_volume)

    root_layout.addWidget(open_box)

    # --- 2. Dossier de sortie --------------------------------------------
    out_box = QGroupBox("2 · Dossier de sortie")
    out_lay = QVBoxLayout(out_box)

    out_path_edit = QLineEdit()
    out_path_edit.setPlaceholderText("(auto : à côté de l'image, sous-dossier _unrolled)")
    out_path_edit.setReadOnly(True)
    out_lay.addWidget(out_path_edit)

    btn_out = QPushButton("🗂️  Choisir un autre dossier…")
    out_lay.addWidget(btn_out)

    root_layout.addWidget(out_box)

    # --- 3. Instructions --------------------------------------------------
    help_box = QGroupBox("3 · Mode d'emploi")
    help_lay = QVBoxLayout(help_box)
    help_label = QLabel(
        "<ol>"
        "<li>Ouvrez un volume (bouton ci-dessus).</li>"
        "<li><b>+ Ajouter une racine</b> crée un calque de points.</li>"
        "<li>Sélectionnez la racine dans la liste : son calque devient actif "
        "et l'outil <i>Add points</i> est armé automatiquement.</li>"
        "<li>Naviguez en Z (curseur du bas) et <b>cliquez au centre</b> "
        "de la racine, de la <b>pointe</b> vers la base.</li>"
        "<li>Répétez pour chaque racine.</li>"
        "<li>Cliquez <b>🌀 DÉROULER TOUT</b>.</li>"
        "</ol>"
        "<i>Supprimer un point : mode <b>Select</b> (icône flèche) → "
        "clic → touche Suppr.</i>"
    )
    help_label.setWordWrap(True)
    help_label.setTextFormat(Qt.RichText)
    help_lay.addWidget(help_label)
    root_layout.addWidget(help_box)

    # --- 4. Liste des racines --------------------------------------------
    list_box = QGroupBox("4 · Racines")
    list_lay = QVBoxLayout(list_box)

    list_widget = QListWidget()
    list_widget.setSelectionMode(QListWidget.SingleSelection)
    list_lay.addWidget(list_widget)

    btns_row = QHBoxLayout()
    btn_add = QPushButton("➕ Ajouter une racine")
    btn_del = QPushButton("🗑️ Supprimer")
    btn_add.setEnabled(False)   # désactivé tant qu'aucun volume n'est chargé
    btn_del.setEnabled(False)
    btns_row.addWidget(btn_add)
    btns_row.addWidget(btn_del)
    list_lay.addLayout(btns_row)

    status_label = QLabel("0 racine.")
    status_label.setStyleSheet("color: #888;")
    list_lay.addWidget(status_label)
    root_layout.addWidget(list_box)

    # --- 5. Paramètres ---------------------------------------------------
    param_box = QGroupBox("5 · Paramètres du déroulé")
    form = QFormLayout(param_box)
    sp_half = QSpinBox(); sp_half.setRange(8, 512); sp_half.setValue(HALF_SIZE)
    sp_step = QDoubleSpinBox(); sp_step.setRange(0.1, 20.0); sp_step.setSingleStep(0.5); sp_step.setValue(STEP)
    form.addRow("Demi-largeur (vx)", sp_half)
    form.addRow("Pas le long du chemin (vx)", sp_step)
    root_layout.addWidget(param_box)

    # --- 6. Sauvegarde / chargement des points ---------------------------
    io_box = QGroupBox("6 · Points de contrôle")
    io_lay = QHBoxLayout(io_box)
    btn_save_pts = QPushButton("💾 Sauver points")
    btn_load_pts = QPushButton("📂 Charger points")
    io_lay.addWidget(btn_save_pts)
    io_lay.addWidget(btn_load_pts)
    root_layout.addWidget(io_box)

    # --- 7. Action principale --------------------------------------------
    btn_run = QPushButton("🌀  DÉROULER TOUT")
    btn_run.setMinimumHeight(48)
    btn_run.setStyleSheet("font-weight: bold; font-size: 14px;")
    btn_run.setEnabled(False)
    root_layout.addWidget(btn_run)

    root_layout.addStretch()

    # =========================================================================
    # LOGIQUE
    # =========================================================================
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

    # Ouverture du volume
    def open_volume():
        path, _ = QFileDialog.getOpenFileName(
            None, "Ouvrir un volume TIFF", "",
            "Fichiers TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        path = Path(path)
        print(f"📂 Chargement : {path}")
        try:
            vol = tiff.imread(str(path))
        except Exception as e:
            QMessageBox.critical(None, "Erreur de lecture", str(e))
            return
        print(f"   shape={vol.shape}, dtype={vol.dtype}")

        # Supprimer l'ancien calque volume si présent
        for layer in list(viewer.layers):
            if layer.name == "Volume":
                viewer.layers.remove(layer)
                break

        state["volume"] = vol
        state["volume_path"] = path
        # Dossier de sortie par défaut : à côté de l'image
        auto_out = path.parent / (path.stem + "_unrolled")
        state["output_dir"] = auto_out
        out_path_edit.setText(str(auto_out))

        viewer.add_image(vol, name="Volume", colormap="gray")

        lbl_volume.setText(f"{path.name}  —  {vol.shape}  {vol.dtype}")
        lbl_volume.setStyleSheet("color: #ccc; font-size: 11px;")
        btn_add.setEnabled(True)
        btn_run.setEnabled(True)

    def choose_output_dir():
        d = QFileDialog.getExistingDirectory(
            None, "Choisir le dossier de sortie",
            str(state["output_dir"] or Path.home())
        )
        if d:
            state["output_dir"] = Path(d)
            out_path_edit.setText(d)

    def add_root():
        if state["volume"] is None:
            return
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

        def _on_data_change(event, i=idx):
            update_list_item(i)
        layer.events.data.connect(_on_data_change)

        root_layers.append(layer)
        item = QListWidgetItem(f"{name}  —  0 pt")
        item.setForeground(color_brush(color))
        list_widget.addItem(item)
        list_widget.setCurrentRow(idx)
        btn_del.setEnabled(True)
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
        for i, l in enumerate(root_layers):
            try:
                l.name = f"Racine {i+1}"
            except Exception:
                pass
            list_widget.item(i).setText(
                f"{l.name}  —  {len(np.asarray(l.data))} pt"
            )
        if not root_layers:
            btn_del.setEnabled(False)
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
        if state["volume"] is None:
            QMessageBox.warning(None, "Pas de volume", "Ouvrez d'abord un volume.")
            return
        if not root_layers:
            QMessageBox.warning(None, "Rien à dérouler", "Ajoutez au moins une racine.")
            return
        output_dir = state["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

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
                    volume=state["volume"],
                    centerline=centerline,
                    output_dir=output_dir,
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
            stack_path = output_dir / f"{organ_id}_straightened.tif"
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
        default = str(
            (state["output_dir"] or Path(".")) / "control_points.json"
        )
        path, _ = QFileDialog.getSaveFileName(
            None, "Sauver les points", default, "JSON (*.json)"
        )
        if not path:
            return
        data = {l.name: np.asarray(l.data).tolist() for l in root_layers}
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"💾 {path}")

    def load_points():
        path, _ = QFileDialog.getOpenFileName(
            None, "Charger les points",
            str(state["output_dir"] or Path(".")), "JSON (*.json)"
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

    # Branchements
    btn_open.clicked.connect(open_volume)
    btn_out.clicked.connect(choose_output_dir)
    btn_add.clicked.connect(add_root)
    btn_del.clicked.connect(remove_selected_root)
    list_widget.currentRowChanged.connect(lambda _: on_selection_changed())
    btn_run.clicked.connect(run_unroll)
    btn_save_pts.clicked.connect(save_points)
    btn_load_pts.clicked.connect(load_points)

    viewer.window.add_dock_widget(panel, name="Déroulé", area="right")
    napari.run()


if __name__ == "__main__":
    main()
