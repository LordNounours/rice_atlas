"""
Analyse manuelle des cortical bridges (et de la surintensité iodée) sur une
racine redressée.

Lancement :
    export QT_QPA_PLATFORM=xcb   # si nécessaire sous Linux
    python scripts/analyze_cortical_bridges.py

Le script démarre napari VIDE. Le panneau de droite permet :

  1. d'ouvrir une image de racine redressée (pile TIFF 3D Z, Y, X) ;
  2. d'indiquer s'il s'agit d'une racine SANS ou AVEC marquage iode ;
  3. de placer le CENTRE de la stèle et de régler quatre cercles concentriques
     (diamètre extérieur racine, diamètre extérieur stèle, et deux cercles
     dérivés à 85 % / 115 %) — par glisser-déposer des poignées ou via les
     champs numériques. La navigation en Z (CTRL + molette de napari) permet de
     vérifier les cercles sur toutes les coupes : ils restent visibles partout.
  4. de segmenter par seuillage (Otsu automatique ou valeur manuelle) la zone
     comprise entre les deux cercles dérivés = zone des cortical bridges, et de
     mesurer coupe par coupe la proportion de cortical bridge (clair) vs
     aérenchyme (sombre) ;
  5. (option iode) de seuiller l'intérieur de la stèle pour discriminer les
     tissus hydratés des tissus iodés, d'échantillonner la valeur de base d'un
     tissu hydraté, de générer une image de surintensité et de mesurer coupe par
     coupe le xylème segmenté, le nombre de composantes et la surintensité
     moyenne.

Les CSV sont enregistrés à côté de l'image source, en reprenant son nom suivi
d'un suffixe explicite :
    <image>_geometry.csv          (une ligne : centre + rayons + seuils)
    <image>_cortical_bridges.csv  (une ligne par coupe)
    <image>_iodine.csv            (une ligne par coupe, uniquement si iode)
"""

from pathlib import Path
import json
import numpy as np
import napari
import tifffile as tiff
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox,
    QDoubleSpinBox, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QLineEdit, QRadioButton, QButtonGroup, QScrollArea,
)
from qtpy.QtCore import Qt

# --- Couleurs des objets géométriques ----------------------------------------
COL_CENTER = "#ff2020"   # centre
COL_ROOT   = "#00ff00"   # cercle extérieur racine (vert)
COL_STELE  = "#ff00ff"   # cercle extérieur stèle (magenta)
COL_CBO    = "#ffff00"   # cercle cortical bridge externe = 85 % racine (jaune)
COL_CBI    = "#00ffff"   # cercle cortical bridge interne = 115 % stèle (cyan)

# Ordre des rayons : [racine, stèle, cb_externe(85%), cb_interne(115%)]
CIRCLE_COLORS = [COL_ROOT, COL_STELE, COL_CBO, COL_CBI]
HANDLE_COLORS = [COL_CENTER, COL_ROOT, COL_STELE, COL_CBO, COL_CBI]
# Angles d'affichage des poignées (en radians) : E, N, O, S
HANDLE_ANGLES = np.deg2rad([0.0, 90.0, 180.0, 270.0])

SAMPLE_RADIUS = 5        # rayon (vx) du disque d'échantillonnage tissu hydraté
IODINE_OFFSET = 10       # décalage seuil pour la surintensité (seuil + 10)


# =============================================================================
# FONCTIONS DE CALCUL PURES (testables sans interface)
# =============================================================================
def make_masks(shape_yx, center_yx, radii):
    """Construit les masques 2D (H, W) à partir du centre et des 4 rayons.

    radii = (r_root, r_stele, r_cb_outer, r_cb_inner)
    Retourne (dist, annulus, stele_disk, root_disk).
    """
    h, w = shape_yx
    cy, cx = center_yx
    r_root, r_stele, r_cbo, r_cbi = radii
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(yy - cy, xx - cx)
    annulus = (dist >= r_cbi) & (dist <= r_cbo)
    stele_disk = dist <= r_stele
    root_disk = dist <= r_root
    return dist, annulus, stele_disk, root_disk


def cortical_ratio_per_slice(volume, annulus, threshold):
    """Proportion (%) de cortical bridge (pixels clairs) dans l'anneau.

    ratio = pixels_clairs / pixels_anneau * 100, par coupe.
    """
    n_ann = int(annulus.sum())
    ratios = np.zeros(len(volume), dtype=float)
    if n_ann == 0:
        return ratios
    for z in range(len(volume)):
        bright = (volume[z] > threshold) & annulus
        ratios[z] = 100.0 * bright.sum() / n_ann
    return ratios


def iodine_metrics_per_slice(volume, stele_disk, threshold, hydrated_value):
    """Mesures de surintensité iodée coupe par coupe.

    Masque de surintensité (xylème) = intérieur stèle ET intensité >
    (threshold + OFFSET) ; sert au ratio de surface et au comptage de
    composantes 4-connexes.
    La surintensité MOYENNE est calculée sur TOUTE la stèle : somme de
    (pixel - hydrated_value) sur tous les pixels de la stèle, divisée par le
    nombre de pixels de la stèle.
    Retourne (xylem_ratio_pct, n_components, mean_overintensity).
    """
    n_stele = int(stele_disk.sum())
    cross = ndi.generate_binary_structure(2, 1)  # 4-connexité
    n = len(volume)
    xylem_ratio = np.zeros(n, dtype=float)
    n_comp = np.zeros(n, dtype=int)
    mean_over = np.zeros(n, dtype=float)
    over_thr = threshold + IODINE_OFFSET
    for z in range(n):
        mask = stele_disk & (volume[z] > over_thr)
        cnt = int(mask.sum())
        if n_stele > 0:
            xylem_ratio[z] = 100.0 * cnt / n_stele
            mean_over[z] = float(
                np.sum(volume[z][stele_disk].astype(float) - hydrated_value)
                / n_stele
            )
        if cnt > 0:
            _, n_comp[z] = ndi.label(mask, structure=cross)
    return xylem_ratio, n_comp, mean_over


# =============================================================================
# INTERFACE
# =============================================================================
def main():
    viewer = napari.Viewer()

    state = {
        "image": None,          # volume original (Z, Y, X)
        "image_f": None,        # volume en float
        "image_path": None,
        "iodine": False,
        "center": None,         # np.array([cy, cx])
        "radii": None,          # np.array([r_root, r_stele, r_cbo, r_cbi])
        "cb_threshold": None,
        "iodine_threshold": None,
        "hydrated": None,
        "pts_prev": None,       # positions précédentes des 5 points de contrôle
        "sync": False,          # garde anti-récursion poignées <-> spinbox
    }

    layers = {"image": None, "ctrl": None, "circles": None,
              "cb_labels": None, "iodine_labels": None,
              "over": None, "hydr_pts": None}

    # -------------------------------------------------------------------------
    # PANNEAU (dans une zone scrollable)
    # -------------------------------------------------------------------------
    panel = QWidget()
    panel.setMinimumWidth(390)
    root_layout = QVBoxLayout(panel)

    # --- 1. Image --------------------------------------------------------
    img_box = QGroupBox("1 · Image (racine redressée)")
    img_lay = QVBoxLayout(img_box)
    btn_open = QPushButton("📂  Ouvrir une image TIFF…")
    btn_open.setMinimumHeight(34)
    img_lay.addWidget(btn_open)
    lbl_img = QLabel("Aucune image chargée.")
    lbl_img.setWordWrap(True)
    lbl_img.setStyleSheet("color: #888; font-size: 11px;")
    img_lay.addWidget(lbl_img)
    form_scale = QFormLayout()
    sp_voxel = QDoubleSpinBox()
    sp_voxel.setDecimals(4); sp_voxel.setRange(0.0001, 1e6); sp_voxel.setValue(1.0)
    sp_step = QDoubleSpinBox()
    sp_step.setDecimals(4); sp_step.setRange(0.0001, 1e6); sp_step.setValue(1.0)
    form_scale.addRow("Taille voxel (unité/vx)", sp_voxel)
    form_scale.addRow("Pas inter-coupe (vx)", sp_step)
    img_lay.addLayout(form_scale)
    root_layout.addWidget(img_box)

    # --- 2. Type de racine ----------------------------------------------
    type_box = QGroupBox("2 · Type de racine")
    type_lay = QVBoxLayout(type_box)
    rb_no = QRadioButton("Sans marquage iode")
    rb_yes = QRadioButton("Avec marquage iode")
    rb_no.setChecked(True)
    grp = QButtonGroup(type_box)
    grp.addButton(rb_no); grp.addButton(rb_yes)
    type_lay.addWidget(rb_no)
    type_lay.addWidget(rb_yes)
    root_layout.addWidget(type_box)

    # --- 3. Géométrie ----------------------------------------------------
    geo_box = QGroupBox("3 · Centre & cercles")
    geo_lay = QVBoxLayout(geo_box)
    geo_help = QLabel(
        "Glissez le point <b>rouge</b> pour déplacer le centre, et les poignées "
        "colorées pour régler chaque cercle (ou saisissez les rayons ci-dessous). "
        "Vérifiez sur toutes les coupes avec <b>CTRL + molette</b>."
    )
    geo_help.setWordWrap(True); geo_help.setTextFormat(Qt.RichText)
    geo_help.setStyleSheet("font-size: 11px;")
    geo_lay.addWidget(geo_help)

    def _clbl(color, text):
        w = QLabel(f"<span style='color:{color}; font-size:15px'>\u2b24</span> {text}")
        w.setTextFormat(Qt.RichText)
        return w

    lbl_legend = QLabel(
        "<b>Légende des couleurs :</b><br>"
        f"<span style='color:{COL_CENTER}; font-size:15px'>\u2b24</span> centre "
        "(point central déplaçable)<br>"
        f"<span style='color:{COL_ROOT}; font-size:15px'>\u2b24</span> diamètre ext. racine&nbsp;&nbsp;"
        f"<span style='color:{COL_STELE}; font-size:15px'>\u2b24</span> diamètre ext. stèle<br>"
        f"<span style='color:{COL_CBO}; font-size:15px'>\u2b24</span> CB externe = 85 % racine&nbsp;&nbsp;"
        f"<span style='color:{COL_CBI}; font-size:15px'>\u2b24</span> CB interne = 115 % stèle"
    )
    lbl_legend.setWordWrap(True); lbl_legend.setTextFormat(Qt.RichText)
    lbl_legend.setStyleSheet("font-size: 11px;")
    geo_lay.addWidget(lbl_legend)

    form_geo = QFormLayout()
    sp_r_root = QDoubleSpinBox(); sp_r_root.setDecimals(1); sp_r_root.setRange(1, 100000)
    sp_r_stele = QDoubleSpinBox(); sp_r_stele.setDecimals(1); sp_r_stele.setRange(1, 100000)
    sp_r_cbo = QDoubleSpinBox(); sp_r_cbo.setDecimals(1); sp_r_cbo.setRange(1, 100000)
    sp_r_cbi = QDoubleSpinBox(); sp_r_cbi.setDecimals(1); sp_r_cbi.setRange(1, 100000)
    form_geo.addRow(_clbl(COL_ROOT, "Rayon ext. racine (vx)"), sp_r_root)
    form_geo.addRow(_clbl(COL_STELE, "Rayon ext. stèle (vx)"), sp_r_stele)
    form_geo.addRow(_clbl(COL_CBO, "CB externe — 85 % racine (vx)"), sp_r_cbo)
    form_geo.addRow(_clbl(COL_CBI, "CB interne — 115 % stèle (vx)"), sp_r_cbi)
    geo_lay.addLayout(form_geo)

    lbl_constraint = QLabel(
        "Contrainte : la stèle est limitée à 90 % du rayon racine "
        "(périmètre stèle ≤ périmètre racine − 10 %)."
    )
    lbl_constraint.setWordWrap(True)
    lbl_constraint.setStyleSheet("color: #cc8800; font-size: 11px;")
    geo_lay.addWidget(lbl_constraint)

    row_geo_btn = QHBoxLayout()
    btn_recompute = QPushButton("↺ Recalculer 85 % / 115 %")
    btn_reset_geo = QPushButton("⟲ Réinitialiser")
    row_geo_btn.addWidget(btn_recompute)
    row_geo_btn.addWidget(btn_reset_geo)
    geo_lay.addLayout(row_geo_btn)
    root_layout.addWidget(geo_box)

    # --- 4. Cortical bridges (seuillage) --------------------------------
    cb_box = QGroupBox("4 · Cortical bridges — seuillage")
    cb_lay = QVBoxLayout(cb_box)
    row_cb1 = QHBoxLayout()
    btn_cb_otsu = QPushButton("Déterminer le seuil automatiquement (Otsu)")
    row_cb1.addWidget(btn_cb_otsu)
    cb_lay.addLayout(row_cb1)

    form_cb = QFormLayout()
    sp_cb_thr = QDoubleSpinBox(); sp_cb_thr.setDecimals(2); sp_cb_thr.setRange(-1e9, 1e9)
    sp_cb_step = QDoubleSpinBox(); sp_cb_step.setDecimals(2); sp_cb_step.setRange(0.01, 1e6); sp_cb_step.setValue(1.0)
    form_cb.addRow("Seuil (manuel ou Otsu)", sp_cb_thr)
    form_cb.addRow("Pas des boutons +/-", sp_cb_step)
    cb_lay.addLayout(form_cb)

    row_cb2 = QHBoxLayout()
    btn_cb_minus = QPushButton("➖ seuil")
    btn_cb_plus = QPushButton("➕ seuil")
    btn_cb_validate = QPushButton("✔ Valider le seuil")
    row_cb2.addWidget(btn_cb_minus)
    row_cb2.addWidget(btn_cb_plus)
    row_cb2.addWidget(btn_cb_validate)
    cb_lay.addLayout(row_cb2)
    lbl_cb = QLabel("Aperçu : clair = cortical bridge, sombre = aérenchyme.")
    lbl_cb.setWordWrap(True); lbl_cb.setStyleSheet("color: #888; font-size: 11px;")
    cb_lay.addWidget(lbl_cb)
    root_layout.addWidget(cb_box)

    # --- 5. Iode (surintensité) -----------------------------------------
    io_box = QGroupBox("5 · Marquage iode — surintensité")
    io_lay = QVBoxLayout(io_box)
    btn_io_otsu = QPushButton("Seuil hydraté/iodé (Otsu dans la stèle)")
    io_lay.addWidget(btn_io_otsu)
    form_io = QFormLayout()
    sp_io_thr = QDoubleSpinBox(); sp_io_thr.setDecimals(2); sp_io_thr.setRange(-1e9, 1e9)
    sp_io_step = QDoubleSpinBox(); sp_io_step.setDecimals(2); sp_io_step.setRange(0.01, 1e6); sp_io_step.setValue(1.0)
    form_io.addRow("Seuil discrimination", sp_io_thr)
    form_io.addRow("Pas des boutons +/-", sp_io_step)
    io_lay.addLayout(form_io)
    row_io = QHBoxLayout()
    btn_io_minus = QPushButton("➖ seuil")
    btn_io_plus = QPushButton("➕ seuil")
    btn_io_validate = QPushButton("✔ Valider le seuil")
    row_io.addWidget(btn_io_minus); row_io.addWidget(btn_io_plus); row_io.addWidget(btn_io_validate)
    io_lay.addLayout(row_io)

    btn_io_sample = QPushButton(f"Cliquer une zone hydratée (r={SAMPLE_RADIUS} vx)")
    io_lay.addWidget(btn_io_sample)
    lbl_io_hydr = QLabel("Valeur tissu hydraté : (non mesurée)")
    lbl_io_hydr.setWordWrap(True); lbl_io_hydr.setStyleSheet("font-size: 11px;")
    io_lay.addWidget(lbl_io_hydr)
    btn_io_over = QPushButton("Générer l'image de surintensité")
    io_lay.addWidget(btn_io_over)
    root_layout.addWidget(io_box)
    io_box.setVisible(False)

    # --- 6. Export -------------------------------------------------------
    exp_box = QGroupBox("6 · Export")
    exp_lay = QVBoxLayout(exp_box)
    btn_export = QPushButton("💾  EXPORTER LES CSV")
    btn_export.setMinimumHeight(42)
    btn_export.setStyleSheet("font-weight: bold; font-size: 14px;")
    exp_lay.addWidget(btn_export)
    root_layout.addWidget(exp_box)

    root_layout.addStretch()

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(panel)

    # =========================================================================
    # LOGIQUE
    # =========================================================================
    def has_geometry():
        return state["center"] is not None and state["radii"] is not None

    def get_masks():
        h, w = state["image"].shape[1:]
        return make_masks((h, w), tuple(state["center"]), tuple(state["radii"]))

    # ---- Construction / mise à jour des poignées & cercles --------------
    def build_ctrl_points():
        cy, cx = state["center"]
        radii = state["radii"]
        pts = [[cy, cx]]
        for r, a in zip(radii, HANDLE_ANGLES):
            pts.append([cy - r * np.sin(a), cx + r * np.cos(a)])
        pts = np.array(pts, dtype=float)
        state["pts_prev"] = pts.copy()

        if layers["ctrl"] is None:
            lay = viewer.add_points(
                pts, name="Contrôle (centre + poignées)", ndim=2,
                size=4.9, face_color=HANDLE_COLORS, border_color="#000000",
                symbol="o",
            )
            lay.events.data.connect(on_ctrl_data)
            layers["ctrl"] = lay
        else:
            state["sync"] = True
            layers["ctrl"].data = pts
            layers["ctrl"].face_color = HANDLE_COLORS
            state["sync"] = False
        try:
            layers["ctrl"].mode = "select"
        except Exception:
            pass

    def redraw_circles():
        if not has_geometry():
            return
        cy, cx = state["center"]
        # napari lit un tableau 2x2 comme [centre, rayons] : on passe donc les
        # 4 coins de la boîte englobante (interprétation non ambiguë).
        boxes = [
            np.array([[cy - r, cx - r], [cy - r, cx + r],
                      [cy + r, cx + r], [cy + r, cx - r]])
            for r in state["radii"]
        ]
        if layers["circles"] is None:
            lay = viewer.add_shapes(
                boxes, shape_type="ellipse", name="Cercles",
                edge_color=CIRCLE_COLORS, face_color="transparent",
                edge_width=0.75, ndim=2,
            )
            lay.editable = False
            layers["circles"] = lay
        else:
            lay = layers["circles"]
            lay.data = boxes
            lay.shape_type = ["ellipse"] * len(boxes)
            lay.edge_color = CIRCLE_COLORS
            lay.face_color = "transparent"

    def set_spinboxes_from_radii():
        state["sync"] = True
        sp_r_root.setValue(state["radii"][0])
        sp_r_stele.setValue(state["radii"][1])
        sp_r_cbo.setValue(state["radii"][2])
        sp_r_cbi.setValue(state["radii"][3])
        state["sync"] = False

    def on_ctrl_data(event=None):
        if state["sync"] or not has_geometry():
            return
        cur = np.asarray(layers["ctrl"].data, dtype=float)
        if cur.shape[0] != 5:
            return  # nombre de points modifié : on ignore (utiliser Réinitialiser)
        prev = state["pts_prev"]
        center = cur[0].copy()
        # Le centre a bougé -> translater les poignées d'autant.
        dc = center - prev[0]
        if np.hypot(dc[0], dc[1]) > 1e-6:
            cur[1:] = prev[1:] + dc
            state["sync"] = True
            layers["ctrl"].data = cur
            state["sync"] = False
        radii = np.hypot(cur[1:, 0] - center[0], cur[1:, 1] - center[1])
        radii = np.maximum(radii, 1.0)
        # Contrainte : r_stele <= 0.9 * r_root -> on ramène la poignèe stèle
        # (indice 2) sur le périmètre limite si nécessaire.
        max_stele = 0.9 * radii[0]
        if radii[1] > max_stele + 1e-9:
            radii[1] = max_stele
            vec = cur[2] - center
            nrm = float(np.hypot(vec[0], vec[1]))
            cur[2] = center + (vec * (max_stele / nrm) if nrm > 1e-9
                               else np.array([0.0, max_stele]))
            state["sync"] = True
            layers["ctrl"].data = cur
            state["sync"] = False
        state["center"] = center
        state["radii"] = radii
        state["pts_prev"] = cur.copy()
        set_spinboxes_from_radii()
        redraw_circles()
        update_previews()

    def on_spin_changed(*_):
        if state["sync"] or not has_geometry():
            return
        radii = np.array([
            sp_r_root.value(), sp_r_stele.value(),
            sp_r_cbo.value(), sp_r_cbi.value(),
        ], dtype=float)
        # Contrainte : r_stele <= 0.9 * r_root.
        max_stele = 0.9 * radii[0]
        if radii[1] > max_stele + 1e-9:
            radii[1] = max_stele
            state["sync"] = True
            sp_r_stele.setValue(radii[1])
            state["sync"] = False
        state["radii"] = radii
        cy, cx = state["center"]
        pts = [[cy, cx]]
        for r, a in zip(radii, HANDLE_ANGLES):
            pts.append([cy - r * np.sin(a), cx + r * np.cos(a)])
        pts = np.array(pts, dtype=float)
        state["pts_prev"] = pts.copy()
        state["sync"] = True
        layers["ctrl"].data = pts
        layers["ctrl"].face_color = HANDLE_COLORS
        state["sync"] = False
        redraw_circles()
        update_previews()

    # ---- Aperçus de segmentation ---------------------------------------
    def update_cb_preview():
        if state["image"] is None or not has_geometry() or state["cb_threshold"] is None:
            return
        _, annulus, _, _ = get_masks()
        thr = state["cb_threshold"]
        bright = state["image_f"] > thr
        ann = annulus[None, :, :]
        seg = np.where(ann & bright, 1, np.where(ann, 2, 0)).astype(np.uint8)
        if layers["cb_labels"] is None:
            layers["cb_labels"] = viewer.add_labels(
                seg, name="Cortical bridges (aperçu)", opacity=0.6
            )
        else:
            layers["cb_labels"].data = seg

    def update_iodine_preview():
        if not state["iodine"] or state["image"] is None or not has_geometry():
            return
        if state["iodine_threshold"] is None:
            return
        _, _, stele, _ = get_masks()
        thr = state["iodine_threshold"]
        iod = (state["image_f"] > thr) & stele[None, :, :]
        seg = iod.astype(np.uint8)
        if layers["iodine_labels"] is None:
            layers["iodine_labels"] = viewer.add_labels(
                seg, name="Iodé (aperçu)", opacity=0.6
            )
        else:
            layers["iodine_labels"].data = seg

    def update_previews():
        update_cb_preview()
        update_iodine_preview()

    # ---- Ouverture image ------------------------------------------------
    def open_image():
        path, _ = QFileDialog.getOpenFileName(
            None, "Ouvrir une racine redressée", "",
            "Fichiers TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        path = Path(path)
        try:
            vol = tiff.imread(str(path))
        except Exception as e:
            QMessageBox.critical(None, "Erreur de lecture", str(e))
            return
        if vol.ndim == 2:
            vol = vol[None, ...]
        if vol.ndim != 3:
            QMessageBox.critical(
                None, "Format inattendu",
                f"Volume de dimension {vol.ndim} (attendu 3 : Z, Y, X)."
            )
            return

        # Nettoyage d'une éventuelle session précédente.
        for key in list(layers):
            if layers[key] is not None:
                try:
                    viewer.layers.remove(layers[key])
                except Exception:
                    pass
                layers[key] = None
        for k in ("cb_threshold", "iodine_threshold", "hydrated"):
            state[k] = None

        state["image"] = vol
        state["image_f"] = vol.astype(np.float32)
        state["image_path"] = path
        layers["image"] = viewer.add_image(vol, name="Image", colormap="gray")

        lbl_img.setText(f"{path.name} — {vol.shape} {vol.dtype}")
        lbl_img.setStyleSheet("color: #ccc; font-size: 11px;")

        # Tentative de lecture du pas depuis les métadonnées du déroulé.
        meta = path.parent / (path.stem.replace("_straightened", "") + "_metadata.json")
        if meta.exists():
            try:
                d = json.loads(meta.read_text())
                if "step" in d:
                    sp_step.setValue(float(d["step"]))
            except Exception:
                pass

        # Bornes de seuil suivant l'intensité.
        vmin, vmax = float(vol.min()), float(vol.max())
        for sp in (sp_cb_thr, sp_io_thr):
            sp.setRange(vmin - 1, vmax + 1)

        # Réglage des bornes de rayon puis géométrie initiale.
        h, w = vol.shape[1:]
        rmax = float(max(h, w))
        for sp in (sp_r_root, sp_r_stele, sp_r_cbo, sp_r_cbi):
            sp.setRange(1, rmax)
        init_geometry()

    def init_geometry():
        if state["image"] is None:
            return
        h, w = state["image"].shape[1:]
        mn = min(h, w)
        r_root = 0.35 * mn
        r_stele = 0.15 * mn
        state["center"] = np.array([h / 2.0, w / 2.0], dtype=float)
        state["radii"] = np.array(
            [r_root, r_stele, 0.85 * r_root, 1.15 * r_stele], dtype=float
        )
        set_spinboxes_from_radii()
        build_ctrl_points()
        redraw_circles()
        try:
            viewer.layers.selection.active = layers["ctrl"]
        except Exception:
            pass

    def recompute_derived():
        if not has_geometry():
            return
        state["sync"] = True
        sp_r_cbo.setValue(0.85 * sp_r_root.value())
        sp_r_cbi.setValue(1.15 * sp_r_stele.value())
        state["sync"] = False
        on_spin_changed()

    # ---- Seuil cortical bridges ----------------------------------------
    def cb_otsu():
        if state["image"] is None or not has_geometry():
            QMessageBox.warning(None, "Géométrie manquante",
                                "Ouvrez une image et réglez les cercles.")
            return
        _, annulus, _, _ = get_masks()
        if annulus.sum() == 0:
            QMessageBox.warning(None, "Anneau vide",
                                "La zone entre les deux cercles dérivés est vide.")
            return
        vals = state["image_f"][:, annulus].ravel()
        try:
            thr = float(threshold_otsu(vals))
        except Exception as e:
            QMessageBox.warning(None, "Otsu impossible", str(e))
            return
        sp_cb_thr.setValue(thr)  # déclenche l'aperçu

    def on_cb_thr_changed(*_):
        state["cb_threshold"] = sp_cb_thr.value()
        update_cb_preview()

    def cb_nudge(sign):
        sp_cb_thr.setValue(sp_cb_thr.value() + sign * sp_cb_step.value())

    def cb_validate():
        if state["cb_threshold"] is None:
            return
        QMessageBox.information(
            None, "Seuil cortical bridges",
            f"Seuil retenu : {state['cb_threshold']:.2f}"
        )

    # ---- Seuil iode -----------------------------------------------------
    def io_otsu():
        if not has_geometry():
            return
        _, _, stele, _ = get_masks()
        if stele.sum() == 0:
            QMessageBox.warning(None, "Stèle vide", "Le cercle stèle est vide.")
            return
        vals = state["image_f"][:, stele].ravel()
        try:
            thr = float(threshold_otsu(vals))
        except Exception as e:
            QMessageBox.warning(None, "Otsu impossible", str(e))
            return
        sp_io_thr.setValue(thr)

    def on_io_thr_changed(*_):
        state["iodine_threshold"] = sp_io_thr.value()
        update_iodine_preview()

    def io_nudge(sign):
        sp_io_thr.setValue(sp_io_thr.value() + sign * sp_io_step.value())

    def io_validate():
        if state["iodine_threshold"] is None:
            return
        QMessageBox.information(
            None, "Seuil hydraté/iodé",
            f"Seuil retenu : {state['iodine_threshold']:.2f}"
        )

    # ---- Échantillonnage tissu hydraté ---------------------------------
    def io_sample():
        if state["image"] is None:
            return
        if layers["hydr_pts"] is None:
            lay = viewer.add_points(
                np.empty((0, 3)), name="Zone hydratée", ndim=3,
                size=2 * SAMPLE_RADIUS, face_color="#00a0ff",
                border_color="#ffffff",
            )
            lay.events.data.connect(on_hydr_point)
            layers["hydr_pts"] = lay
        viewer.layers.selection.active = layers["hydr_pts"]
        try:
            layers["hydr_pts"].mode = "add"
        except Exception:
            pass
        QMessageBox.information(
            None, "Zone hydratée",
            "Cliquez sur une zone purement hydratée de la coupe affichée."
        )

    def on_hydr_point(event=None):
        lay = layers["hydr_pts"]
        data = np.asarray(lay.data)
        if len(data) == 0:
            return
        p = data[-1]
        z = int(round(p[0])); y = int(round(p[1])); x = int(round(p[2]))
        vol = state["image_f"]
        z = np.clip(z, 0, vol.shape[0] - 1)
        h, w = vol.shape[1:]
        yy, xx = np.ogrid[:h, :w]
        disk = np.hypot(yy - y, xx - x) <= SAMPLE_RADIUS
        if disk.sum() == 0:
            return
        val = float(np.median(vol[z][disk]))
        state["hydrated"] = val
        lbl_io_hydr.setText(f"Valeur tissu hydraté : {val:.2f}")
        try:
            lay.mode = "select"
        except Exception:
            pass
        QMessageBox.information(
            None, "Tissu hydraté",
            f"Valeur de base (médiane, r={SAMPLE_RADIUS}) : {val:.2f}"
        )

    def io_generate_over():
        if state["image"] is None or not has_geometry():
            return
        if state["hydrated"] is None:
            QMessageBox.warning(None, "Valeur manquante",
                                "Échantillonnez d'abord une zone hydratée.")
            return
        _, _, stele, _ = get_masks()
        over = np.clip(state["image_f"] - state["hydrated"], 0, None)
        over = (over * stele[None, :, :]).astype(np.float32)
        if layers["over"] is None:
            layers["over"] = viewer.add_image(
                over, name="Surintensité", colormap="inferno",
                blending="additive",
            )
        else:
            layers["over"].data = over

        # --- Sauvegardes à côté de l'image source ----------------------
        path = state["image_path"]
        stem, parent = path.stem, path.parent
        written = []

        # Pile de surintensité en float32.
        p_stack = parent / f"{stem}_overintensity.tif"
        try:
            tiff.imwrite(str(p_stack), over)  # over est déjà float32
            written.append(p_stack.name)
        except Exception as e:
            QMessageBox.warning(None, "Écriture impossible",
                                f"Pile de surintensité : {e}")

        # Capture « jolie » de la coupe affichée (sans les objets géométriques).
        overlay_layers = [layers[k] for k in
                          ("ctrl", "circles", "cb_labels", "iodine_labels",
                           "hydr_pts")]
        prev_vis = {}
        for lay in overlay_layers:
            if lay is not None:
                prev_vis[lay] = lay.visible
                lay.visible = False
        p_png = parent / f"{stem}_overintensity_view.png"
        try:
            viewer.screenshot(path=str(p_png), canvas_only=True, flash=False)
            written.append(p_png.name)
        except Exception as e:
            QMessageBox.warning(None, "Capture impossible", str(e))
        finally:
            for lay, vis in prev_vis.items():
                lay.visible = vis

        if written:
            QMessageBox.information(
                None, "Surintensité enregistrée",
                "Fichiers écrits à côté de l'image :\n" + "\n".join(written)
            )

    # ---- Type de racine -------------------------------------------------
    def on_type_changed():
        state["iodine"] = rb_yes.isChecked()
        io_box.setVisible(state["iodine"])
        if not state["iodine"] and layers["iodine_labels"] is not None:
            try:
                viewer.layers.remove(layers["iodine_labels"])
            except Exception:
                pass
            layers["iodine_labels"] = None

    # ---- Export ---------------------------------------------------------
    def export_csv():
        if state["image"] is None:
            QMessageBox.warning(None, "Pas d'image", "Ouvrez d'abord une image.")
            return
        if not has_geometry():
            QMessageBox.warning(None, "Géométrie manquante", "Réglez les cercles.")
            return
        if state["cb_threshold"] is None:
            QMessageBox.warning(None, "Seuil manquant",
                                "Déterminez le seuil des cortical bridges.")
            return

        path = state["image_path"]
        stem = path.stem
        parent = path.parent
        voxel = sp_voxel.value()
        step = sp_step.value()
        cy, cx = state["center"]
        r_root, r_stele, r_cbo, r_cbi = state["radii"]
        _, annulus, stele, _ = get_masks()
        vol = state["image_f"]
        nz = vol.shape[0]
        z_index = np.arange(nz)
        z_phys = z_index * step * voxel

        written = []

        # -- CSV 1 : géométrie ------------------------------------------
        geo_row = {
            "source_image": path.name,
            "x_center_vx": cx,
            "y_center_vx": cy,
            "r_root_vx": r_root,
            "r_stele_vx": r_stele,
            "r_cb_outer_85pct_root_vx": r_cbo,
            "r_cb_inner_115pct_stele_vx": r_cbi,
            "cortical_bridge_threshold": state["cb_threshold"],
            "iodine_threshold": state["iodine_threshold"] if state["iodine"] else "",
            "hydrated_value": state["hydrated"] if state["iodine"] else "",
            "voxel_size": voxel,
            "inter_slice_step_vx": step,
        }
        p_geo = parent / f"{stem}_geometry.csv"
        pd.DataFrame([geo_row]).to_csv(p_geo, index=False)
        written.append(p_geo.name)

        # -- CSV 2 : cortical bridges par coupe -------------------------
        ratios = cortical_ratio_per_slice(vol, annulus, state["cb_threshold"])
        df_cb = pd.DataFrame({
            "slice_index": z_index,
            "z_physical": z_phys,
            "cortical_bridge_ratio_percent": ratios,
        })
        p_cb = parent / f"{stem}_cortical_bridges.csv"
        df_cb.to_csv(p_cb, index=False)
        written.append(p_cb.name)

        # -- CSV 3 : iode -----------------------------------------------
        if state["iodine"]:
            missing = []
            if state["iodine_threshold"] is None:
                missing.append("seuil hydraté/iodé")
            if state["hydrated"] is None:
                missing.append("valeur tissu hydraté")
            if missing:
                QMessageBox.warning(
                    None, "Données iode manquantes",
                    "Manque : " + ", ".join(missing) +
                    ".\nLes CSV 1 et 2 ont tout de même été écrits."
                )
            else:
                xyl, ncomp, mover = iodine_metrics_per_slice(
                    vol, stele, state["iodine_threshold"], state["hydrated"]
                )
                df_io = pd.DataFrame({
                    "slice_index": z_index,
                    "z_physical": z_phys,
                    "xylem_area_ratio_percent": xyl,
                    "n_overintensity_components": ncomp,
                    "mean_overintensity_over_stele": mover,
                })
                p_io = parent / f"{stem}_iodine.csv"
                df_io.to_csv(p_io, index=False)
                written.append(p_io.name)

        QMessageBox.information(
            None, "Export terminé",
            "Fichiers écrits à côté de l'image :\n" + "\n".join(written)
        )

    # -------------------------------------------------------------------------
    # BRANCHEMENTS
    # -------------------------------------------------------------------------
    btn_open.clicked.connect(open_image)
    rb_no.toggled.connect(lambda _: on_type_changed())
    rb_yes.toggled.connect(lambda _: on_type_changed())

    for sp in (sp_r_root, sp_r_stele, sp_r_cbo, sp_r_cbi):
        sp.valueChanged.connect(on_spin_changed)
    btn_recompute.clicked.connect(recompute_derived)
    btn_reset_geo.clicked.connect(init_geometry)

    btn_cb_otsu.clicked.connect(cb_otsu)
    sp_cb_thr.valueChanged.connect(on_cb_thr_changed)
    btn_cb_minus.clicked.connect(lambda: cb_nudge(-1))
    btn_cb_plus.clicked.connect(lambda: cb_nudge(+1))
    btn_cb_validate.clicked.connect(cb_validate)

    btn_io_otsu.clicked.connect(io_otsu)
    sp_io_thr.valueChanged.connect(on_io_thr_changed)
    btn_io_minus.clicked.connect(lambda: io_nudge(-1))
    btn_io_plus.clicked.connect(lambda: io_nudge(+1))
    btn_io_validate.clicked.connect(io_validate)
    btn_io_sample.clicked.connect(io_sample)
    btn_io_over.clicked.connect(io_generate_over)

    btn_export.clicked.connect(export_csv)

    viewer.window.add_dock_widget(scroll, name="Cortical bridges", area="right")
    napari.run()


if __name__ == "__main__":
    main()
