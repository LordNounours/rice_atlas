from napari import Viewer
import napari
from rice_atlas._widget import make_segment_widget

v = Viewer()
w = make_segment_widget(viewer=v)
v.window.add_dock_widget(w, name="Segmentation widget")
napari.run()  # pour ouvrir la fenêtre si tu es en script autonome
