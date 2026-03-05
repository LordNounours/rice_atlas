__version__ = "0.0.1"

# Only import slice_extraction functions, skip the widget if it has issues
try:
    from ._widget import ExampleQWidget
except (ImportError, SyntaxError):
    # Widget has incomplete code, skip it
    ExampleQWidget = None

# Import slice extraction functions from the new location
from .reconstruction_of_atlas.slice_extraction import (
    process_batch,
    extract_and_save_slices,
    extract_organ_volume,
    interpolate_path_with_frames,
    trilinear_interpolation,
    load_centerline_csv
)

__all__ = [
    "process_batch",
    "extract_and_save_slices",
    "extract_organ_volume",
    "interpolate_path_with_frames",
    "trilinear_interpolation",
    "load_centerline_csv"
]

if ExampleQWidget is not None:
    __all__.append("ExampleQWidget")
