from ._widget import load_volume_widget


def make_widget():
    widget = load_volume_widget()
    
    return widget


__all__ = ["napari_get_reader", "write_single_image", "write_multiple", "make_sample_data", "make_widget"]
