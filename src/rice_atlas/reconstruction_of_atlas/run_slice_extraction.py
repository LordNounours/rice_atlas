#Equivalent to run this in command line : 

    ## From the repository root
    #cd /home/rfernandez/Python_prog/rice_atlas_stage_thomas

    ## Run on a volume with centerlines
    #python -m rice_atlas.slice_extraction \
    #    /path/to/volume.tif \
    #    /path/to/centerlines/ \
    #    /path/to/output/ \
    #    --half-size 64 \
    #    --step 1.0

from rice_atlas import process_batch, extract_and_save_slices, load_centerline_csv
import numpy as np
import tifffile as tiff
from pathlib import Path

# Load volume
volume_path = "/media/rfernandez/Crucial X9/Test_Charlotte_2026_01_root_and_leave/Img_raw_crop_8b.tif"
centerlines_dir = Path("/media/rfernandez/Crucial X9/Test_Charlotte_2026_01_root_and_leave/leaves/")
output_dir = Path("/media/rfernandez/Crucial X9/Test_Charlotte_2026_01_root_and_leave/output/")

print(f"Loading volume: {volume_path}")
volume = tiff.imread(volume_path)
print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")

# Get ALL CSV files
csv_files = sorted(centerlines_dir.glob("*.csv"))
if not csv_files:
    print(f"No CSV files found in {centerlines_dir}")
    exit(1)

print(f"\nFound {len(csv_files)} centerline files")

# Process each CSV file
for idx, csv_file in enumerate(csv_files, 1):
    print(f"\n{'='*70}")
    print(f"Processing {idx}/{len(csv_files)}: {csv_file.name}")
    print(f"{'='*70}")
    
    # Load centerline
    centerline = load_centerline_csv(csv_file)
    print(f"Loaded {len(centerline)} points")
    
    # Extract slices
    organ_id = csv_file.stem
    organ_output = output_dir / organ_id
    
    metadata = extract_and_save_slices(
        volume=volume,
        centerline=centerline,
        output_dir=organ_output,
        organ_id=organ_id,
        organ_type="root",
        half_size=64,
        step=1.0,
        save_3d_stack=True,
        save_individual=False,
        save_metadata=True,
        debug=False
    )
    
    print(f"✓ Extracted {metadata.num_slices} slices to {organ_output}")

print(f"\n{'='*70}")
print(f"✅ ALL DONE! Processed {len(csv_files)} organs")
print(f"{'='*70}")
