#!/usr/bin/env python3
"""
Create mosaic of straightened organ images.
Concatenates all root and leaf stacks into separate mosaics.
"""
import numpy as np
import tifffile as tiff
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

def find_organ_stacks(base_dir, pattern_prefix):
    """
    Find all straightened stacks matching a pattern.
    
    Args:
        base_dir: Base directory containing organ folders
        pattern_prefix: Prefix pattern (e.g., 'root_' or 'T')
    
    Returns:
        List of tuples (organ_name, stack_path)
    """
    base_dir = Path(base_dir)
    stacks = []
    
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        
        # Check if folder matches pattern
        if pattern_prefix == 'root_' and folder_name.startswith('root_'):
            organ_name = folder_name  # e.g., 'root_A'
        elif pattern_prefix == 'T' and folder_name.startswith('T'):
            organ_name = folder_name  # e.g., 'T1', 'T2a', etc.
        else:
            continue
        
        # Look for straightened stack
        stack_files = list(folder.glob(f"{folder_name}_straightened.tif"))
        if stack_files:
            stacks.append((organ_name, stack_files[0]))
    
    return stacks


def load_and_normalize_stack(stack_path):
    """
    Load a stack.
    
    Returns:
        3D array (Z, Y, X)
    """
    stack = tiff.imread(str(stack_path))
    
    # Convert to uint8 only if necessary (should already be uint8)
    if stack.dtype != np.uint8:
        print(f"  Warning: Stack is {stack.dtype}, converting to uint8")
        stack = stack.astype(np.uint8)
    
    return stack


def add_text_to_image(img, text, position='top', font_size=20):
    """
    Add text label to an image.
    
    Args:
        img: PIL Image or numpy array
        text: Text to add
        position: 'top' or 'bottom'
        font_size: Font size in pixels
    
    Returns:
        PIL Image with text
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position
    x = (img.width - text_width) // 2
    if position == 'top':
        y = 5
    else:
        y = img.height - text_height - 5
    
    # Draw white background rectangle
    padding = 3
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill='white'
    )
    
    # Draw text in black
    draw.text((x, y), text, fill='black', font=font)
    
    return img


def create_mosaic(stacks_info, output_path, organ_type='root'):
    """
    Create a mosaic from multiple organ stacks.
    
    Args:
        stacks_info: List of (organ_name, stack_path) tuples
        output_path: Path to save mosaic
        organ_type: 'root' or 'leaf' for display
    """
    if not stacks_info:
        print(f"No {organ_type}s found!")
        return
    
    print(f"\n{'='*70}")
    print(f"Creating {organ_type} mosaic with {len(stacks_info)} organs")
    print(f"{'='*70}")
    
    # Load all stacks and get dimensions
    stacks = []
    max_z = 0
    max_y = 0
    max_x = 0
    
    for organ_name, stack_path in stacks_info:
        print(f"Loading {organ_name}...")
        stack = load_and_normalize_stack(stack_path)
        stacks.append((organ_name, stack))
        
        z, y, x = stack.shape
        max_z = max(max_z, z)
        max_y = max(max_y, y)
        max_x = max(max_x, x)
        print(f"  Shape: {stack.shape}")
    
    print(f"\nMax dimensions: Z={max_z}, Y={max_y}, X={max_x}")
    
    # Calculate grid layout (try to make it roughly square)
    n_organs = len(stacks)
    n_cols = math.ceil(math.sqrt(n_organs))
    n_rows = math.ceil(n_organs / n_cols)
    
    print(f"Grid layout: {n_rows} rows × {n_cols} cols")
    
    # Create mosaic for each Z slice
    mosaic_z = max_z
    mosaic_y = n_rows * max_y
    mosaic_x = n_cols * max_x
    
    print(f"Mosaic shape: ({mosaic_z}, {mosaic_y}, {mosaic_x})")
    print("Creating mosaic...")
    
    mosaic = np.zeros((mosaic_z, mosaic_y, mosaic_x), dtype=np.uint8)
    
    for idx, (organ_name, stack) in enumerate(stacks):
        row = idx // n_cols
        col = idx % n_cols
        
        z, y, x = stack.shape
        
        y_start = row * max_y
        x_start = col * max_x
        
        # Place stack in mosaic (top-left aligned)
        mosaic[:z, y_start:y_start+y, x_start:x_start+x] = stack
        
        # Add label on first slice
        first_slice = mosaic[0, y_start:y_start+max_y, x_start:x_start+max_x].copy()
        first_slice_pil = add_text_to_image(first_slice, organ_name, position='top', font_size=16)
        mosaic[0, y_start:y_start+max_y, x_start:x_start+max_x] = np.array(first_slice_pil)
    
    # Save mosaic
    print(f"Saving mosaic to {output_path}...")
    tiff.imwrite(str(output_path), mosaic)
    
    print(f"✓ Mosaic saved: {output_path}")
    print(f"  Shape: {mosaic.shape}")
    print(f"  Size: {mosaic.nbytes / (1024**2):.1f} MB")
    
    # Also save max projection for quick visualization
    max_proj_path = output_path.parent / f"{output_path.stem}_maxproj.png"
    max_proj = np.max(mosaic, axis=0)
    Image.fromarray(max_proj).save(str(max_proj_path))
    print(f"✓ Max projection saved: {max_proj_path}")


def main():
    """Main function to create root and leaf mosaics."""
    base_dir = Path("/media/rfernandez/Crucial X9/Test_Charlotte_2026_01_root_and_leave/output")
    
    print("="*70)
    print("ORGAN MOSAIC CREATOR")
    print("="*70)
    print(f"Base directory: {base_dir}")
    
    # Find root stacks
    root_stacks = find_organ_stacks(base_dir, 'root_')
    print(f"\nFound {len(root_stacks)} root stacks:")
    for name, path in root_stacks:
        print(f"  - {name}")
    
    # Find leaf stacks
    leaf_stacks = find_organ_stacks(base_dir, 'T')
    print(f"\nFound {len(leaf_stacks)} leaf stacks:")
    for name, path in leaf_stacks:
        print(f"  - {name}")
    
    # Create mosaics
    if root_stacks:
        root_mosaic_path = base_dir / "roots_mosaic.tif"
        create_mosaic(root_stacks, root_mosaic_path, organ_type='root')
    
    if leaf_stacks:
        leaf_mosaic_path = base_dir / "leaves_mosaic.tif"
        create_mosaic(leaf_stacks, leaf_mosaic_path, organ_type='leaf')
    
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)


if __name__ == "__main__":
    main()