"""
Organ slice extraction along centerlines.
Standalone functions for batch processing and GUI integration.

This module extracts orthogonal 2D slices from a 3D volume along 
organ centerlines, producing "straightened" organ representations.
"""

import numpy as np
import tifffile as tiff
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import splprep, splev


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OrganMetadata:
    """Metadata for an extracted organ."""
    organ_id: str
    organ_type: str  # 'root' or 'leaf'
    plant_id: Optional[str] = None
    population: Optional[str] = None
    num_slices: int = 0
    centerline_length: float = 0.0
    patch_size: int = 128
    step: float = 1.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class SliceInfo:
    """Information about a single extracted slice."""
    index: int
    filename: str
    center_zyx: Tuple[float, float, float]
    arc_length: float
    

# ============================================================================
# CORE INTERPOLATION FUNCTIONS
# ============================================================================

def trilinear_interpolation(volume: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Trilinear interpolation in a 3D volume.
    
    Samples the volume at non-integer coordinates using trilinear interpolation,
    which provides smooth values by weighting the 8 surrounding voxels.
    
    Args:
        volume: 3D array with shape (Z, Y, X)
        coords: Float coordinates with shape (3, H, W) ordered as (z, y, x)
    
    Returns:
        Interpolated 2D array with shape (H, W)
    
    Example:
        >>> vol = np.random.rand(100, 100, 100)
        >>> coords = np.random.rand(3, 64, 64) * 99
        >>> slice_img = trilinear_interpolation(vol, coords)
        >>> slice_img.shape
        (64, 64)
    """
    z_coords = coords[0]
    y_coords = coords[1]
    x_coords = coords[2]

    # Clamp coordinates to valid range (leaving small epsilon for floor/ceil)
    z_coords = np.clip(z_coords, 0, volume.shape[0] - 1.001)
    y_coords = np.clip(y_coords, 0, volume.shape[1] - 1.001)
    x_coords = np.clip(x_coords, 0, volume.shape[2] - 1.001)

    # Integer indices for 8 surrounding voxels
    z0 = np.floor(z_coords).astype(int)
    y0 = np.floor(y_coords).astype(int)
    x0 = np.floor(x_coords).astype(int)
    z1 = np.clip(z0 + 1, 0, volume.shape[0] - 1)
    y1 = np.clip(y0 + 1, 0, volume.shape[1] - 1)
    x1 = np.clip(x0 + 1, 0, volume.shape[2] - 1)

    # Fractional parts for interpolation weights
    zd = z_coords - z0
    yd = y_coords - y0
    xd = x_coords - x0

    # Sample 8 corner values
    c000 = volume[z0, y0, x0]
    c001 = volume[z0, y0, x1]
    c010 = volume[z0, y1, x0]
    c011 = volume[z0, y1, x1]
    c100 = volume[z1, y0, x0]
    c101 = volume[z1, y0, x1]
    c110 = volume[z1, y1, x0]
    c111 = volume[z1, y1, x1]

    # Interpolate along x
    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd

    # Interpolate along y
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd

    # Interpolate along z
    return c0 * (1 - zd) + c1 * zd


def compute_orthonormal_frame(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute an orthonormal frame from a tangent vector.
    Used only for the FIRST frame of a path. Subsequent frames should use
    propagate_frame() for continuity.
    
    Args:
        tangent: 3D vector (z, y, x)
    
    Returns:
        Tuple of (axis0, axis1, axis2)
    """
    tangent = np.array(tangent, dtype=np.float64)
    norm = np.linalg.norm(tangent)
    
    if norm < 1e-8:
        return np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])
    
    axis2 = tangent / norm
    
    # Choose an arbitrary vector not parallel to axis2 (same logic as original code)
    if abs(axis2[1]) < 0.9999:
        arbitrary = np.array([0., -1., 0.])
    else:
        arbitrary = np.array([-1., 0., 0.])
    
    axis0 = np.cross(arbitrary, axis2)
    axis0 = axis0 / np.linalg.norm(axis0)
    
    axis1 = np.cross(axis2, axis0)
    axis1 = axis1 / np.linalg.norm(axis1)
    
    return axis0, axis1, axis2


def propagate_frame(tangent: np.ndarray, prev_axis0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute orthonormal frame by propagating axis0 from the previous frame.
    This ensures smooth rotation along the path (parallel transport).
    
    Args:
        tangent: Current tangent vector (z, y, x)
        prev_axis0: axis0 from the previous frame
    
    Returns:
        Tuple of (axis0, axis1, axis2) with continuous rotation
    """
    tangent = np.array(tangent, dtype=np.float64)
    norm = np.linalg.norm(tangent)
    
    if norm < 1e-8:
        return prev_axis0, np.cross(np.array([0., 0., 1.]), prev_axis0), np.array([0., 0., 1.])
    
    axis2 = tangent / norm
    
    # Project previous axis0 onto the plane perpendicular to new tangent
    # axis1 = cross(axis2, prev_axis0) then axis0 = cross(axis1, axis2)
    axis1 = np.cross(axis2, prev_axis0)
    n1 = np.linalg.norm(axis1)
    
    if n1 < 1e-8:
        # Degenerate case: prev_axis0 is parallel to new tangent
        # Fall back to independent computation
        return compute_orthonormal_frame(tangent)
    
    axis1 = axis1 / n1
    axis0 = np.cross(axis1, axis2)
    axis0 = axis0 / np.linalg.norm(axis0)
    
    return axis0, axis1, axis2


def resample_curve_with_spline(curve_points: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """
    Resample a curve using cubic B-spline interpolation for smooth trajectory.
    
    This creates a smooth spline through the control points, then resamples
    it at regular intervals. This avoids sharp angles at control points.
    
    Args:
        curve_points: Array of shape (N, 3) with (z, y, x) coordinates
        spacing: Desired spacing between resampled points
    
    Returns:
        Resampled curve as array (M, 3) where M is number of resampled points
    """
    curve_points = np.atleast_2d(curve_points).astype(np.float64)
    
    if len(curve_points) < 2:
        return curve_points
    
    # Calculate total curve length (using linear segments as approximation)
    segments = np.diff(curve_points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    curve_length = np.sum(segment_lengths)
    
    if curve_length < 1e-8:
        return curve_points
    
    # Calculate number of points for desired spacing
    num_points = max(2, int(np.ceil(curve_length / spacing)))
    
    # Create B-spline representation
    # s=0 means interpolate exactly through all points
    # k=3 for cubic spline (or k=min(3, len(curve_points)-1) if few points)
    k = min(3, len(curve_points) - 1)
    
    try:
        tck, u = splprep(curve_points.T, s=0, k=k, per=False)
    except Exception as e:
        print(f"Warning: Spline fitting failed ({e}), using linear interpolation")
        return curve_points
    
    # Resample at regular intervals
    u_new = np.linspace(0, 1, num_points)
    curve_resampled = np.column_stack(splev(u_new, tck, der=0))
    
    return curve_resampled


def interpolate_path_with_frames(
    path_points: np.ndarray, 
    step: float = 1.0
) -> List[Dict]:
    """
    Interpolate a 3D path with regular arc-length spacing and compute local frames.
    
    First fits a smooth B-spline through the control points, then resamples
    at regular intervals. Uses parallel transport to ensure smooth frame rotation.
    
    Args:
        path_points: Array of shape (N, 3) with coordinates as (z, y, x)
        step: Distance between interpolated points in voxel units
    
    Returns:
        List of dicts, each containing:
            - 'point': np.ndarray (z, y, x) coordinates
            - 'axis0': np.ndarray first perpendicular axis
            - 'axis1': np.ndarray second perpendicular axis  
            - 'axis2': np.ndarray tangent vector
            - 'arc_length': float cumulative distance from start
    
    Example:
        >>> path = np.array([[0,0,0], [0,0,10], [0,10,10]])
        >>> frames = interpolate_path_with_frames(path, step=2.0)
        >>> len(frames)  # approximately (10 + 10) / 2 = 10 points
        11
    """
    path_points = np.atleast_2d(path_points).astype(np.float64)
    
    if len(path_points) < 2:
        return []
    
    # STEP 1: Resample with smooth spline
    print(f"  Fitting B-spline through {len(path_points)} control points...")
    resampled_points = resample_curve_with_spline(path_points, spacing=step)
    print(f"  Resampled to {len(resampled_points)} points with spacing ~{step}")
    
    if len(resampled_points) < 2:
        return []
    
    # STEP 2: Compute tangents from resampled smooth curve
    segments = np.diff(resampled_points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    
    # Cumulative arc length
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    
    print(f"  Total arc length: {total_length:.2f} voxels")
    
    # STEP 3: Compute frames at each resampled point
    frames = []
    prev_axis0 = None
    
    for i in range(len(resampled_points) - 1):
        point = resampled_points[i]
        tangent = segments[i]  # Already smooth thanks to spline
        arc_length = cumulative[i]
        
        # Compute frame: first frame independently, then propagate
        if prev_axis0 is None:
            axis0, axis1, axis2 = compute_orthonormal_frame(tangent)
        else:
            axis0, axis1, axis2 = propagate_frame(tangent, prev_axis0)
        
        prev_axis0 = axis0
        
        frames.append({
            'point': point,
            'axis0': axis0,
            'axis1': axis1,
            'axis2': axis2,
            'arc_length': float(arc_length)
        })
    
    # Add last point
    point = resampled_points[-1]
    tangent = segments[-1]
    arc_length = cumulative[-1]
    
    if prev_axis0 is None:
        axis0, axis1, axis2 = compute_orthonormal_frame(tangent)
    else:
        axis0, axis1, axis2 = propagate_frame(tangent, prev_axis0)
    
    frames.append({
        'point': point,
        'axis0': axis0,
        'axis1': axis1,
        'axis2': axis2,
        'arc_length': float(arc_length)
    })
    
    return frames


# ============================================================================
# SLICE EXTRACTION
# ============================================================================

def extract_single_slice(
    volume: np.ndarray,
    center: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
    half_size: int = 64,
    debug: bool = False,
    slice_idx: int = 0
) -> np.ndarray:
    """
    Extract a single 2D slice from volume at given position and orientation.
    
    Args:
        volume: 3D array (Z, Y, X)
        center: Slice center as (z, y, x)
        axis0: First in-plane axis
        axis1: Second in-plane axis
        half_size: Half-width of extracted patch
        debug: If True, print debug info and save visualization
        slice_idx: Index of this slice (for debug output)
    
    Returns:
        2D array of shape (2*half_size, 2*half_size)
    """
    z, y, x = center
    
    if debug:
        print(f"\n{'='*60}")
        print(f"DEBUG: Slice {slice_idx}")
        print(f"{'='*60}")
        print(f"Center (z,y,x): ({z:.2f}, {y:.2f}, {x:.2f})")
        print(f"Volume shape: {volume.shape}")
        print(f"Center in bounds: z=[0,{volume.shape[0]}], y=[0,{volume.shape[1]}], x=[0,{volume.shape[2]}]")
        print(f"\nOrthonormal frame:")
        print(f"  axis0 (1st perpendicular): {axis0}")
        print(f"  axis1 (2nd perpendicular): {axis1}")
        print(f"  |axis0| = {np.linalg.norm(axis0):.6f}")
        print(f"  |axis1| = {np.linalg.norm(axis1):.6f}")
        print(f"  axis0 · axis1 = {np.dot(axis0, axis1):.6f} (should be ~0)")
    
    # Build 2D grid
    grid = np.arange(-half_size, half_size)
    yy, xx = np.meshgrid(grid, grid, indexing='ij')
    
    if debug:
        print(f"\nGrid: {grid.shape}, range [{grid[0]}, {grid[-1]}]")
    
    # Transform to global coordinates
    # center is (z, y, x), axis0/axis1 are also (z, y, x)
    coords_zyx = (
        center[None, None, :] +
        xx[:, :, None] * axis0[None, None, :] +
        yy[:, :, None] * axis1[None, None, :]
    )
    
    if debug:
        print(f"\nCoordinate ranges after transformation:")
        print(f"  Z: [{coords_zyx[:,:,0].min():.2f}, {coords_zyx[:,:,0].max():.2f}]")
        print(f"  Y: [{coords_zyx[:,:,1].min():.2f}, {coords_zyx[:,:,1].max():.2f}]")
        print(f"  X: [{coords_zyx[:,:,2].min():.2f}, {coords_zyx[:,:,2].max():.2f}]")
        
        # Check if coordinates are outside volume
        out_of_bounds = (
            (coords_zyx[:,:,0] < 0) | (coords_zyx[:,:,0] >= volume.shape[0]) |
            (coords_zyx[:,:,1] < 0) | (coords_zyx[:,:,1] >= volume.shape[1]) |
            (coords_zyx[:,:,2] < 0) | (coords_zyx[:,:,2] >= volume.shape[2])
        )
        pct_oob = 100 * out_of_bounds.sum() / out_of_bounds.size
        print(f"  Out of bounds: {out_of_bounds.sum()} / {out_of_bounds.size} ({pct_oob:.1f}%)")
    
    # Reorder for trilinear_interpolation which expects (z, y, x) in axis 0
    sample_coords = np.stack([
        coords_zyx[:, :, 0],  # z
        coords_zyx[:, :, 1],  # y
        coords_zyx[:, :, 2],  # x
    ], axis=0)
    
    slice_img = trilinear_interpolation(volume, sample_coords)
    
    if debug:
        print(f"\nExtracted slice statistics:")
        print(f"  Shape: {slice_img.shape}")
        print(f"  Dtype: {slice_img.dtype}")
        print(f"  Mean: {slice_img.mean():.2f}")
        print(f"  Std: {slice_img.std():.2f}")
        print(f"  Min: {slice_img.min():.2f}")
        print(f"  Max: {slice_img.max():.2f}")
        print(f"  Non-zero: {np.count_nonzero(slice_img)} / {slice_img.size}")
        
        # Visualize slice
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show slice
        im = axes[0].imshow(slice_img, cmap='gray')
        axes[0].set_title(f'Slice {slice_idx} at (z={z:.1f}, y={y:.1f}, x={x:.1f})')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=axes[0])
        
        # Show histogram
        axes[1].hist(slice_img.ravel(), bins=50, alpha=0.7)
        axes[1].set_xlabel('Intensity')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Intensity Distribution')
        axes[1].axvline(slice_img.mean(), color='r', linestyle='--', label=f'Mean={slice_img.mean():.1f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'debug_slice_{slice_idx:04d}.png', dpi=100)
        plt.close()
        print(f"  Saved: debug_slice_{slice_idx:04d}.png")
    
    return slice_img


def extract_organ_volume(
    volume: np.ndarray,
    centerline: np.ndarray,
    half_size: int = 64,
    step: float = 1.0,
    progress: bool = True,
    debug: bool = False,
    debug_max_slices: int = 10
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Extract a straightened 3D organ volume from centerline.
    
    Creates a 3D array where:
    - Z axis = arc length along centerline
    - X, Y axes = perpendicular to centerline
    
    Args:
        volume: Input 3D volume (Z, Y, X)
        centerline: Path coordinates as (N, 3) array with (z, y, x)
        half_size: Half-width of cross-sections
        step: Spacing between slices along path
        progress: Show progress bar
        debug: Enable debug output
        debug_max_slices: Maximum number of slices to extract in debug mode
    
    Returns:
        Tuple of:
            - straightened: 3D array (num_slices, 2*half_size, 2*half_size)
            - frames: List of frame dicts with position info
    """
    print(f"\n{'='*70}")
    print(f"EXTRACTION PARAMETERS")
    print(f"{'='*70}")
    print(f"Volume shape: {volume.shape}")
    print(f"Centerline points: {len(centerline)}")
    print(f"Centerline range:")
    print(f"  Z: [{centerline[:,0].min():.1f}, {centerline[:,0].max():.1f}]")
    print(f"  Y: [{centerline[:,1].min():.1f}, {centerline[:,1].max():.1f}]")
    print(f"  X: [{centerline[:,2].min():.1f}, {centerline[:,2].max():.1f}]")
    print(f"Step size: {step}")
    print(f"Patch half-size: {half_size}")
    
    frames = interpolate_path_with_frames(centerline, step=step)
    
    if not frames:
        return np.array([]), []
    
    print(f"\nInterpolated to {len(frames)} frames")
    print(f"Total path length: {frames[-1]['arc_length']:.2f} voxels")
    
    if debug:
        print(f"\nDEBUG MODE: Processing only first {debug_max_slices} slices")
        frames = frames[:debug_max_slices]
        
        # Visualize centerline and frames
        fig = plt.figure(figsize=(15, 5))
        
        # 3D view of centerline
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(centerline[:,2], centerline[:,1], centerline[:,0], 'b-', linewidth=2, label='Centerline')
        
        # Plot frame positions
        frame_points = np.array([f['point'] for f in frames])
        ax1.scatter(frame_points[:,2], frame_points[:,1], frame_points[:,0], 
                   c='r', s=50, label='Sample points')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Centerline and Sample Points')
        ax1.legend()
        
        # 2D projections
        ax2 = fig.add_subplot(132)
        ax2.plot(centerline[:,2], centerline[:,1], 'b-', alpha=0.5)
        ax2.scatter(frame_points[:,2], frame_points[:,1], c=np.arange(len(frames)), 
                   cmap='rainbow', s=30)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY Projection')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(133)
        ax3.plot(centerline[:,0], centerline[:,2], 'b-', alpha=0.5)
        ax3.scatter(frame_points[:,0], frame_points[:,2], c=np.arange(len(frames)), 
                   cmap='rainbow', s=30)
        ax3.set_xlabel('Z')
        ax3.set_ylabel('X')
        ax3.set_title('ZX Projection')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('debug_centerline.png', dpi=150)
        plt.close()
        print("Saved: debug_centerline.png")
    
    patch_size = 2 * half_size
    straightened = np.zeros((len(frames), patch_size, patch_size), dtype=volume.dtype)
    
    iterator = tqdm(frames, desc="Extracting slices") if progress and not debug else frames
    
    for i, frame in enumerate(iterator):
        straightened[i] = extract_single_slice(
            volume,
            frame['point'],
            frame['axis0'],
            frame['axis1'],
            half_size,
            debug=debug,
            slice_idx=i
        )
        
        if debug and i == 0:
            # Additional check for first slice
            print(f"\nFirst slice detailed check:")
            print(f"  Frame keys: {frame.keys()}")
            print(f"  Tangent (axis2): {frame.get('axis2', 'N/A')}")
    
    if debug:
        # Summary statistics
        print(f"\n{'='*70}")
        print(f"EXTRACTION SUMMARY")
        print(f"{'='*70}")
        print(f"Extracted {len(straightened)} slices")
        print(f"Stack shape: {straightened.shape}")
        print(f"Overall statistics:")
        print(f"  Mean: {straightened.mean():.2f}")
        print(f"  Std: {straightened.std():.2f}")
        print(f"  Min: {straightened.min():.2f}")
        print(f"  Max: {straightened.max():.2f}")
        
        # Per-slice statistics
        print(f"\nPer-slice statistics:")
        for i in range(min(len(straightened), debug_max_slices)):
            print(f"  Slice {i:2d}: mean={straightened[i].mean():6.2f}, "
                  f"std={straightened[i].std():6.2f}, "
                  f"nonzero={np.count_nonzero(straightened[i]):5d}/{straightened[i].size}")
    
    return straightened, frames


def extract_and_save_slices(
    volume: np.ndarray,
    centerline: np.ndarray,
    output_dir: Path,
    organ_id: str = "organ",
    organ_type: str = "root",
    half_size: int = 64,
    step: float = 1.0,
    save_3d_stack: bool = True,
    save_individual: bool = False,
    save_metadata: bool = True,
    debug: bool = False,
    debug_max_slices: int = 10
) -> OrganMetadata:
    """
    Extract and save organ slices to disk.
    
    Args:
        volume: 3D volume (Z, Y, X)
        centerline: Path as (N, 3) array with (z, y, x) coordinates
        output_dir: Output directory
        organ_id: Identifier for this organ
        organ_type: 'root' or 'leaf'
        half_size: Half-width of patches
        step: Arc-length spacing
        save_3d_stack: Save as single 3D TIFF
        save_individual: Save each slice separately
        save_metadata: Save JSON metadata
        debug: Enable debug mode with detailed output
        debug_max_slices: Max slices to extract in debug mode
    
    Returns:
        OrganMetadata with extraction info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract
    straightened, frames = extract_organ_volume(
        volume, centerline, half_size, step, 
        progress=not debug, 
        debug=debug,
        debug_max_slices=debug_max_slices
    )
    
    if len(straightened) == 0:
        print(f"⚠️ No slices extracted for {organ_id}")
        return OrganMetadata(organ_id=organ_id, organ_type=organ_type)
    
    # Save 3D stack
    if save_3d_stack:
        stack_path = output_dir / f"{organ_id}_straightened.tif"
        tiff.imwrite(str(stack_path), straightened)
        print(f"  Saved 3D stack: {stack_path}")
    
    # Save individual slices
    if save_individual or debug:
        slices_dir = output_dir / f"{organ_id}_slices"
        slices_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            z, y, x = frame['point']
            filename = f"{organ_id}_slice_{i:04d}_z{int(z)}_y{int(y)}_x{int(x)}.tif"
            tiff.imwrite(str(slices_dir / filename), straightened[i])
    
    # Build metadata
    metadata = OrganMetadata(
        organ_id=organ_id,
        organ_type=organ_type,
        num_slices=len(frames),
        centerline_length=frames[-1]['arc_length'] if frames else 0,
        patch_size=2 * half_size,
        step=step
    )
    
    # Save metadata
    if save_metadata:
        meta_dict = metadata.to_dict()
        meta_dict['volume_shape'] = list(volume.shape)
        meta_dict['centerline'] = centerline.tolist()
        meta_dict['frames'] = [
            {
                'index': i,
                'center_zyx': f['point'].tolist(),
                'arc_length': f['arc_length']
            }
            for i, f in enumerate(frames)
        ]
        
        with open(output_dir / f"{organ_id}_metadata.json", 'w') as f:
            json.dump(meta_dict, f, indent=2)
    
    return metadata


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def load_centerline_csv(csv_path: Path) -> np.ndarray:
    """
    Load centerline from CSV file.
    
    Accepts formats:
        - With columns named X, Y, Slice (preferred)
        - With header: z,y,x
        - Without header: 3 columns assumed as z,y,x
        - Delimiter: comma or semicolon
    
    Removes duplicate consecutive slices (keeps only first point when multiple 
    points are on the same Z slice).
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Array of shape (N, 3) with (z, y, x) coordinates, duplicates removed
    """
    csv_path = Path(csv_path)
    
    # Try different delimiters
    for sep in [',', ';', '\t', ' ']:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            break
        except:
            continue
    else:
        raise ValueError(f"Could not parse CSV: {csv_path}")
    
    # Check for named columns (case-insensitive)
    cols_lower = {c.lower().strip(): c for c in df.columns}
    
    # Priority 1: Look for X, Y, Slice columns
    if 'x' in cols_lower and 'y' in cols_lower and 'slice' in cols_lower:
        x_col = cols_lower['x']
        y_col = cols_lower['y']
        z_col = cols_lower['slice']  # Slice = Z coordinate
        centerline = df[[z_col, y_col, x_col]].values.astype(np.float64)
    # Priority 2: Look for z, y, x columns
    elif 'z' in cols_lower and 'y' in cols_lower and 'x' in cols_lower:
        z_col = cols_lower['z']
        y_col = cols_lower['y']
        x_col = cols_lower['x']
        centerline = df[[z_col, y_col, x_col]].values.astype(np.float64)
    # Priority 3: Assume first 3 columns are z, y, x
    elif df.shape[1] >= 3:
        centerline = df.iloc[:, :3].values.astype(np.float64)
    else:
        raise ValueError(f"CSV must have columns 'X', 'Y', 'Slice' or 'z', 'y', 'x', or at least 3 columns: {csv_path}")
    
    # Remove consecutive duplicate slices (keep only first point per Z slice)
    if len(centerline) > 1:
        # Find where Z coordinate changes
        z_coords = centerline[:, 0]
        z_diff = np.diff(z_coords)
        
        # Keep first point and points where Z changes
        keep_mask = np.ones(len(centerline), dtype=bool)
        keep_mask[1:] = (z_diff != 0)  # Keep points where Z is different from previous
        
        n_removed = (~keep_mask).sum()
        if n_removed > 0:
            print(f"  Removed {n_removed} duplicate consecutive points (same Z slice)")
        
        centerline = centerline[keep_mask]
    
    return centerline


def process_batch(
    volume_path: Path,
    centerlines_dir: Path,
    output_dir: Path,
    half_size: int = 64,
    step: float = 1.0,
    save_3d_stack: bool = True,
    save_individual: bool = False
) -> pd.DataFrame:
    """
    Process all centerlines in a directory.
    
    Args:
        volume_path: Path to 3D TIFF volume
        centerlines_dir: Directory with CSV files (one per organ)
        output_dir: Base output directory
        half_size: Patch half-size
        step: Arc-length spacing
        save_3d_stack: Save 3D straightened volumes
        save_individual: Save individual slices
    
    Returns:
        DataFrame with processing summary
    """
    volume_path = Path(volume_path)
    centerlines_dir = Path(centerlines_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load volume
    print(f"Loading volume: {volume_path}")
    volume = tiff.imread(str(volume_path))
    print(f"  Shape: {volume.shape}, dtype: {volume.dtype}")
    
    # Find CSV files
    csv_files = sorted(list(centerlines_dir.glob("*.csv")))
    print(f"Found {len(csv_files)} centerline files")
    
    if not csv_files:
        print(f"⚠️ No CSV files in {centerlines_dir}")
        return pd.DataFrame()
    
    results = []
    
    for csv_path in csv_files:
        organ_id = csv_path.stem
        
        # Determine organ type from filename
        organ_type = "leaf" if "leaf" in organ_id.lower() else "root"
        
        print(f"\n{'='*60}")
        print(f"Processing: {organ_id} ({organ_type})")
        print(f"{'='*60}")
        
        try:
            # Load centerline
            centerline = load_centerline_csv(csv_path)
            print(f"  Loaded {len(centerline)} points")
            
            # Extract
            organ_output = output_dir / organ_id
            metadata = extract_and_save_slices(
                volume=volume,
                centerline=centerline,
                output_dir=organ_output,
                organ_id=organ_id,
                organ_type=organ_type,
                half_size=half_size,
                step=step,
                save_3d_stack=save_3d_stack,
                save_individual=save_individual,
                save_metadata=True
            )
            
            results.append({
                'organ_id': organ_id,
                'organ_type': organ_type,
                'status': 'success',
                'num_slices': metadata.num_slices,
                'centerline_length': metadata.centerline_length,
                'centerline_points': len(centerline)
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'organ_id': organ_id,
                'organ_type': organ_type,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "extraction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved: {summary_path}")
    
    return summary_df


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract straightened 3D organ images along centerlines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m rice_atlas.slice_extraction volume.tif centerlines/ output/
  
  # Custom parameters
  python -m rice_atlas.slice_extraction volume.tif centerlines/ output/ \\
      --half-size 128 --step 0.5 --save-individual
  
Centerline CSV format:
  z,y,x
  100,200,150
  101,201,151
  ...
        """
    )
    
    parser.add_argument("volume", type=Path, help="Path to 3D TIFF volume")
    parser.add_argument("centerlines_dir", type=Path, help="Directory with centerline CSVs")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--half-size", type=int, default=64,
                        help="Half-size of extracted patches (default: 64)")
    parser.add_argument("--step", type=float, default=1.0,
                        help="Spacing between slices in voxels (default: 1.0)")
    parser.add_argument("--save-individual", action="store_true",
                        help="Save individual slice TIFFs")
    parser.add_argument("--no-3d-stack", action="store_true",
                        help="Don't save 3D straightened stack")
    
    args = parser.parse_args()
    
    # Validate
    if not args.volume.exists():
        print(f"❌ Volume not found: {args.volume}")
        return 1
    
    if not args.centerlines_dir.is_dir():
        print(f"❌ Centerlines directory not found: {args.centerlines_dir}")
        return 1
    
    # Run
    summary = process_batch(
        volume_path=args.volume,
        centerlines_dir=args.centerlines_dir,
        output_dir=args.output_dir,
        half_size=args.half_size,
        step=args.step,
        save_3d_stack=not args.no_3d_stack,
        save_individual=args.save_individual
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    
    return 0


if __name__ == "__main__":
    exit(main())