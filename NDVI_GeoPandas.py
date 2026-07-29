"""
tif_crop_to_ndvi_png.py
GreenSight - GeoHackathon 2026[cite: 2]

Install:
    pip install rasterio pillow numpy scipy
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from pathlib import Path
import scipy.ndimage as ndimage


def find_valid_crop_center(tif_path, size=1024, red_band=3, grid=20,
                            min_valid_fraction=0.5):
    """
    Scan the image on a coarse grid to find a size x size region that
    actually contains real (non-NoData) data, instead of assuming the
    center of the full canvas has data.[cite: 2]
    """
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        best = None
        best_fraction = 0.0

        step_r = max(1, (src.height - size) // grid)
        step_c = max(1, (src.width - size) // grid)

        for gr in range(grid + 1):
            for gc in range(grid + 1):
                row_off = min(gr * step_r, max(0, src.height - size))
                col_off = min(gc * step_c, max(0, src.width - size))

                corners = [
                    (row_off, col_off),
                    (row_off, col_off + size - 64),
                    (row_off + size - 64, col_off),
                    (row_off + size - 64, col_off + size - 64),
                    (row_off + size // 2 - 32, col_off + size // 2 - 32),
                ]
                fractions = []

                for r, c in corners:
                    sample = src.read(
                        red_band,
                        window=Window(c, r, 64, 64),
                    )
                    fractions.append(1.0 if nodata is None else float((sample != nodata).mean()))

                valid_fraction = float(np.mean(fractions))
                all_clean = all(f >= 0.95 for f in fractions)

                if valid_fraction > best_fraction:
                    best_fraction = valid_fraction
                    best = (row_off + size // 2, col_off + size // 2)

                if best_fraction >= min_valid_fraction and all_clean:
                    print(f"Found strong candidate crop center at row={best[0]}, "
                          f"col={best[1]} ({valid_fraction*100:.0f}% average valid data; "
                          f"corners/center clean)")
                    return best

    if best is not None:
        print(f"Best candidate found: row={best[0]}, col={best[1]} "
              f"({best_fraction*100:.0f}% valid data - not fully clean, "
              f"consider a smaller crop size or manual center)")
    else:
        print("WARNING: could not find any region with real data.")
    return best


def crop_to_ndvi_png(tif_path, png_path, center_row=None, center_col=None,
                      size=1024, red_band=3, nir_band=4, green_band=2, blue_band=1,
                      scale_max=4095, auto_find_center=True):
    """
    Crop a size x size window from the .tif and save two images:
    1. False-color-infrared PNG (for NDVI calculation)
    2. True-color original PNG (with percentile contrast stretch for realistic brightness)
    """
    if center_row is None or center_col is None:
        if auto_find_center:
            print("No crop center given - scanning for a region with real data...")
            found = find_valid_crop_center(tif_path, size=size, red_band=red_band)
            if found is not None:
                center_row, center_col = found
        
        if center_row is None or center_col is None:
            with rasterio.open(tif_path) as src:
                center_row = center_row or src.height // 2
                center_col = center_col or src.width // 2

    with rasterio.open(tif_path) as src:
        row_off = max(0, min(center_row - size // 2, src.height - size))
        col_off = max(0, min(center_col - size // 2, src.width - size))
        actual_h = min(size, src.height - row_off)
        actual_w = min(size, src.width - col_off)
        window = Window(col_off, row_off, actual_w, actual_h)

        print(f"Cropping window: row_off={row_off}, col_off={col_off}, "
              f"size={actual_w}x{actual_h}")

        # Read bands[cite: 2]
        red = src.read(red_band, window=window).astype("float32")
        nir = src.read(nir_band, window=window).astype("float32")
        green = src.read(green_band, window=window).astype("float32")
        
        # Read blue band for true color[cite: 2]
        try:
            blue = src.read(blue_band, window=window).astype("float32")
        except IndexError:
            print("WARNING: Blue band not found. Falling back to green for the blue channel.")
            blue = green.copy()

        nodata = src.nodata

    # 1. False-Color Linear Scaling (Keeps math exact for compute_ndvi_from_png)
    def scale_false_color(arr):
        scaled = np.clip(arr / scale_max * 255, 0, 255)
        if nodata is not None:
            scaled[arr == nodata] = 0
        return scaled.astype("uint8")

    fc_r = scale_false_color(nir)     # NIR stored in R channel[cite: 2]
    fc_g = scale_false_color(red)     # RED stored in G channel[cite: 2]
    fc_b = scale_false_color(green)   # GREEN stored in B channel[cite: 2]
    rgb_false_color = np.dstack([fc_r, fc_g, fc_b])

    # 2. True-Color Percentile Contrast Stretch (Fixes dark original image)
    rgb_original = np.zeros((actual_h, actual_w, 3), dtype="uint8")
    raw_channels = [red, green, blue]

    valid_mask = (red != nodata) if nodata is not None else np.ones((actual_h, actual_w), dtype=bool)

    for c in range(3):
        ch = raw_channels[c]
        valid_pixels = ch[valid_mask]
        
        if len(valid_pixels) > 0:
            # 2nd and 98th percentile stretch brings natural daylight brightness
            p_low, p_high = np.percentile(valid_pixels, (2, 98))
            if p_high > p_low:
                ch_scaled = np.clip((ch - p_low) / (p_high - p_low) * 255.0, 0, 255)
            else:
                ch_scaled = np.clip(ch / scale_max * 255.0, 0, 255)
        else:
            ch_scaled = ch

        if nodata is not None:
            ch_scaled[~valid_mask] = 0
            
        rgb_original[:, :, c] = ch_scaled.astype("uint8")

    # Pad to exactly size x size if we hit an image edge[cite: 2]
    def pad_image(rgb_array):
        if rgb_array.shape[0] != size or rgb_array.shape[1] != size:
            padded = np.zeros((size, size, 3), dtype="uint8")
            padded[: rgb_array.shape[0], : rgb_array.shape[1], :] = rgb_array
            return padded
        return rgb_array

    rgb_false_color = pad_image(rgb_false_color)
    rgb_original = pad_image(rgb_original)

    # Prepare file paths[cite: 2]
    base, ext = os.path.splitext(png_path)
    fc_path = f"{base}_false_color{ext}"
    orig_path = f"{base}_original{ext}"

    # Save False Color[cite: 2]
    img_fc = Image.fromarray(rgb_false_color, mode="RGB")
    img_fc.save(fc_path)
    print(f"Saved false-color-infrared PNG -> {fc_path} ({size}x{size})")

    # Save Brightened True Color Original[cite: 2]
    img_orig = Image.fromarray(rgb_original, mode="RGB")
    img_orig.save(orig_path)
    print(f"Saved original true-color PNG (contrast enhanced) -> {orig_path} ({size}x{size})")

    return fc_path


def compute_ndvi_from_png(png_path, scale_max=4095):
    """
    Recover an APPROXIMATE NDVI directly from a PNG saved by
    crop_to_ndvi_png(), by reversing the 0-255 scaling back toward
    the original value range.[cite: 2]
    """
    img = np.array(Image.open(png_path).convert("RGB")).astype("float32")

    nir_approx = img[:, :, 0] / 255.0 * scale_max   # R channel -> NIR[cite: 2]
    red_approx = img[:, :, 1] / 255.0 * scale_max   # G channel -> RED[cite: 2]

    denom = nir_approx + red_approx
    denom[denom == 0] = 1e-6
    ndvi = (nir_approx - red_approx) / denom

    green_mask = ndvi >= 0.3

    print(f"NDVI from PNG: min={ndvi.min():.3f}, max={ndvi.max():.3f}, "
          f"mean={ndvi.mean():.3f}, green_pixels={green_mask.sum()}/{green_mask.size}")

    return ndvi, green_mask


def save_ndvi_mask(ndvi, output_path):
    """
    Saves a mask where Black (0) represents valid plantable areas
    and White (255) represents non-plantable areas or boundaries.
    Applies morphological operations for smooth grouping.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize mask as White (255 = Cannot plant)[cite: 2]
    mask = np.full(ndvi.shape, 255, dtype=np.uint8)

    h, w = ndvi.shape
    margin = max(10, min(h, w) // 20)

    # Extract inner region[cite: 2]
    inner = ndvi[margin:h-margin, margin:w-margin]

    # Target NDVI range for planting
    can_plant_mask = (inner >= 0.2) & (inner <= 0.4)

    # Morphological Grouping
    structure_5x5 = np.ones((5, 5))
    structure_3x3 = np.ones((3, 3))

    # Connect nearby plantable blobs
    can_plant_mask = ndimage.binary_closing(can_plant_mask, structure=structure_5x5)
    # Filter out tiny isolated specks (noise)
    can_plant_mask = ndimage.binary_opening(can_plant_mask, structure=structure_3x3)

    # Apply Black (0) where we CAN plant, and White (255) where we CANNOT
    mask[margin:h-margin, margin:w-margin] = np.where(can_plant_mask, 0, 255)

    Image.fromarray(mask, mode="L").save(output_path)
    print(f"Saved plantable mask (Black=Plant, White=No-Plant) to: {output_path}")

    return output_path, mask


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_tif = base_dir / "Theos2.tif"
    output_png_base = base_dir / "crop_1024.png"
    output_mask_png = base_dir / "plantable_mask.png"

    # 1. Generate false-color and brightened true-color PNGs[cite: 2]
    fc_png_path = crop_to_ndvi_png(str(input_tif), str(output_png_base), size=1024)

    # 2. Compute NDVI from false-color PNG[cite: 2]
    ndvi, green_mask = compute_ndvi_from_png(fc_png_path)
    print("\nNDVI successfully computed FROM THE PNG FILE - no need to reopen the original .tif.")

    # 3. Save plantable mask and return matrix[cite: 2]
    mask_path, mask_matrix = save_ndvi_mask(ndvi, output_mask_png)

    print(f"\nExtracted mask matrix with shape: {mask_matrix.shape}")
    print(f"Unique values in matrix: {np.unique(mask_matrix)}")

    # Optional: Save matrix values to CSV
    csv_path = base_dir / "plantable_mask_matrix.csv"
    np.savetxt(csv_path, mask_matrix, fmt="%d", delimiter=",")
    print(f"Saved raw matrix data to: {csv_path}")