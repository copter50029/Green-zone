"""
greensight_demo.py
GreenSight - GeoHackathon 2026

Purpose:
    Build a self-contained DEMO of the GreenSight pipeline:
      1. Grid-cut the full .tif into non-overlapping 1024x1024 tiles
      2. Auto-select 3 tiles that actually contain real data (not NoData)
      3. For each of the 3 tiles, generate:
           - a viewable false-color crop (NIR/RED/GREEN composite)
           - an NDVI colormap visualization (so NDVI values are visually
             checkable, not just a black/white guess)
           - a black/white plantability mask:
               BLACK = candidate area where a tree COULD be planted
                       (low NDVI - grey/bare/abandoned zone)
               WHITE = area where planting is NOT recommended here
                       (high NDVI - already vegetated / existing green zone)
      4. Package everything into one HTML demo page, plus a link to the
         full-image interactive pan/zoom viewer (see tif_interactive_viewer.py)

    Does NOT process the whole multi-GB image - only the 3 selected
    1024x1024 tiles are ever read at full resolution, as requested.

IMPORTANT NOTE ON NDVI ACCURACY (read before trusting the mask):
    Plain NDVI thresholding can genuinely misclassify some pixels:
      - Certain rooftop materials (especially bright/reflective or some
        painted metal roofs) reflect more NIR than typical impervious
        surfaces, which can push their NDVI up toward "vegetation"
        range even though they are buildings, not plants.
      - Bare/waste land with sparse dry grass or mixed soil can show
        LOWER NDVI than expected, sometimes looking similar to bare
        pavement.
    This is a known, real limitation of using NDVI alone - not a bug
    in this script. The NDVI colormap output is included specifically
    so you can visually sanity-check the mask against the real crop,
    rather than trusting the black/white result blindly. If this
    matters for your final accuracy requirement (>=80%), consider
    adding a building mask (e.g. from OSM footprints) to exclude
    known structures from the "plantable" class, on top of NDVI.

Install:
    pip install rasterio numpy pillow matplotlib
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import matplotlib
from scipy import ndimage


# =================================================================
# STEP 1: Grid-cut the image into non-overlapping tiles
# =================================================================
def grid_cut_tiles(tif_path, size=1024):
    """Return a list of (row_off, col_off) covering the full image in a grid."""
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width

    tiles = []
    for row_off in range(0, height, size):
        for col_off in range(0, width, size):
            # keep only full-size tiles (skip partial edge tiles for a clean demo)
            if row_off + size <= height and col_off + size <= width:
                tiles.append((row_off, col_off))
    return tiles


# =================================================================
# STEP 2: Keep only tiles that actually contain real (non-NoData) data
# 5-point check (center + 4 corners) so partially-empty tiles near
# the footprint edge are correctly rejected, not just center-checked.
# =================================================================
def is_tile_valid(src, row_off, col_off, size, red_band, min_valid_fraction=0.98):
    nodata = src.nodata
    if nodata is None:
        return True

    check_size = 48
    points = [
        (row_off + size // 2 - check_size // 2, col_off + size // 2 - check_size // 2),  # center
        (row_off, col_off),                                                              # top-left
        (row_off, col_off + size - check_size),                                          # top-right
        (row_off + size - check_size, col_off),                                          # bottom-left
        (row_off + size - check_size, col_off + size - check_size),                      # bottom-right
    ]
    for r, c in points:
        sample = src.read(red_band, window=Window(c, r, check_size, check_size))
        if (sample == nodata).mean() > (1 - min_valid_fraction):
            return False
    return True


def select_demo_tiles(tif_path, n=3, size=1024, red_band=3, spread=True):
    """
    Find tiles that pass the validity check and pick n of them, spread
    out across the valid list so the 3 demo tiles aren't all clustered
    in one corner of the image.
    """
    all_tiles = grid_cut_tiles(tif_path, size=size)

    with rasterio.open(tif_path) as src:
        valid_tiles = [
            t for t in all_tiles
            if is_tile_valid(src, t[0], t[1], size, red_band)
        ]

    print(f"Grid-cut into {len(all_tiles)} tiles total, {len(valid_tiles)} contain real data.")

    if not valid_tiles:
        raise RuntimeError("No valid (data-containing) tiles found - check band/nodata settings.")

    if len(valid_tiles) <= n:
        return valid_tiles

    if spread:
        idxs = np.linspace(0, len(valid_tiles) - 1, n).astype(int)
        return [valid_tiles[i] for i in idxs]
    else:
        return valid_tiles[:n]


# =================================================================
# STEP 3: Per-tile outputs - crop / NDVI colormap / black-white mask
# =================================================================
def scale_uint8(arr, scale_max, nodata=None):
    scaled = np.clip(arr / scale_max * 255, 0, 255)
    if nodata is not None:
        scaled[arr == nodata] = 0
    return scaled.astype("uint8")


def remove_small_regions(binary_mask, min_pixels):
    """
    Remove connected regions smaller than min_pixels from a boolean
    mask, flipping them to the opposite value. Used to clean up noise
    in the plantability mask - both tiny misclassified specks (small
    'not plantable' islands sitting inside a plantable area, likely
    noise rather than a real feature) and impractically small
    plantable patches (too small to actually plant a tree in).

    Uses scipy.ndimage.label, which has a stable, version-independent
    API (unlike skimage.morphology.remove_small_objects, whose
    min_size/max_size parameter was renamed in skimage 0.26 - the
    same kind of API break that caused the earlier matplotlib error).
    """
    labeled, n_regions = ndimage.label(binary_mask)
    if n_regions == 0:
        return binary_mask.copy()

    sizes = ndimage.sum(binary_mask, labeled, range(1, n_regions + 1))
    cleaned = binary_mask.copy()
    for region_id, size in enumerate(sizes, start=1):
        if size < min_pixels:
            cleaned[labeled == region_id] = False
    return cleaned


def process_tile(tif_path, row_off, col_off, size, out_dir, tile_id,
                  red_band=3, nir_band=4, green_band=2,
                  scale_max=4095, ndvi_threshold=0.3,
                  min_not_plantable_pixels=80, min_plantable_pixels=200):
    with rasterio.open(tif_path) as src:
        window = Window(col_off, row_off, size, size)
        red = src.read(red_band, window=window).astype("float32")
        nir = src.read(nir_band, window=window).astype("float32")
        green = src.read(green_band, window=window).astype("float32")
        nodata = src.nodata

    os.makedirs(out_dir, exist_ok=True)

    # --- (a) false-color crop, viewable like a normal image ---
    crop_rgb = np.dstack([
        scale_uint8(nir, scale_max, nodata),
        scale_uint8(red, scale_max, nodata),
        scale_uint8(green, scale_max, nodata),
    ])
    crop_path = os.path.join(out_dir, f"tile{tile_id}_crop.png")
    Image.fromarray(crop_rgb, mode="RGB").save(crop_path)

    # --- (b) NDVI colormap (visually checkable, not just binary) ---
    denom = nir + red
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom
    ndvi_norm = np.clip((ndvi + 1) / 2, 0, 1)  # map [-1,1] -> [0,1] for colormap
    ndvi_colored = (matplotlib.colormaps["RdYlGn"](ndvi_norm)[:, :, :3] * 255).astype("uint8")
    ndvi_path = os.path.join(out_dir, f"tile{tile_id}_ndvi.png")
    Image.fromarray(ndvi_colored, mode="RGB").save(ndvi_path)

    # --- (c) black/white plantability mask, noise-cleaned ---
    # BLACK (0)   = plantable candidate (NDVI below threshold - grey/bare zone)
    # WHITE (255) = not recommended here (NDVI at/above threshold - already green)
    raw_not_plantable = ndvi >= ndvi_threshold  # True = white/not-plantable

    # Pass 1: remove tiny "not plantable" specks (likely noise, not a
    # real feature) sitting inside otherwise-plantable areas.
    not_plantable_clean = remove_small_regions(raw_not_plantable, min_not_plantable_pixels)

    # Pass 2: remove tiny plantable islands too small to practically
    # plant a tree in (flips them to "not plantable").
    plantable = ~not_plantable_clean
    plantable_clean = remove_small_regions(plantable, min_plantable_pixels)

    mask = np.where(~plantable_clean, 255, 0).astype("uint8")
    mask_path = os.path.join(out_dir, f"tile{tile_id}_mask.png")
    Image.fromarray(mask, mode="L").save(mask_path)

    plantable_pct = (mask == 0).mean() * 100
    print(f"Tile {tile_id}: row_off={row_off}, col_off={col_off}, "
          f"plantable area = {plantable_pct:.1f}%")

    return {
        "tile_id": tile_id, "row_off": row_off, "col_off": col_off,
        "crop_path": crop_path, "ndvi_path": ndvi_path, "mask_path": mask_path,
        "plantable_pct": round(plantable_pct, 1),
    }


# =================================================================
# STEP 4: Build the HTML demo gallery
# =================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>GreenSight Demo</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background:#0e0e0e; color:#eee; margin:0; padding:24px; }}
  h1 {{ color: #7ed957; }}
  .full-link {{ display:block; margin-bottom:28px; padding:14px; background:#1a1a1a;
                border-radius:8px; color:#7ed957; text-decoration:none; }}
  .tile {{ display:flex; gap:12px; margin-bottom:32px; background:#1a1a1a; padding:16px; border-radius:8px; }}
  .tile img {{ width: 300px; height:300px; object-fit:cover; border-radius:6px; }}
  .tile .col {{ text-align:center; }}
  .tile .col p {{ margin:6px 0 0; font-size:13px; color:#aaa; }}
  .badge {{ display:inline-block; background:#2a2a2a; padding:2px 8px; border-radius:4px; font-size:12px; }}
  .note {{ font-size:13px; color:#999; max-width:900px; line-height:1.5; }}
</style>
</head>
<body>
  <h1>GreenSight - Demo (3 sample tiles, {size}x{size} px each)</h1>
  <a class="full-link" href="{full_preview_path}" target="_blank">
    View full image (downsampled preview) &rarr;
  </a>
  <p class="note">For a full pan/zoom interactive view of the entire image
  (not just downsampled), run <code>tif_interactive_viewer.py</code> separately.</p>

  {tiles_html}

  <p class="note"><b>NDVI accuracy note:</b> the black/white mask uses a simple
  NDVI threshold ({threshold}). Some rooftop materials can reflect enough NIR to
  falsely resemble vegetation, and sparse/dry bare land can show lower NDVI than
  expected. Use the NDVI colormap column to visually sanity-check the mask -
  it is not guaranteed to be pixel-perfect from NDVI alone.</p>
</body>
</html>
"""

TILE_TEMPLATE = """
  <div class="tile">
    <div class="col"><img src="{crop_rel}"><p>Tile {tile_id}<br>False-color crop</p></div>
    <div class="col"><img src="{ndvi_rel}"><p>NDVI colormap<br>(red=low, green=high)</p></div>
    <div class="col"><img src="{mask_rel}"><p>Plantability mask<br>
        <span class="badge">black = plantable ({plantable_pct}%)</span></p></div>
  </div>
"""


def build_demo(tif_path, out_dir, n_tiles=3, size=1024, full_preview_path=None,
                scale_max=4095, ndvi_threshold=0.3,
                min_not_plantable_pixels=80, min_plantable_pixels=200):
    os.makedirs(out_dir, exist_ok=True)

    selected = select_demo_tiles(tif_path, n=n_tiles, size=size)

    results = []
    for i, (row_off, col_off) in enumerate(selected, start=1):
        result = process_tile(tif_path, row_off, col_off, size, out_dir, i,
                               scale_max=scale_max, ndvi_threshold=ndvi_threshold,
                               min_not_plantable_pixels=min_not_plantable_pixels,
                               min_plantable_pixels=min_plantable_pixels)
        results.append(result)

    tiles_html = "".join(
        TILE_TEMPLATE.format(
            crop_rel=os.path.basename(r["crop_path"]),
            ndvi_rel=os.path.basename(r["ndvi_path"]),
            mask_rel=os.path.basename(r["mask_path"]),
            tile_id=r["tile_id"], plantable_pct=r["plantable_pct"],
        ) for r in results
    )

    html = HTML_TEMPLATE.format(
        size=size,
        full_preview_path=full_preview_path or "preview.png",
        tiles_html=tiles_html,
        threshold=ndvi_threshold,
    )

    html_path = os.path.join(out_dir, "greensight_demo.html")
    with open(html_path, "w") as f:
        f.write(html)

    print(f"\nDemo built -> {html_path}")
    print(f"Open that file in a browser to view the demo.")
    return html_path, results


if __name__ == "__main__":
    # ---- EDIT THESE BEFORE RUNNING ----
    INPUT_TIF = r"C:\Users\User\OneDrive\Desktop\New folder\geohackathon\Theos2.tif"
    OUT_DIR = r"C:\Users\User\OneDrive\Desktop\New folder\geohackathon\demo"
    FULL_PREVIEW = "preview.png"  # generate this first with tif_to_png_preview.py
    # ------------------------------------

    build_demo(INPUT_TIF, OUT_DIR, n_tiles=3, size=1024,
               full_preview_path=FULL_PREVIEW, scale_max=4095, ndvi_threshold=0.3)