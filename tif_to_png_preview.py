"""
tif_to_png_preview.py
GreenSight - GeoHackathon 2026

Purpose:
    Turn a large, multi-band GeoTIFF into a normal PNG image you can
    open like any other picture (Photos app, browser, etc.) - even
    if the source .tif is several GB.

    Key trick: rasterio can read a DECIMATED (downsampled) version of
    the raster directly, without ever loading the full-resolution
    image into memory. We ask for, say, a 2000px-wide preview, and
    rasterio + GDAL do the downsampling efficiently while reading.

Install:
    pip install rasterio pillow numpy
"""

import os
import numpy as np
import rasterio
from PIL import Image


def stretch_to_uint8(band_array, nodata=None, low_pct=2, high_pct=98):
    """
    Contrast-stretch a single band to 0-255 using percentile clipping,
    so the preview looks like a normal photo instead of a washed-out
    or too-dark image. NoData pixels are excluded from the stretch
    calculation and set to 0 (black) in the output.
    """
    arr = band_array.astype("float32")

    if nodata is not None:
        valid_mask = arr != nodata
    else:
        valid_mask = np.ones_like(arr, dtype=bool)

    if not valid_mask.any():
        return np.zeros_like(arr, dtype="uint8")

    valid_values = arr[valid_mask]
    lo, hi = np.percentile(valid_values, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1  # avoid divide-by-zero on flat bands

    stretched = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
    stretched[~valid_mask] = 0  # NoData -> black

    return stretched.astype("uint8")


def tif_to_png(tif_path, png_path, preview_width=2000,
               bands=(3, 2, 1), low_pct=2, high_pct=98):
    """
    Read a downsampled version of the given bands (default: Red=3,
    Green=2, Blue=1, i.e. natural-color order) and save as a PNG.

    preview_width controls how large the output image is - larger
    values give a sharper preview but take longer / use more memory.
    2000px is normally plenty for "does this look right" viewing.
    """
    with rasterio.open(tif_path) as src:
        scale = preview_width / src.width
        out_height = int(src.height * scale)
        out_width = preview_width

        print(f"Reading downsampled preview: {out_width} x {out_height} px "
              f"(source is {src.width} x {src.height} px, scale={scale:.4f})")

        channels = []
        for band in bands:
            # out_shape triggers a DECIMATED read - GDAL downsamples
            # while reading, so the full-resolution data is never
            # pulled into memory.
            data = src.read(band, out_shape=(out_height, out_width))
            channels.append(stretch_to_uint8(data, nodata=src.nodata,
                                              low_pct=low_pct, high_pct=high_pct))

        rgb = np.dstack(channels)  # (height, width, 3)

    img = Image.fromarray(rgb, mode="RGB")
    img.save(png_path)
    print(f"Saved preview -> {png_path}")
    return png_path


if __name__ == "__main__":
    # ---- EDIT THESE BEFORE RUNNING ----
    INPUT_TIF = r"images/theos_image01.tif"
    OUTPUT_PNG = r"images/preview.png"
    # ------------------------------------

    # bands=(3,2,1) assumes Blue=1, Green=2, Red=3, NIR=4 -> natural color.
    # If colors look wrong/inverted, check your actual band order first.
    tif_to_png(INPUT_TIF, OUTPUT_PNG, preview_width=2000, bands=(3, 2, 1))