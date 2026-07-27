"""
tif_interactive_viewer.py
GreenSight - GeoHackathon 2026

Purpose:
    Open a large .tif as an interactive, pannable, zoomable map in
    your browser - like an online TIF viewer (e.g. geotiff.io) - but
    running entirely locally, with your data never leaving your
    machine. Works on multi-GB files because it serves the image as
    small map tiles on demand (like Google Maps), not as one big
    loaded image.

How it's different from tif_to_png_preview.py:
    - tif_to_png_preview.py     -> ONE static downsampled image (fast look)
    - tif_interactive_viewer.py -> a real pan/zoom map in your browser,
      reading full resolution only for the small area currently on
      screen, tile by tile, on demand

Install:
    pip install localtileserver

Run:
    python tif_interactive_viewer.py
    A local HTML file opens automatically in your browser.
    Keep the terminal running while you use the viewer - closing it
    stops the tile server.
"""

import os
import webbrowser
from localtileserver import TileClient


VIEWER_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>GreenSight - TIF Viewer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
        #info {{
            position: absolute; top: 10px; right: 10px; z-index: 1000;
            background: white; padding: 8px 12px; border-radius: 6px;
            font-family: sans-serif; font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="info">GreenSight TIF Viewer<br/>{filename}</div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], {default_zoom});

        L.tileLayer('{tile_url}', {{
            minZoom: {min_zoom},
            maxZoom: {max_zoom},
            tileSize: 256,
            attribution: 'GreenSight local tile server'
        }}).addTo(map);
    </script>
</body>
</html>
"""


def launch_viewer(tif_path, band_indices=(3, 2, 1), stretch="linear",
                   html_out="tif_viewer.html", open_browser=True):
    """
    Start a local tile server for the given .tif, generate a
    standalone HTML page with a Leaflet pan/zoom map pointed at it,
    and open it in the default browser.

    band_indices: which bands to display as R, G, B. Default (3,2,1)
    assumes Blue=1, Green=2, Red=3, NIR=4 -> natural color display.
    """
    tif_path = os.path.abspath(tif_path)
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"File not found: {tif_path}")

    print(f"Starting local tile server for: {tif_path}")
    client = TileClient(tif_path)

    tile_url = client.get_tile_url(indexes=list(band_indices), stretch=stretch)
    center_lat, center_lon = client.center()

    html = VIEWER_HTML_TEMPLATE.format(
        filename=os.path.basename(tif_path),
        center_lat=center_lat,
        center_lon=center_lon,
        default_zoom=client.default_zoom,
        min_zoom=client.min_zoom,
        max_zoom=client.max_zoom,
        tile_url=tile_url,
    )

    html_path = os.path.abspath(html_out)
    with open(html_path, "w") as f:
        f.write(html)

    print(f"Viewer page written -> {html_path}")
    print(f"Tile server running at -> {client.server_base_url}")
    print("\nKeep this script running while you use the viewer.")
    print("Press Ctrl+C or close the terminal to stop the server.")

    if open_browser:
        webbrowser.open(f"file://{html_path}")

    return client, html_path


if __name__ == "__main__":
    # ---- EDIT THIS PATH BEFORE RUNNING ----
    INPUT_TIF = r"images/theos_image01.tif"
    # -----------------------------------------

    client, html_path = launch_viewer(INPUT_TIF, band_indices=(3, 2, 1))

    try:
        input("\nPress Enter to stop the server and exit...\n")
    except KeyboardInterrupt:
        pass