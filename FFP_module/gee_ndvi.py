import os
import ee
import numpy as np
import pyproj
from shapely.geometry import Polygon
import pickle
import xarray as xr
import geemap

# -------------------------
# Projection
# -------------------------
to_merc = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# -------------------------
# Init GEE
# -------------------------
def init_gee(config):
    try:
        ee.Initialize(project=config["gee"]["project_id"])
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=config["gee"]["project_id"])

# -------------------------
# Geometry
# -------------------------
def contour_to_polygon(xr, yr, lat, lon):
    x0, y0 = to_merc.transform(lon, lat)

    polygons = []

    for x_seg, y_seg in zip(xr, yr):
        x_abs = np.array(x_seg) + x0
        y_abs = np.array(y_seg) + y0

        lon_arr, lat_arr = to_wgs84.transform(x_abs, y_abs)
        coords = list(zip(lon_arr, lat_arr))

        polygons.append(coords)

    return polygons


def shapely_to_ee(coords):
    return ee.Geometry.Polygon(coords)


def load_monthly_ffp(config):

    base_path = config["database"]["base_path"]
    site = config["database"]["site_name"][0]
    year = config["database"]["years"][0]

    pickle_dir = os.path.join(
        base_path,
        "FFP_output",
        site,
        str(year),
        "pickle"
    )

    monthly_contours = {}

    if not os.path.exists(pickle_dir):
        raise FileNotFoundError(f"FFP pickle folder not found: {pickle_dir}")

    for file in os.listdir(pickle_dir):
        if file.endswith(".pkl"):
            month_key = file.replace("FFP_full_", "").replace(".pkl", "")
            file_path = os.path.join(pickle_dir, file)

            with open(file_path, "rb") as f:
                monthly_contours[month_key] = pickle.load(f)

    return monthly_contours


def get_union_geometry(monthly_contours, lat, lon):

    geoms = []

    for c in monthly_contours.values():
        xr, yr = c["xr"], c["yr"]

        if xr is None or yr is None:
            continue

        polygons = contour_to_polygon(xr, yr, lat, lon)

        for poly in polygons:
            geoms.append(shapely_to_ee(poly))

    return ee.FeatureCollection(geoms).geometry()


def get_ffp_union_geometry(config, monthly_contours=None, lat=None, lon=None):

    """
    Smart wrapper:
    - If monthly_contours is None → load from pickle
    - Else → use provided FFP output
    """

    if lat is None:
        lat = config["lat_lon"]["lat"]

    if lon is None:
        lon = config["lat_lon"]["lon"]

    # -------------------------
    # Decide source of FFP
    # -------------------------
    if monthly_contours is None:

        print("[INFO] Loading FFP from pickle...")
        monthly_contours = load_monthly_ffp(config)

    else:
        print("[INFO] Using in-memory FFP output...")

    # -------------------------
    # Build geometry
    # -------------------------
    return get_union_geometry(monthly_contours, lat, lon)

# -------------------------
# Landsat preprocessing
# -------------------------
def mask(image):

    qa = image.select("QA_PIXEL")

    # Bits (C2 L2):
    # 1 = dilated cloud
    # 2 = cirrus
    # 3 = cloud
    # 4 = cloud shadow

    cloud_mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )

    # Scale reflectance + apply mask
    image = image.updateMask(cloud_mask)

    return image

def add_ndvi(image):

    # scale factor
    red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    nir = image.select("SR_B5").multiply(0.0000275).add(-0.2)

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

    return image.addBands(ndvi)

def collection(start, end, geom, config):

    return (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterDate(start, end)
        .filterBounds(geom)
        .map(mask)
        .map(add_ndvi)
    )

# -------------------------
# Monthly timeline
# -------------------------
def generate_months(years):

    months = []

    for y in years:
        for m in range(1, 13):

            start = f"{y}-{m:02d}-01"

            if m == 12:
                end = f"{y+1}-01-01"
            else:
                end = f"{y}-{m+1:02d}-01"

            months.append((f"{y}-{m:02d}", start, end))

    return months

# -------------------------
# Single NDVI map
# -------------------------
def single_ndvi(geom, config):

    years = config["database"]["years"]
    start = f"{years}-01-01"
    end = f"{years}-12-31"

    return collection(start, end, geom, config).select("NDVI").median()

# -------------------------
# Monthly NDVI maps
# -------------------------
def monthly_ndvi(geom, config):

    out = {}
    years = config["database"]["years"]
    start = f"{years}-01-01"
    end = f"{years}-12-31"

    for label, start, end in generate_months(config["database"]["years"]):

        img = collection(start, end, geom, config).select("NDVI").median()
        out[label] = img

    return out

# -------------------------
# Time series
# -------------------------
def monthly_ndvi_timeseries(geom, config):

    years = config["database"]["years"]
    results = []

    for label, start, end in generate_months(years):

        img = collection(start, end, geom, config).select("NDVI").median()

        mean_dict = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=30,
            maxPixels=1e9
        )

        ndvi = mean_dict.get("NDVI").getInfo()

        results.append({
            "month": label,
            "NDVI": ndvi
        })

    return results

# def ndvi_timeseries(monthly_contours, config):

#     lat = config["lat_lon"]["lat"]
#     lon = config["lat_lon"]["lon"]

#     results = []

#     for label, start, end in generate_months(config["database"]["years"]):

#         if label not in monthly_contours:
#             continue

#         c = monthly_contours[label]

#         xr, yr = c["xr"], c["yr"]
#         if xr is None or yr is None:
#             continue

#         poly = contour_to_polygon(xr, yr, lat, lon)
#         geom = shapely_to_ee(poly)

#         img = collection(start, end, geom, config).select("NDVI").median()

#         val = img.reduceRegion(
#             reducer=ee.Reducer.mean(),
#             geometry=geom,
#             scale=config["gee"]["scale"],
#             maxPixels=1e9
#         )

#         results.append({
#             "month": label,
#             "ndvi_mean": val.get("NDVI").getInfo()
#         })

#     return results

# -------------------------
# Export GeoTIFF
# -------------------------
# def plot_ffp_with_ndvi(
#     FFP,
#     ndvi_img,
#     lat,
#     lon,
#     site_name,
#    year,
#     config
# ):
#     import numpy as np
#     import os
#     import matplotlib.pyplot as plt
#     import contextily as ctx
#     import geemap
#     import ee
#     import pyproj

#     # -----------------------------
#     # CRS transformers
#     # -----------------------------
#     wgs84 = pyproj.CRS("EPSG:4326")
#     merc = pyproj.CRS("EPSG:3857")

#     to_merc = pyproj.Transformer.from_crs(
#         wgs84, merc, always_xy=True
#     ).transform

#     # tower location in meters
#     x0, y0 = to_merc(lon, lat)

#     # -----------------------------
#     # FFP grid
#     # -----------------------------
#     X_abs = FFP["x_2d"] + x0
#     Y_abs = FFP["y_2d"] + y0
#     Z = FFP["fclim_2d"]

#     # -----------------------------
#     # NDVI → Web Mercator
#     # -----------------------------
#     ndvi_merc = ndvi_img.reproject(crs="EPSG:3857", scale=30)

#     region = ee.Geometry.Rectangle(
#         [X_abs.min(), Y_abs.min(), X_abs.max(), Y_abs.max()],
#         proj="EPSG:3857",
#         geodesic=False
#     )

#     ndvi_np = geemap.ee_to_numpy(
#         ndvi_merc,
#         region=region,
#         scale=30
#     )

#     if ndvi_np is None:
#         print("NDVI extraction failed")
#         return

#     ndvi_np = ndvi_np.squeeze()

#     # -----------------------------
#     # Plot
#     # -----------------------------
#     fig, ax = plt.subplots(figsize=(10, 8))

#     # ---- NDVI background ----
#     extent = [
#         X_abs.min(), X_abs.max(),
#         Y_abs.min(), Y_abs.max()
#     ]

#     im = ax.imshow(
#         ndvi_np,
#         extent=extent,
#         origin="upper",
#         cmap="RdYlGn",
#         alpha=0.6
#     )

#     cbar_ndvi = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
#     cbar_ndvi.set_label("NDVI")

#     # ---- Footprint weights ----
#     cs = ax.contourf(
#         X_abs, Y_abs, Z,
#         levels=100,
#         cmap="viridis",
#         alpha=0.25
#     )

#     cbar_fp = plt.colorbar(cs, ax=ax, fraction=0.03, pad=0.08)
#     cbar_fp.set_label("Footprint Weight (m$^{-2}$)")

#     # ---- Contour lines ----
#     for i, rs_val in enumerate(FFP["rs"]):
#         if FFP["xr"][i] is not None and FFP["yr"][i] is not None:
#             xr_arr = np.array(FFP["xr"][i]) + x0
#             yr_arr = np.array(FFP["yr"][i]) + y0
#             ax.plot(
#                 xr_arr, yr_arr,
#                 linewidth=1.5,
#                 label=f"{int(rs_val*100)}%"
#             )

#     # ---- Tower ----
#     ax.scatter(x0, y0, c="red", marker="^", s=80, label="Tower")

#     # ---- Zoom to 80% contour ----
#     rs = np.array(FFP["rs"])
#     idx80 = np.argmin(np.abs(rs - 0.8))

#     if FFP["xr"][idx80] is not None:
#         xr80 = np.array(FFP["xr"][idx80]) + x0
#         yr80 = np.array(FFP["yr"][idx80]) + y0

#         margin = 50
#         ax.set_xlim(xr80.min() - margin, xr80.max() + margin)
#         ax.set_ylim(yr80.min() - margin, yr80.max() + margin)

#     # ---- Basemap ----
#     google_sat = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
#     ctx.add_basemap(ax, source=google_sat, crs="EPSG:3857")

#     # ---- Labels ----
#     ax.set_xlabel("x [m]")
#     ax.set_ylabel("y [m]")
#     ax.set_title(f"NDVI + Footprint Map ({site_name}, {year})")
#     ax.legend()

#     # -----------------------------
#     # Save
#     # -----------------------------
#     map_path = os.path.join(
#         config["database"]["base_path"],
#         "FFP_output",
#         site_name,
#         str(year),
#         "map",
#         "NDVI_FFP_maps"
#     )

#     os.makedirs(map_path, exist_ok=True)

#     map_file = os.path.join(
#         map_path,
#         f"NDVI_FFP_{site_name}_{month_str}.png"
#     )

#     plt.savefig(map_file, dpi=120, bbox_inches="tight")
#     plt.close(fig)

#     print(f"Saved map: {map_file}")