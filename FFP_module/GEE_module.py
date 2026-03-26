import numpy as np
import pyproj
from shapely.geometry import Polygon

#Step 1: convert monthly 80% contour to lon/lat polygon

# tower location
lat0, lon0 = config["lat_lon"]["lat"], config["lat_lon"]["lon"]

# transformers
to_merc = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

x0, y0 = to_merc.transform(lon0, lat0)

def contour_to_lonlat_polygon(xr, yr, x0, y0):
    # relative meters -> absolute meters
    x_abs = np.array(xr) + x0
    y_abs = np.array(yr) + y0

    # meters -> lon/lat
    lon, lat = to_wgs84.transform(x_abs, y_abs)

    coords = list(zip(lon, lat))
    return Polygon(coords)

#Step 2: Initialize GEE and convert to GEE geometry

import ee
ee.Initialize()

def shapely_to_ee_polygon(poly):
    coords = list(poly.exterior.coords)
    return ee.Geometry.Polygon(coords)

#Step 3: Preprocess Landsat data
def mask_landsat_l2(image):
    qa = image.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    cloud_shadow = qa.bitwiseAnd(1 << 4).neq(0)
    snow = qa.bitwiseAnd(1 << 5).neq(0)
    mask = cloud.Or(cloud_shadow).Or(snow).Not()
    return image.updateMask(mask)

def add_ndvi_l8_l9(image):
    red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    nir = image.select("SR_B5").multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    return image.addBands(ndvi)

#Step 4:  Extract monthly NDVI for each monthly footprint

def monthly_ndvi_from_footprint(poly, start_date, end_date):
    geom = shapely_to_ee_polygon(poly)

    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .map(mask_landsat_l2)
        .map(add_ndvi_l8_l9)
    )

    ndvi_img = collection.select("NDVI").median()

    stats = ndvi_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=30,
        maxPixels=1e9
    )

    count = ndvi_img.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geom,
        scale=30,
        maxPixels=1e9
    )

    return {
        "ndvi_mean": stats.get("NDVI").getInfo(),
        "pixel_count": count.get("NDVI").getInfo()
    }

#Step 5: Run and Integrate into pipeline
results = []

for month_str, contour in monthly_contours.items():
    xr80 = contour["xr"]
    yr80 = contour["yr"]

    if xr80 is None or yr80 is None:
        continue

    poly = contour_to_lonlat_polygon(xr80, yr80, x0, y0)

    start = pd.to_datetime(month_str + "-01")
    end = start + pd.offsets.MonthBegin(1)

    out = monthly_ndvi_from_footprint(
        poly,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d")
    )

    results.append({
        "month": month_str,
        "ndvi_mean": out["ndvi_mean"],
        "pixel_count": out["pixel_count"]
    })

ndvi_df = pd.DataFrame(results)
print(ndvi_df)