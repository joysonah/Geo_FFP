import os
import glob
import re

import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np


# ============================================================
# USER OPTIONS
# ============================================================

START_DATE = "2021-01-01"
END_DATE   = "2025-12-31"

NDVI_SENSOR = "Landsat"      # "Landsat" or "Sentinel"
LAI_SENSOR  = "Sentinel"     # "Landsat" or "Sentinel"

CONTOUR_VALUE = 0.8

MONTH_FIELD = "month"
CONTOUR_FIELD = "rs"


# ============================================================
# GEE INITIALIZATION
# ============================================================

def init_gee(config):

    try:
        ee.Initialize(
            project=config["gee"]["project_id"]
        )

    except Exception:
        ee.Authenticate()

        ee.Initialize(
            project=config["gee"]["project_id"]
        )


# ============================================================
# LOAD FFP SHAPEFILE
# ============================================================

def load_ffp_shapefile(config):

    base_path = config["database"]["base_path"]
    site = config["database"]["site_name"][0]

    shp_dir = os.path.join(
        base_path,
        "FFP_output",
        site,
        "shapefile"
    )

    shp_files = glob.glob(
        os.path.join(shp_dir, "*.shp")
    )

    if not shp_files:
        raise FileNotFoundError(
            f"No shapefile found in:\n{shp_dir}"
        )

    print(f"[INFO] Found {len(shp_files)} shapefile(s)")

    gdfs = []

    for shp in shp_files:

        print(f"[INFO] Reading: {os.path.basename(shp)}")

        gdf = gpd.read_file(shp)

        if gdf.crs is None:
            raise ValueError(
                f"CRS missing:\n{shp}"
            )

        gdfs.append(
            gdf.to_crs("EPSG:4326")
        )

    # Combine all shapefiles
    gdf = gpd.GeoDataFrame(
        pd.concat(
            gdfs,
            ignore_index=True
        ),
        crs="EPSG:4326"
    )

    return gdf


# ============================================================
# GET MONTH FROM ATTRIBUTE TABLE
# ============================================================

def normalize_month(value):

    """
    Converts different month representations to MM.

    Examples:
        1       -> 01
        01      -> 01
        January -> 01
        Jan     -> 01
        2024-01 -> 01
    """

    value = str(value).strip()

    # Numeric month
    try:
        number = int(float(value))

        if 1 <= number <= 12:
            return f"{number:02d}"

    except Exception:
        pass

    # YYYY-MM
    match = re.search(
        r"(0?[1-9]|1[0-2])$",
        value
    )

    if match:
        return f"{int(match.group(1)):02d}"

    # Month names
    months = {
        "jan": "01",
        "january": "01",
        "feb": "02",
        "february": "02",
        "mar": "03",
        "march": "03",
        "apr": "04",
        "april": "04",
        "may": "05",
        "jun": "06",
        "june": "06",
        "jul": "07",
        "july": "07",
        "aug": "08",
        "august": "08",
        "sep": "09",
        "sept": "09",
        "september": "09",
        "oct": "10",
        "october": "10",
        "nov": "11",
        "november": "11",
        "dec": "12",
        "december": "12"
    }

    return months.get(
        value.lower(),
        None
    )


# ============================================================
# GET 0.8 FFP GEOMETRY FOR MONTH
# ============================================================

def get_monthly_ffp_geometry(
    gdf,
    month,
    month_field=MONTH_FIELD,
    contour_field=CONTOUR_FIELD,
    contour_value=CONTOUR_VALUE
):

    # --------------------------------------------------------
    # Convert month column to MM
    # --------------------------------------------------------

    month_values = (
        gdf[month_field]
        .apply(normalize_month)
    )

    # --------------------------------------------------------
    # Select month
    # --------------------------------------------------------

    monthly = gdf[
        month_values == month
    ]

    # --------------------------------------------------------
    # Select 0.8 contour
    # --------------------------------------------------------

    contour_80 = monthly[
        np.isclose(
            monthly[contour_field].astype(float),
            contour_value
        )
    ]

    if contour_80.empty:

        print(
            f"[WARNING] No {contour_value} "
            f"contour found for month {month}"
        )

        return None

    # --------------------------------------------------------
    # Merge polygon(s)
    # --------------------------------------------------------

    geometry = contour_80.geometry.union_all()

    # --------------------------------------------------------
    # Convert to EE geometry
    # --------------------------------------------------------

    geojson = gpd.GeoSeries(
        [geometry],
        crs="EPSG:4326"
    ).__geo_interface__["features"][0]["geometry"]

    return ee.Geometry(geojson)


# ============================================================
# LANDSAT MASK
# ============================================================

def mask_landsat(image):

    qa = image.select("QA_PIXEL")

    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )

    return image.updateMask(mask)


# ============================================================
# SENTINEL MASK
# ============================================================

def mask_sentinel(image):

    scl = image.select("SCL")

    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    return image.updateMask(mask)


# ============================================================
# LANDSAT NDVI
# ============================================================

def landsat_ndvi(start, end, geom):

    def add_ndvi(image):

        red = (
            image.select("SR_B4")
            .multiply(0.0000275)
            .add(-0.2)
        )

        nir = (
            image.select("SR_B5")
            .multiply(0.0000275)
            .add(-0.2)
        )

        ndvi = (
            nir.subtract(red)
            .divide(nir.add(red))
            .rename("NDVI")
        )

        return image.addBands(ndvi)

    collection = (
        ee.ImageCollection(
            "LANDSAT/LC08/C02/T1_L2"
        )
        .merge(
            ee.ImageCollection(
                "LANDSAT/LC09/C02/T1_L2"
            )
        )
        .filterDate(start, end)
        .filterBounds(geom)
        .map(mask_landsat)
        .map(add_ndvi)
    )

    return collection.select("NDVI").median()


# ============================================================
# SENTINEL NDVI
# ============================================================

def sentinel_ndvi(start, end, geom):

    def add_ndvi(image):

        red = image.select("B4").multiply(0.0001)
        nir = image.select("B8").multiply(0.0001)

        ndvi = (
            nir.subtract(red)
            .divide(nir.add(red))
            .rename("NDVI")
        )

        return image.addBands(ndvi)

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                70
            )
        )
        .map(mask_sentinel)
        .map(add_ndvi)
    )

    return collection.select("NDVI").median()


# ============================================================
# LANDSAT LAI
# ============================================================

def landsat_lai(start, end, geom):

    def add_lai(image):

        red = (
            image.select("SR_B4")
            .multiply(0.0000275)
            .add(-0.2)
        )

        nir = (
            image.select("SR_B5")
            .multiply(0.0000275)
            .add(-0.2)
        )

        blue = (
            image.select("SR_B2")
            .multiply(0.0000275)
            .add(-0.2)
        )

        evi = (
            nir.subtract(red)
            .multiply(2.5)
            .divide(
                nir
                .add(red.multiply(6))
                .subtract(blue.multiply(7.5))
                .add(1)
            )
        )

        lai = (
            evi.multiply(3.618)
            .subtract(0.118)
            .max(0)
            .rename("LAI")
        )

        return image.addBands(lai)

    collection = (
        ee.ImageCollection(
            "LANDSAT/LC08/C02/T1_L2"
        )
        .merge(
            ee.ImageCollection(
                "LANDSAT/LC09/C02/T1_L2"
            )
        )
        .filterDate(start, end)
        .filterBounds(geom)
        .map(mask_landsat)
        .map(add_lai)
    )

    return collection.select("LAI").median()


# ============================================================
# SENTINEL LAI
# ============================================================

def sentinel_lai(start, end, geom):

    def add_lai(image):

        red = image.select("B4").multiply(0.0001)
        nir = image.select("B8").multiply(0.0001)
        blue = image.select("B2").multiply(0.0001)

        evi = (
            nir.subtract(red)
            .multiply(2.5)
            .divide(
                nir
                .add(red.multiply(6))
                .subtract(blue.multiply(7.5))
                .add(1)
            )
        )

        lai = (
            evi.multiply(3.618)
            .subtract(0.118)
            .max(0)
            .rename("LAI")
        )

        return image.addBands(lai)

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                70
            )
        )
        .map(mask_sentinel)
        .map(add_lai)
    )

    return collection.select("LAI").median()


# ============================================================
# SENSOR SELECTORS
# ============================================================

def get_ndvi(
    sensor,
    start,
    end,
    geom
):

    sensor = sensor.lower()

    if sensor == "landsat":

        return landsat_ndvi(
            start,
            end,
            geom
        )

    elif sensor == "sentinel":

        return sentinel_ndvi(
            start,
            end,
            geom
        )

    else:

        raise ValueError(
            "NDVI_SENSOR must be "
            "'Landsat' or 'Sentinel'"
        )


def get_lai(
    sensor,
    start,
    end,
    geom
):

    sensor = sensor.lower()

    if sensor == "landsat":

        return landsat_lai(
            start,
            end,
            geom
        )

    elif sensor == "sentinel":

        return sentinel_lai(
            start,
            end,
            geom
        )

    else:

        raise ValueError(
            "LAI_SENSOR must be "
            "'Landsat' or 'Sentinel'"
        )


# ============================================================
# EXPORT MAP
# ============================================================

def export_map(
    image,
    geom,
    filename,
    scale
):

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    geemap.ee_export_image(
        image,
        filename=filename,
        scale=scale,
        region=geom,
        file_per_band=False
    )

    print(f"[SAVED] {filename}")


# ============================================================
# MAIN PROCESS
# ============================================================

def monthly_ndvi_lai(
    config,
    start_date=START_DATE,
    end_date=END_DATE,
    ndvi_sensor=NDVI_SENSOR,
    lai_sensor=LAI_SENSOR
):

    site = config["database"]["site_name"][0]
    base_path = config["database"]["base_path"]

    # --------------------------------------------------------
    # Read FFP shapefile
    # --------------------------------------------------------

    gdf = load_ffp_shapefile(config)

    print("\n[INFO] Shapefile fields:")
    print(gdf.columns.tolist())

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # --------------------------------------------------------
    # Generate monthly periods
    # --------------------------------------------------------

    months = pd.period_range(
        start=start,
        end=end,
        freq="M"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = os.path.join(
        base_path,
        "FFP_output",
        site,
        "NDVI_LAI"
    )

    ndvi_dir = os.path.join(
        output_dir,
        "NDVI_maps"
    )

    lai_dir = os.path.join(
        output_dir,
        "LAI_maps"
    )

    os.makedirs(
        ndvi_dir,
        exist_ok=True
    )

    os.makedirs(
        lai_dir,
        exist_ok=True
    )

    results = []

    # ========================================================
    # MONTH LOOP
    # ========================================================

    for period in months:

        month = f"{period.month:02d}"

        month_start = max(
            period.start_time,
            start
        )

        month_end = min(
            period.end_time,
            end
        )

        # GEE end date is exclusive
        gee_start = (
            month_start.strftime("%Y-%m-%d")
        )

        gee_end = (
            (month_end + pd.Timedelta(days=1))
            .strftime("%Y-%m-%d")
        )

        label = period.strftime("%Y-%m")

        print("\n" + "=" * 70)
        print(f"Processing {label}")
        print(f"NDVI sensor: {ndvi_sensor}")
        print(f"LAI sensor : {lai_sensor}")

        # ----------------------------------------------------
        # Get monthly 0.8 FFP
        # ----------------------------------------------------

        geom = get_monthly_ffp_geometry(
            gdf,
            month=month
        )

        if geom is None:
            continue

        # ----------------------------------------------------
        # NDVI
        # ----------------------------------------------------

        ndvi_img = get_ndvi(
            ndvi_sensor,
            gee_start,
            gee_end,
            geom
        )

        ndvi_value = (
            ndvi_img
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=30,
                maxPixels=1e9
            )
            .get("NDVI")
            .getInfo()
        )

        # ----------------------------------------------------
        # LAI
        # ----------------------------------------------------

        lai_img = get_lai(
            lai_sensor,
            gee_start,
            gee_end,
            geom
        )

        lai_value = (
            lai_img
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=10,
                maxPixels=1e9
            )
            .get("LAI")
            .getInfo()
        )

        # ----------------------------------------------------
        # Export NDVI
        # ----------------------------------------------------

        ndvi_file = os.path.join(
            ndvi_dir,
            f"NDVI_FFP80_{site}_{label}.tif"
        )

        export_map(
            ndvi_img,
            geom,
            ndvi_file,
            scale=30
        )

        # ----------------------------------------------------
        # Export LAI
        # ----------------------------------------------------

        lai_file = os.path.join(
            lai_dir,
            f"LAI_FFP80_{site}_{label}.tif"
        )

        export_map(
            lai_img,
            geom,
            lai_file,
            scale=10
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({
            "month": label,
            "NDVI_sensor": ndvi_sensor,
            "NDVI": ndvi_value,
            "LAI_sensor": lai_sensor,
            "LAI": lai_value
        })

        print(
            f"NDVI = {ndvi_value}"
        )

        print(
            f"LAI = {lai_value}"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    df = pd.DataFrame(results)

    csv_file = os.path.join(
        output_dir,
        f"NDVI_LAI_timeseries_{site}_"
        f"{start_date}_{end_date}.csv"
    )

    df.to_csv(
        csv_file,
        index=False
    )

    print("\n" + "=" * 70)
    print(f"[SAVED] {csv_file}")

    return df