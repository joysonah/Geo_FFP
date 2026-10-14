import os
import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import mapping


# ============================================================
# USER OPTIONS
# ============================================================

NDVI_SENSOR = "Landsat"       # "Landsat" or "Sentinel"
LAI_SENSOR  = "Sentinel"      # "Landsat" or "Sentinel"

CONTOUR_VALUE = 0.8

MONTH_FIELD = "month"
CONTOUR_FIELD = "rs"

# Minimum number of valid images required in a month
MIN_IMAGES = 1

# Maximum cloud percentage allowed for a scene
MAX_CLOUD = 70


# ============================================================
# GEE INITIALIZATION
# ============================================================

# def init_gee(config):

#     try:
#         ee.Initialize(project=config["gee"]["project_id"])

#     except Exception:
#         ee.Authenticate()
#         ee.Initialize(project=config["gee"]["project_id"])

def init_gee(config):

    project_id = config["gee"]["project_id"]

    try:
        ee.Initialize(project=project_id)
        print("GEE initialized successfully.")

    except Exception:

        print("GEE authentication required.")

        ee.Authenticate()

        ee.Initialize(
            project=project_id
        )

        print("GEE authentication and initialization successful.")


# ============================================================
# DATE HANDLING
# ============================================================

def get_date_range(config, start_date=None, end_date=None):

    years = config["database"]["years"]

    # Convert years to list
    if isinstance(years, (int, np.integer)):
        years = [int(years)]
    else:
        years = [int(y) for y in years]

    # Both dates missing
    if start_date is None and end_date is None:
        start_date = f"{min(years)}-01-01"
        end_date = f"{max(years)}-12-31"

    # Only start date provided
    elif start_date is not None and end_date is None:
        start_dt = pd.to_datetime(start_date)
        end_date = f"{start_dt.year}-12-31"

    # Only end date provided
    elif start_date is None and end_date is not None:
        end_dt = pd.to_datetime(end_date)
        start_date = f"{end_dt.year}-01-01"

    return pd.to_datetime(start_date), pd.to_datetime(end_date)


# ============================================================
# MONTH NORMALIZATION
# ============================================================

def normalize_month(value):

    if pd.isna(value):
        return None

    # Numeric month
    try:
        num = float(value)

        if 1 <= num <= 12:
            return f"{int(num):02d}"

    except Exception:
        pass

    value = str(value).strip()

    # YYYY-MM
    if len(value) >= 7:
        try:
            dt = pd.to_datetime(value)
            return f"{dt.month:02d}"
        except Exception:
            pass

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
        "december": "12",
    }

    return months.get(value.lower())


# ============================================================
# LOAD MONTHLY FFP 0.8 CONTOURS
# ============================================================

def load_ffp_shapefile(config):

    base_path = config["database"]["base_path"]
    site_name = config["database"]["site_name"][0]

    shp_dir = os.path.join(
        base_path,
        "FFP_output",
        site_name,
        "shapefile"
    )

    shp_files = [
        os.path.join(shp_dir, f)
        for f in os.listdir(shp_dir)
        if f.lower().endswith(".shp")
    ]

    if not shp_files:
        raise FileNotFoundError(
            f"No shapefile found in:\n{shp_dir}"
        )

    gdfs = []

    for shp in shp_files:
        gdf = gpd.read_file(shp)
        gdfs.append(gdf)

    gdf = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=gdfs[0].crs
    )

    # Convert to geographic coordinates
    gdf = gdf.to_crs("EPSG:4326")

    return gdf


# ============================================================
# GET MONTHLY 0.8 FFP GEOMETRY
# ============================================================

def get_monthly_ffp_geometry(
    gdf,
    month,
    contour_value=0.8
):

    month_norm = gdf[MONTH_FIELD].apply(normalize_month)

    subset = gdf[
        (month_norm == month) &
        (
            np.isclose(
                pd.to_numeric(
                    gdf[CONTOUR_FIELD],
                    errors="coerce"
                ),
                contour_value
            )
        )
    ].copy()

    if subset.empty:
        return None

    # Combine multiple polygons if necessary
    geometry = subset.geometry.union_all()

    return ee.Geometry(mapping(geometry))


# ============================================================
# LANDSAT PREPROCESSING
# ============================================================

def mask_landsat(image):

    qa = image.select("QA_PIXEL")

    # QA_PIXEL bits:
    # 1 = dilated cloud
    # 2 = cirrus
    # 3 = cloud
    # 4 = cloud shadow
    # 5 = snow

    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )

    # Remove saturated pixels
    saturation = image.select("QA_RADSAT").eq(0)

    # Scale surface reflectance
    optical = (
        image
        .select("SR_B.*")
        .multiply(0.0000275)
        .add(-0.2)
    )

    return (
        image
        .addBands(
            optical,
            overwrite=True
        )
        .updateMask(mask)
        .updateMask(saturation)
    )


# ============================================================
# SENTINEL-2 PREPROCESSING
# ============================================================

def mask_sentinel(image):

    # Scene Classification Layer
    scl = image.select("SCL")

    # Remove:
    # 3  = cloud shadow
    # 8  = medium probability cloud
    # 9  = high probability cloud
    # 10 = cirrus
    # 11 = snow/ice

    scl_mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    # QA60 cloud/cirrus mask
    qa60 = image.select("QA60")

    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11

    qa_mask = (
        qa60.bitwiseAnd(cloud_bit).eq(0)
        .And(
            qa60.bitwiseAnd(cirrus_bit).eq(0)
        )
    )

    # Scale Sentinel reflectance
    optical = (
        image
        .select("B.*")
        .multiply(0.0001)
    )

    return (
        image
        .addBands(
            optical,
            overwrite=True
        )
        .updateMask(scl_mask)
        .updateMask(qa_mask)
    )


# ============================================================
# LANDSAT COLLECTION
# ============================================================

def get_landsat_collection(
    geometry,
    start_date,
    end_date
):

    l8 = (
        ee.ImageCollection(
            "LANDSAT/LC08/C02/T1_L2"
        )
        .filterBounds(geometry)
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.lte(
                "CLOUD_COVER",
                MAX_CLOUD
            )
        )
        .map(mask_landsat)
    )

    l9 = (
        ee.ImageCollection(
            "LANDSAT/LC09/C02/T1_L2"
        )
        .filterBounds(geometry)
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.lte(
                "CLOUD_COVER",
                MAX_CLOUD
            )
        )
        .map(mask_landsat)
    )

    return l8.merge(l9)


# ============================================================
# SENTINEL COLLECTION
# ============================================================

def get_sentinel_collection(
    geometry,
    start_date,
    end_date
):

    return (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(geometry)
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                MAX_CLOUD
            )
        )
        .map(mask_sentinel)
    )


# ============================================================
# NDVI
# ============================================================

def add_ndvi(image, sensor):

    if sensor.lower() == "landsat":

        ndvi = image.normalizedDifference(
            ["SR_B5", "SR_B4"]
        ).rename("NDVI")

    elif sensor.lower() == "sentinel":

        ndvi = image.normalizedDifference(
            ["B8", "B4"]
        ).rename("NDVI")

    else:
        raise ValueError(
            "Sensor must be 'Landsat' or 'Sentinel'"
        )

    # Remove physically invalid values
    ndvi = ndvi.updateMask(
        ndvi.gte(-1).And(ndvi.lte(1))
    )

    return image.addBands(ndvi)


# ============================================================
# EVI
# ============================================================

def add_evi(image, sensor):

    if sensor.lower() == "landsat":

        evi = image.expression(
            "2.5 * ((NIR - RED) / "
            "(NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": image.select("SR_B5"),
                "RED": image.select("SR_B4"),
                "BLUE": image.select("SR_B2")
            }
        ).rename("EVI")

    elif sensor.lower() == "sentinel":

        evi = image.expression(
            "2.5 * ((NIR - RED) / "
            "(NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": image.select("B8"),
                "RED": image.select("B4"),
                "BLUE": image.select("B2")
            }
        ).rename("EVI")

    else:
        raise ValueError(
            "Sensor must be 'Landsat' or 'Sentinel'"
        )

    evi = evi.updateMask(
        evi.gte(-1).And(evi.lte(1))
    )

    return image.addBands(evi)


# ============================================================
# LAI FROM EVI
# ============================================================

def add_lai(image):

    # Empirical EVI -> LAI relationship
    #
    # LAI = 3.618 * EVI - 0.118

    lai = (
        image
        .select("EVI")
        .multiply(3.618)
        .subtract(0.118)
        .max(0)
        .rename("LAI")
    )

    return image.addBands(lai)


# ============================================================
# MONTHLY COMPOSITE
# ============================================================

def make_monthly_composite(
    geometry,
    start_date,
    end_date,
    sensor
):

    if sensor.lower() == "landsat":

        collection = get_landsat_collection(
            geometry,
            start_date,
            end_date
        )

    elif sensor.lower() == "sentinel":

        collection = get_sentinel_collection(
            geometry,
            start_date,
            end_date
        )

    else:
        raise ValueError(
            "Sensor must be 'Landsat' or 'Sentinel'"
        )

    count = collection.size()

    # Median composite reduces residual cloud/outlier effects
    composite = collection.median()

    return collection, composite, count


# ============================================================
# MONTHLY NDVI + LAI
# ============================================================

def monthly_ndvi_lai(
    config,
    start_date=None,
    end_date=None,
    ndvi_sensor="Landsat",
    lai_sensor="Sentinel"
):

    site_name = config["database"]["site_name"][0]
    base_path = config["database"]["base_path"]

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    start_date, end_date = get_date_range(
        config,
        start_date,
        end_date
    )

    print(
        f"Processing: "
        f"{start_date.date()} → {end_date.date()}"
    )

    # --------------------------------------------------------
    # FFP shapefile
    # --------------------------------------------------------

    gdf = load_ffp_shapefile(config)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_dir = os.path.join(
        base_path,
        "FFP_output",
        site_name,
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

    os.makedirs(ndvi_dir, exist_ok=True)
    os.makedirs(lai_dir, exist_ok=True)

    # --------------------------------------------------------
    # Monthly periods
    # --------------------------------------------------------

    periods = pd.period_range(
        start=start_date,
        end=end_date,
        freq="M"
    )

    results = []

    # ========================================================
    # LOOP THROUGH MONTHS
    # ========================================================

    for period in periods:

        month = f"{period.month:02d}"

        print(
            f"\nProcessing {period} "
            f"(month = {month})"
        )

        # ----------------------------------------------------
        # FFP 80% contour
        # ----------------------------------------------------

        geometry = get_monthly_ffp_geometry(
            gdf,
            month,
            CONTOUR_VALUE
        )

        if geometry is None:

            print(
                f"No FFP {CONTOUR_VALUE} contour "
                f"found for month {month}"
            )

            continue

        # ----------------------------------------------------
        # Correct date boundaries
        # ----------------------------------------------------

        month_start = max(
            period.start_time,
            start_date
        )

        month_end = min(
            period.end_time,
            end_date
        )

        # GEE end date is exclusive
        gee_start = month_start.strftime(
            "%Y-%m-%d"
        )

        gee_end = (
            month_end + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")

        # ====================================================
        # NDVI
        # ====================================================

        ndvi_collection, ndvi_composite, ndvi_count = (
            make_monthly_composite(
                geometry,
                gee_start,
                gee_end,
                ndvi_sensor
            )
        )

        n_ndvi = ndvi_count.getInfo()

        print(
            f"NDVI sensor: {ndvi_sensor}"
        )

        print(
            f"Valid scenes before pixel masking: "
            f"{n_ndvi}"
        )

        if n_ndvi < MIN_IMAGES:

            print(
                "Not enough NDVI images. Skipping."
            )

            continue

        ndvi_image = add_ndvi(
            ndvi_composite,
            ndvi_sensor
        ).select("NDVI")

        # Native spatial resolution
        ndvi_scale = (
            30
            if ndvi_sensor.lower() == "landsat"
            else 10
        )

        # ----------------------------------------------------
        # Regional NDVI mean
        # ----------------------------------------------------

        ndvi_mean = ndvi_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=ndvi_scale,
            maxPixels=1e9,
            bestEffort=True
        ).get("NDVI").getInfo()

        # ----------------------------------------------------
        # NDVI GeoTIFF
        # ----------------------------------------------------

        ndvi_filename = (
            f"NDVI_FFP80_{site_name}_"
            f"{period.strftime('%Y-%m')}.tif"
        )

        ndvi_path = os.path.join(
            ndvi_dir,
            ndvi_filename
        )

        geemap.ee_export_image(
            ndvi_image,
            filename=ndvi_path,
            scale=ndvi_scale,
            region=geometry,
            file_per_band=False
        )

        # ====================================================
        # LAI
        # ====================================================

        lai_collection, lai_composite, lai_count = (
            make_monthly_composite(
                geometry,
                gee_start,
                gee_end,
                lai_sensor
            )
        )

        n_lai = lai_count.getInfo()

        print(
            f"LAI sensor: {lai_sensor}"
        )

        print(
            f"Valid scenes before pixel masking: "
            f"{n_lai}"
        )

        if n_lai < MIN_IMAGES:

            print(
                "Not enough LAI images. Skipping LAI."
            )

            lai_mean = np.nan

        else:

            lai_image = (
                add_lai(
                    add_evi(
                        lai_composite,
                        lai_sensor
                    )
                )
                .select("LAI")
            )

            lai_scale = (
                30
                if lai_sensor.lower() == "landsat"
                else 10
            )

            # ------------------------------------------------
            # Regional LAI mean
            # ------------------------------------------------

            lai_mean = lai_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=lai_scale,
                maxPixels=1e9,
                bestEffort=True
            ).get("LAI").getInfo()

            # ------------------------------------------------
            # LAI GeoTIFF
            # ------------------------------------------------

            lai_filename = (
                f"LAI_FFP80_{site_name}_"
                f"{period.strftime('%Y-%m')}.tif"
            )

            lai_path = os.path.join(
                lai_dir,
                lai_filename
            )

            geemap.ee_export_image(
                lai_image,
                filename=lai_path,
                scale=lai_scale,
                region=geometry,
                file_per_band=False
            )

        # ====================================================
        # SAVE RESULTS
        # ====================================================

        results.append({

            "month": period.strftime("%Y-%m"),

            "NDVI_sensor": ndvi_sensor,
            "NDVI": ndvi_mean,

            "LAI_sensor": lai_sensor,
            "LAI": lai_mean,

            "NDVI_n_scenes": n_ndvi,
            "LAI_n_scenes": n_lai
        })

        print(
            f"NDVI = {ndvi_mean}"
        )

        print(
            f"LAI = {lai_mean}"
        )

    # ========================================================
    # CSV
    # ========================================================

    df = pd.DataFrame(results)

    csv_name = (
        f"NDVI_LAI_timeseries_"
        f"{site_name}_"
        f"{start_date.strftime('%Y%m%d')}_"
        f"{end_date.strftime('%Y%m%d')}.csv"
    )

    csv_path = os.path.join(
        output_dir,
        csv_name
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print("\n======================================")
    print("Processing complete")
    print("======================================")
    print(f"CSV: {csv_path}")
    print(f"NDVI maps: {ndvi_dir}")
    print(f"LAI maps: {lai_dir}")

    return df