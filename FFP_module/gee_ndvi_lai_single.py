import os
import glob
import re

import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import mapping


# ============================================================
# USER OPTIONS
# ============================================================

# START_DATE = "2021-01-01"
# END_DATE   = "2025-12-31"

# NDVI_SENSOR = config["gee"]["NDVI_SENSOR"],      # "Landsat" or "Sentinel"
# LAI_SENSOR  = config["gee"]["LAI_SENSOR"]     # "Landsat" or "Sentinel"

CONTOUR_VALUE = 0.8

MONTH_FIELD = "month"
CONTOUR_FIELD = "rs"

# Maximum cloud percentage allowed for a scene
MAX_CLOUD = 70




# ============================================================
# GEE INITIALIZATION
# ============================================================

def init_gee(config):

    project_id = config["gee"]["project_id"]

    print(f"Initializing Google Earth Engine...")
    print(f"GEE project: {project_id}")

    try:

        # Try existing credentials
        ee.Initialize(
            project=project_id
        )

        # Actually test the connection
        test = ee.Number(1).getInfo()

        if test == 1:
            print("Google Earth Engine initialized successfully.")

    except Exception as e:

        print("Existing GEE credentials could not be used.")
        print(f"Initialization error: {e}")
        print("Starting GEE authentication...")

        try:

            ee.Authenticate()

            ee.Initialize(
                project=project_id
            )

            # Test GEE
            test = ee.Number(1).getInfo()

            if test == 1:
                print(
                    "Google Earth Engine "
                    "authentication successful."
                )

        except Exception as auth_error:

            raise RuntimeError(
                "\nGoogle Earth Engine initialization failed.\n"
                f"Project: {project_id}\n\n"
                f"Authentication error:\n{auth_error}"
            )


# ============================================================
# LOAD FFP SHAPEFILE
# ============================================================

def load_ffp_shapefile(config):

    base_path = config["database"]["base_path"]
    site_name = config["database"]["site_name"][0]
    years = config["database"]["years"]

    # If only one year is provided
    if isinstance(years, (int, np.integer)):
        year = int(years)

    # If years is a list, use the first year
    else:
        year = int(years[0])

    # --------------------------------------------------------
    # Monthly FFP contour shapefile
    # --------------------------------------------------------

    shp_path = os.path.join(
        base_path,
        "FFP_output",
        site_name,
        str(year),
        "shapefile",
        "Monthly",
        f"FFP_monthly_{site_name}_{year}_contours.shp"
    )

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(shp_path):

        raise FileNotFoundError(
            f"\nMonthly FFP contour shapefile not found:\n"
            f"{shp_path}"
        )

    print(
        f"\nLoading monthly FFP contours:\n"
        f"{shp_path}"
    )

    # --------------------------------------------------------
    # Read shapefile
    # --------------------------------------------------------

    gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise ValueError(
            f"FFP shapefile is empty:\n{shp_path}"
        )

    # --------------------------------------------------------
    # Convert to WGS84 for Google Earth Engine
    # --------------------------------------------------------

    gdf = gdf.to_crs("EPSG:4326")

    print(
        f"Loaded {len(gdf)} FFP contour features."
    )

    print(
        f"Columns: {list(gdf.columns)}"
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

def get_monthly_ffp_geometry(gdf, month, contour_value=0.80):
    """
    Return the monthly FFP contour as an Earth Engine Geometry.

    Parameters
    ----------
    gdf : GeoDataFrame
        Monthly FFP contour shapefile.
    month : int
        Calendar month (1-12).
    contour_value : float
        FFP isopleth, normally 0.80.

    Returns
    -------
    ee.Geometry or None
    """

    monthly = gdf.copy()

    # --------------------------------------------------
    # Normalize month field
    # --------------------------------------------------
    monthly["month"] = (
        monthly["month"]
        .astype(str)
        .str.strip()
    )

    month_names = {
        "1": 1, "01": 1, "jan": 1, "january": 1,
        "2": 2, "02": 2, "feb": 2, "february": 2,
        "3": 3, "03": 3, "mar": 3, "march": 3,
        "4": 4, "04": 4, "apr": 4, "april": 4,
        "5": 5, "05": 5, "may": 5,
        "6": 6, "06": 6, "jun": 6, "june": 6,
        "7": 7, "07": 7, "jul": 7, "july": 7,
        "8": 8, "08": 8, "aug": 8, "august": 8,
        "9": 9, "09": 9, "sep": 9, "september": 9,
        "10": 10, "oct": 10, "october": 10,
        "11": 11, "nov": 11, "november": 11,
        "12": 12, "dec": 12, "december": 12,
    }

    monthly["_month_num"] = (
        monthly["month"]
        .str.lower()
        .map(month_names)
    )

    # --------------------------------------------------
    # Select month
    # --------------------------------------------------
    
    monthly = monthly[
        monthly["_month_num"] == int(month)
    ].copy()

    # --------------------------------------------------
    # Select 0.8 isopleth
    # --------------------------------------------------
    monthly["isopleth"] = pd.to_numeric(
        monthly["isopleth"],
        errors="coerce"
    )

    monthly = monthly[
        np.isclose(
            monthly["isopleth"],
            contour_value,
            atol=0.001
        )
    ].copy()

    # --------------------------------------------------
    # Check
    # --------------------------------------------------
    if monthly.empty:
        print(
            f"[WARNING] No {contour_value} contour "
            f"found for month {int(month):02d}"
        )
        return None

    print(
        f"[INFO] Selected {contour_value} contour "
        f"for month {int(month):02d}"
    )

    # --------------------------------------------------
    # Combine geometry
    # --------------------------------------------------
    geom = monthly.geometry.union_all()

    if geom is None or geom.is_empty:
        print(
            f"[WARNING] Empty FFP geometry "
            f"for month {int(month):02d}"
        )
        return None

    # --------------------------------------------------
    # Shapely → GeoJSON → Earth Engine Geometry
    # --------------------------------------------------
    geom_geojson = mapping(geom)

    ee_geom = ee.Geometry(geom_geojson)

    return ee_geom


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



    return (
        image
        .updateMask(mask)
        .updateMask(saturation)
    )


# ============================================================
# SENTINEL-2 PREPROCESSING
# ============================================================
# ============================================================
# SENTINEL MASK
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

    return (
        image
        .updateMask(scl_mask)
        .updateMask(qa_mask)
    )


# ============================================================
# LANDSAT NDVI
# ============================================================

# def landsat_ndvi(start_date, end_date, geom):
    """
    Generate monthly Landsat NDVI composite.
    """

    # --------------------------------------------------
    # Landsat 8
    # --------------------------------------------------
    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .filter(
            ee.Filter.lte(
                "CLOUD_COVER",
                MAX_CLOUD
            )
        )
    )

    # --------------------------------------------------
    # Landsat 9
    # --------------------------------------------------
    l9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .filter(
            ee.Filter.lte(
                "CLOUD_COVER",
                MAX_CLOUD
            )
        )
    )

    # --------------------------------------------------
    # Combine Landsat 8 + 9
    # --------------------------------------------------
    collection = l8.merge(l9)

    count = collection.size().getInfo()

    print(
        f"[INFO] Landsat scenes available: {count}"
    )

    if count == 0:
        return None

    # --------------------------------------------------
    # Cloud / shadow / snow masking
    # --------------------------------------------------
    collection = collection.map(mask_landsat)

    # --------------------------------------------------
    # Monthly median composite
    # --------------------------------------------------
    image = collection.median()

    # --------------------------------------------------
    # NDVI
    # --------------------------------------------------
    ndvi = image.normalizedDifference(
        ["SR_B5", "SR_B4"]
    ).rename("NDVI")

    # Keep physically valid NDVI
    ndvi = ndvi.updateMask(
        ndvi.gte(-1).And(ndvi.lte(1))
    )

    return ndvi
def get_landsat_ndvi_collection(start_date, end_date, geom):

    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .filter(ee.Filter.lte("CLOUD_COVER", MAX_CLOUD))
    )

    l9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .filter(ee.Filter.lte("CLOUD_COVER", MAX_CLOUD))
    )

    collection = l8.merge(l9)

    print(
        f"[INFO] Landsat scenes available: "
        f"{collection.size().getInfo()}"
    )

    # Cloud masking
    collection = collection.map(mask_landsat)

    # Calculate NDVI for EACH image
    def add_ndvi(image):

        ndvi = image.normalizedDifference(
            ["SR_B5", "SR_B4"]
        ).rename("NDVI")

        return image.addBands(ndvi)

    collection = collection.map(add_ndvi)

    return collection

# ============================================================
# SENTINEL NDVI
# ============================================================

# def sentinel_ndvi(start, end, geom):

    def add_ndvi(image):

        red = image.select("B4").multiply(0.0001)
        nir = image.select("B8").multiply(0.0001)

        ndvi = (
            nir.subtract(red)
            .divide(nir.add(red))
            .rename("NDVI")
        )
        # Remove physically invalid values
        ndvi = ndvi.updateMask(
            ndvi.gte(-1).And(ndvi.lte(1))
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
                MAX_CLOUD
            )
        )
        .map(mask_sentinel)
        .map(add_ndvi)
    )

    count_S2 = collection.size().getInfo()

    print(
        f"[INFO] Sentinel-2 scenes available: {count_S2}"
    )

    if count_S2 == 0:
        return None

    return collection.select("NDVI").median()
def get_sentinel_lai_collection(start_date, end_date, geom):

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                MAX_CLOUD
            )
        )
        .map(mask_sentinel)
    )

    print(
        f"[INFO] Sentinel scenes available: "
        f"{collection.size().getInfo()}"
    )

    # Calculate LAI for EACH image
    def add_lai(image):

        evi = image.expression(
            "2.5 * ((NIR - RED) / "
            "(NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": image.select("B8"),
                "RED": image.select("B4"),
                "BLUE": image.select("B2")
            }
        ).rename("EVI")

        lai = (
            evi
            .multiply(3.618)
            .subtract(0.118)
            .max(0)
            .rename("LAI")
        )

        return image.addBands([evi, lai])

    return collection.map(add_lai)

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
    count = collection.size().getInfo()

    print(
        f"[INFO] Landsat scenes available: {count}"
    )

    if count == 0:
        return None

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
                MAX_CLOUD
            )
        )
        .map(mask_sentinel)
        .map(add_lai)
    )
    count = collection.size().getInfo()

    print(
        f"[INFO] Sentinel-2 scenes available: {count}"
    )

    if count == 0:
        return None
    
    return collection.select("LAI").median()


# ============================================================
# SENSOR SELECTORS
# ============================================================

def get_ndvi(sensor, start_date, end_date, geom):
    """
    Generate a monthly NDVI image using the selected sensor.
    """

    sensor = sensor.strip().lower()

    if sensor == "landsat":
        return landsat_ndvi(
            start_date,
            end_date,
            geom
        )

    elif sensor == "sentinel":
        return sentinel_ndvi(
            start_date,
            end_date,
            geom
        )

    else:
        raise ValueError(
            f"Unsupported NDVI sensor: {sensor}. "
            f"Use 'Landsat' or 'Sentinel'."
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
    start_date=None,
    end_date=None
):

    # ========================================================
    # GEE SETTINGS
    # ========================================================

    project_id = config["gee"]["project_id"]
    scale = config["gee"]["scale"]

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    base_path = config["database"]["base_path"]
    site_name = config["database"]["site_name"][0]

    years = config["database"]["years"]

    if isinstance(years, (int, np.integer)):
        year = int(years)
    else:
        year = int(years[0])

    year_str = str(year)

    ndvi_sensor = config["gee"]["NDVI_SENSOR"]
    lai_sensor = config["gee"]["LAI_SENSOR"]

    if start_date is None:
        start_date = config["gee"]["START_DATE"]

    if end_date is None:
        end_date = config["gee"]["END_DATE"]

    print("\n" + "=" * 70)
    print("NDVI / LAI EXTRACTION")
    print("=" * 70)

    print(f"Site        : {site_name}")
    print(f"Year        : {year}")
    print(f"NDVI sensor : {ndvi_sensor}")
    print(f"LAI sensor  : {lai_sensor}")
    print(f"Start date  : {start_date}")
    print(f"End date    : {end_date}")

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

    # ========================================================
    # OUTPUT
    # ========================================================

    # One common folder for everything
    output_dir = os.path.join(
        base_path,
        "FFP_output",
        site_name,
        year_str,
        "NDVI_LAI"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # ========================================================
    # MONTH LOOP
    # ========================================================

    for period in months:

        # IMPORTANT:
        # Keep month as integer for get_monthly_ffp_geometry()
        month = int(period.month)

        label = period.strftime("%Y-%m")

        # ----------------------------------------------------
        # Monthly date range
        # ----------------------------------------------------

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

        print("\n" + "=" * 70)
        print(f"Processing {label}")
        print(f"GEE period: {gee_start} → {gee_end}")
        print("=" * 70)

        # ====================================================
        # GET MONTHLY 0.8 FFP
        # ====================================================

        geom = get_monthly_ffp_geometry(
            gdf=gdf,
            month=month,
            contour_value=CONTOUR_VALUE
        )

        if geom is None:

            print(
                f"[WARNING] No FFP geometry for {label}"
            )

            continue

        # ====================================================
        # NDVI
        # ====================================================

        if ndvi_sensor.strip().lower() == "landsat":

            print(
                f"\n[INFO] Extracting NDVI using Landsat..."
            )

            # ------------------------------------------------
            # Get ALL Landsat images for this month
            # ------------------------------------------------

            landsat_collection = (
                get_landsat_ndvi_collection(
                    gee_start,
                    gee_end,
                    geom
                )
            )

            n_landsat = (
                landsat_collection
                .size()
                .getInfo()
            )

            print(
                f"[INFO] Landsat scenes available: "
                f"{n_landsat}"
            )

            # ------------------------------------------------
            # Loop through every Landsat image
            # ------------------------------------------------

            if n_landsat > 0:

                landsat_list = (
                    landsat_collection
                    .toList(n_landsat)
                )

                for i in range(n_landsat):

                    image = ee.Image(
                        landsat_list.get(i)
                    )

                    # ----------------------------------------
                    # Date
                    # ----------------------------------------

                    date = (
                        ee.Date(
                            image.get(
                                "system:time_start"
                            )
                        )
                        .format("YYYY-MM-dd")
                        .getInfo()
                    )

                    # ----------------------------------------
                    # Scene ID
                    # ----------------------------------------

                    scene_id = image.get(
                        "LANDSAT_PRODUCT_ID"
                    ).getInfo()

                    # ----------------------------------------
                    # Mean NDVI within FFP 0.8
                    # ----------------------------------------

                    result = (
                        image
                        .select("NDVI")
                        .reduceRegion(
                            reducer=ee.Reducer.median(),
                            geometry=geom,
                            scale=30,
                            maxPixels=1e9,
                            bestEffort=True
                        )
                        .getInfo()
                    )

                    # If all pixels are masked,
                    # result can be an empty dictionary
                    ndvi_value = result.get(
                        "NDVI"
                    )

                    # ----------------------------------------
                    # Store result
                    # ----------------------------------------

                    results.append({

                        "date": date,

                        "year": year,

                        "month": month,

                        "month_label": label,

                        "sensor": "Landsat",

                        "variable": "NDVI",

                        "scene_id": scene_id,

                        "value": ndvi_value
                    })

                    print(
                        f"    Landsat | "
                        f"{date} | "
                        f"NDVI = {ndvi_value}"
                    )

            else:

                print(
                    f"[WARNING] No Landsat scenes "
                    f"available for {label}"
                )

        else:

            print(
                f"[WARNING] NDVI sensor "
                f"'{ndvi_sensor}' is not configured "
                f"for all-image extraction."
            )

        # ====================================================
        # LAI
        # ====================================================

        if lai_sensor.strip().lower() == "sentinel":

            print(
                f"\n[INFO] Extracting LAI using Sentinel..."
            )

            # ------------------------------------------------
            # Get ALL Sentinel images for this month
            # ------------------------------------------------

            sentinel_collection = (
                get_sentinel_lai_collection(
                    gee_start,
                    gee_end,
                    geom
                )
            )

            n_sentinel = (
                sentinel_collection
                .size()
                .getInfo()
            )

            print(
                f"[INFO] Sentinel scenes available: "
                f"{n_sentinel}"
            )

            # ------------------------------------------------
            # Loop through every Sentinel image
            # ------------------------------------------------

            if n_sentinel > 0:

                sentinel_list = (
                    sentinel_collection
                    .toList(n_sentinel)
                )

                for i in range(n_sentinel):

                    image = ee.Image(
                        sentinel_list.get(i)
                    )

                    # ----------------------------------------
                    # Date
                    # ----------------------------------------

                    date = (
                        ee.Date(
                            image.get(
                                "system:time_start"
                            )
                        )
                        .format("YYYY-MM-dd")
                        .getInfo()
                    )

                    # ----------------------------------------
                    # Scene ID
                    # ----------------------------------------

                    scene_id = image.get(
                        "PRODUCT_ID"
                    ).getInfo()

                    # ----------------------------------------
                    # Mean LAI within FFP 0.8
                    # ----------------------------------------

                    result = (
                        image
                        .select("LAI")
                        .reduceRegion(
                            reducer=ee.Reducer.median(),
                            geometry=geom,
                            scale=10,
                            maxPixels=1e9,
                            bestEffort=True
                        )
                        .getInfo()
                    )

                    lai_value = result.get(
                        "LAI"
                    )

                    # ----------------------------------------
                    # Store result
                    # ----------------------------------------

                    results.append({

                        "date": date,

                        "year": year,

                        "month": month,

                        "month_label": label,

                        "sensor": "Sentinel",

                        "variable": "LAI",

                        "scene_id": scene_id,

                        "value": lai_value
                    })

                    print(
                        f"    Sentinel | "
                        f"{date} | "
                        f"LAI = {lai_value}"
                    )

            else:

                print(
                    f"[WARNING] No Sentinel scenes "
                    f"available for {label}"
                )

        else:

            print(
                f"[WARNING] LAI sensor "
                f"'{lai_sensor}' is not configured "
                f"for all-image extraction."
            )

    # ========================================================
    # SAVE COMBINED CSV
    # ========================================================

    df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    if not df.empty:

        df = df.sort_values(
            by=[
                "date",
                "variable",
                "sensor"
            ]
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # Single output filename
    # --------------------------------------------------------

    csv_file = os.path.join(
        output_dir,
        f"NDVI_LAI_all_images_{site_name}_{year_str}.csv"
    )

    df.to_csv(
        csv_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("NDVI / LAI EXTRACTION COMPLETED")
    print("=" * 70)

    print(
        f"[INFO] Total observations: {len(df)}"
    )

    if not df.empty:

        print("\nObservations by sensor/variable:")

        print(
            df.groupby(
                ["sensor", "variable"]
            ).size()
        )

    print(
        f"\n[SAVED] {csv_file}"
    )

    print(
        f"[SAVED] Output folder: {output_dir}"
    )

    return df

# def monthly_ndvi_lai(
#     config,
#     start_date=None,
#     end_date=None
# ):

    # ========================================================
    # GEE SETTINGS
    # ========================================================

    project_id = config["gee"]["project_id"]
    scale = config["gee"]["scale"]

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------
    base_path = config["database"]["base_path"]
    site_name = config["database"]["site_name"][0]
    year = config["database"]["years"]
    year_str = str(year[0])

    ndvi_sensor = config["gee"]["NDVI_SENSOR"]
    lai_sensor = config["gee"]["LAI_SENSOR"]


    if start_date is None:
        start_date = config["gee"]["START_DATE"]

    if end_date is None:
        end_date = config["gee"]["END_DATE"]

    print(f"NDVI sensor : {ndvi_sensor}")
    print(f"LAI sensor  : {lai_sensor}")
    print(f"Start date  : {start_date}")
    print(f"End date    : {end_date}")

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
        site_name,
        year_str,
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

        # ----------------------------------------------------
        # Get monthly 0.8 FFP
        # ----------------------------------------------------

        geom = get_monthly_ffp_geometry(
            gdf=gdf,
            month=month,
            contour_value=CONTOUR_VALUE
)

        if geom is None:
            continue

        # ----------------------------------------------------
        # NDVI
        # ----------------------------------------------------

        print(f"[INFO] Extracting NDVI using {ndvi_sensor}...")

        ndvi_img = get_ndvi(
            ndvi_sensor,
            gee_start,
            gee_end,
            geom
        )

        ndvi_value = None

        if ndvi_img is not None:

            ndvi_result = ndvi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=30,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()

            ndvi_value = ndvi_result.get("NDVI")

            if ndvi_value is None:
                print(
                    f"[WARNING] No valid NDVI pixels "
                    # f"inside FFP for {year}-{month:02d}"
                )
            else:
                print(
                    f"[INFO] NDVI = {ndvi_value:.4f}"
                )

        else:
            print(
                f"[WARNING] No {ndvi_sensor} NDVI image "
                # f"available for {year}-{month:02d}"
            )


        # ============================================================
        # LAI
        # ============================================================

        print(f"[INFO] Extracting LAI using {lai_sensor}...")

        lai_img = get_lai(
            lai_sensor,
            gee_start,
            gee_end,
            geom
        )

        lai_value = None

        if lai_img is not None:

            lai_result = lai_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=10,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()

            lai_value = lai_result.get("LAI")

            if lai_value is None:
                print(
                    f"[WARNING] No valid LAI pixels "
                    # f"inside FFP for {year}-{month:02d}"
                )
            else:
                print(
                    f"[INFO] LAI = {lai_value:.4f}"
                )

        else:
            print(
                f"[WARNING] No {lai_sensor} LAI image "
                # f"available for {year}-{month:02d}"
            )

        # ----------------------------------------------------
        # Export NDVI
        # ----------------------------------------------------

        ndvi_file = os.path.join(
            ndvi_dir,
            f"NDVI_FFP80_{config['database']['site_name'][0]}_{label}.tif"
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
            f"LAI_FFP80_{config['database']['site_name'][0]}_{label}.tif"
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
        f"NDVI_LAI_timeseries_{config['database']['site_name'][0]}_"
        f"{start_date}_{end_date}.csv"
    )

    df.to_csv(
        csv_file,
        index=False
    )

    print("\n" + "=" * 70)
    print(f"[SAVED] {csv_file}")

    return df