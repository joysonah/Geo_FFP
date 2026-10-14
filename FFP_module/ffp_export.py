# ============================================================
# ffp_export.py
#
# FFP export module
#
# SINGLE:
#   NetCDF
#   GeoTIFF
#   Contour Shapefile
#   Tower Shapefile
#
# MONTHLY:
#   One 12-month NetCDF
#   One 12-band GeoTIFF
#   One contour Shapefile containing all months
#   One tower Shapefile
# ============================================================

import os

import numpy as np
import xarray as xr
import pyproj
import geopandas as gpd

from shapely.geometry import LineString, Point

import rasterio
from rasterio.transform import from_origin


# ============================================================
# 1. UTM CRS
# ============================================================

def get_utm_crs(lat, lon):

    zone = int((lon + 180) / 6) + 1

    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return pyproj.CRS.from_epsg(epsg)


# ============================================================
# 2. Tower coordinates
# ============================================================

def get_tower_coordinates(lat, lon, crs):

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    x0, y0 = transformer.transform(
        lon,
        lat
    )

    return x0, y0


# ============================================================
# 3. Prepare FFP grid
# ============================================================

def prepare_ffp_grid(FFP, x0, y0):

    X_local = np.asarray(
        FFP["x_2d"],
        dtype=float
    )

    Y_local = np.asarray(
        FFP["y_2d"],
        dtype=float
    )

    fclim = np.asarray(
        FFP["fclim_2d"],
        dtype=np.float32
    )

    X_abs = X_local + x0
    Y_abs = Y_local + y0

    x = X_abs[0, :]
    y = Y_abs[:, 0]

    return x, y, fclim


# ============================================================
# 4. Prepare contours
# ============================================================

def prepare_contours(
    FFP,
    x0,
    y0
):

    rs = np.asarray(
        FFP["rs"],
        dtype=float
    )

    geometries = []
    isopleths = []

    for i, r in enumerate(rs):

        xr_i = FFP["xr"][i]
        yr_i = FFP["yr"][i]

        if xr_i is None or yr_i is None:
            continue

        xr_i = np.asarray(
            xr_i,
            dtype=float
        )

        yr_i = np.asarray(
            yr_i,
            dtype=float
        )

        valid = (
            np.isfinite(xr_i)
            &
            np.isfinite(yr_i)
        )

        xr_i = xr_i[valid]
        yr_i = yr_i[valid]

        n = min(
            len(xr_i),
            len(yr_i)
        )

        if n < 2:
            continue

        coords = list(
            zip(
                xr_i[:n] + x0,
                yr_i[:n] + y0
            )
        )

        # Close contour
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        geometries.append(
            LineString(coords)
        )

        isopleths.append(
            float(r)
        )

    return geometries, isopleths


# ============================================================
# 5. Export SINGLE NetCDF
# ============================================================

def export_single_netcdf(
    FFP,
    site_name,
    year,
    filename,
    lat0,
    lon0,
    crs
):

    x0, y0 = get_tower_coordinates(
        lat0,
        lon0,
        crs
    )

    x, y, fclim = prepare_ffp_grid(
        FFP,
        x0,
        y0
    )

    rs = np.asarray(
        FFP["rs"],
        dtype=np.float32
    )

    n_rs = len(rs)

    # --------------------------------------------------------
    # Contour arrays
    # --------------------------------------------------------

    contour_lengths = []

    for i in range(n_rs):

        xr_i = FFP["xr"][i]
        yr_i = FFP["yr"][i]

        if xr_i is None or yr_i is None:
            contour_lengths.append(0)
            continue

        xr_i = np.asarray(xr_i)
        yr_i = np.asarray(yr_i)

        valid = (
            np.isfinite(xr_i)
            &
            np.isfinite(yr_i)
        )

        contour_lengths.append(
            np.sum(valid)
        )

    max_nodes = max(
        contour_lengths,
        default=0
    )

    contour_x = np.full(
        (n_rs, max_nodes),
        np.nan,
        dtype=np.float64
    )

    contour_y = np.full(
        (n_rs, max_nodes),
        np.nan,
        dtype=np.float64
    )

    nodes_count = np.zeros(
        n_rs,
        dtype=np.int32
    )

    for i in range(n_rs):

        xr_i = FFP["xr"][i]
        yr_i = FFP["yr"][i]

        if xr_i is None or yr_i is None:
            continue

        xr_i = np.asarray(
            xr_i,
            dtype=float
        )

        yr_i = np.asarray(
            yr_i,
            dtype=float
        )

        valid = (
            np.isfinite(xr_i)
            &
            np.isfinite(yr_i)
        )

        xr_i = xr_i[valid]
        yr_i = yr_i[valid]

        n = min(
            len(xr_i),
            len(yr_i)
        )

        if n < 2:
            continue

        contour_x[i, :n] = xr_i[:n] + x0
        contour_y[i, :n] = yr_i[:n] + y0

        nodes_count[i] = n

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    ds = xr.Dataset(

        data_vars={

            "fclim": (
                ["y", "x"],
                fclim,
                {
                    "long_name":
                        "Normalised footprint function",

                    "units":
                        "m-2",

                    "grid_mapping":
                        "spatial_ref"
                }
            ),

            "rs": (
                ["isopleth"],
                rs,
                {
                    "long_name":
                        "Footprint isopleth level",

                    "units":
                        "%"
                }
            ),

            "contour_x": (
                ["isopleth", "node"],
                contour_x,
                {
                    "units":
                        "m"
                }
            ),

            "contour_y": (
                ["isopleth", "node"],
                contour_y,
                {
                    "units":
                        "m"
                }
            ),

            "nodes_count": (
                ["isopleth"],
                nodes_count
            )
        },

        coords={

            "x": (
                ["x"],
                x,
                {
                    "standard_name":
                        "projection_x_coordinate",

                    "units":
                        "m"
                }
            ),

            "y": (
                ["y"],
                y,
                {
                    "standard_name":
                        "projection_y_coordinate",

                    "units":
                        "m"
                }
            ),

            "isopleth":
                np.arange(n_rs),

            "node":
                np.arange(max_nodes)
        },

        attrs={

            "title":
                f"FFP climatology - {site_name}",

            "site":
                site_name,

            "year":
                str(year),

            "latitude":
                lat0,

            "longitude":
                lon0,

            "crs":
                crs.to_string(),

            "Conventions":
                "CF-1.9"
        }
    )

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    ds["spatial_ref"] = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name":
                "transverse_mercator",

            "epsg_code":
                crs.to_string(),

            "crs_wkt":
                crs.to_wkt()
        }
    )

    # --------------------------------------------------------
    # Tower
    # --------------------------------------------------------

    ds["tower_x"] = xr.DataArray(
        x0,
        attrs={"units": "m"}
    )

    ds["tower_y"] = xr.DataArray(
        y0,
        attrs={"units": "m"}
    )

    ds["tower_latitude"] = xr.DataArray(
        lat0,
        attrs={"units": "degrees_north"}
    )

    ds["tower_longitude"] = xr.DataArray(
        lon0,
        attrs={"units": "degrees_east"}
    )

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    ds.to_netcdf(
        filename,
        format="NETCDF4",
        encoding={
            "fclim": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": -9999.0
            }
        }
    )

    print(
        f"Single NetCDF saved: {filename}"
    )


# ============================================================
# 6. Export SINGLE GeoTIFF
# ============================================================

def export_single_geotiff(
    FFP,
    filename,
    lat0,
    lon0,
    crs
):

    x0, y0 = get_tower_coordinates(
        lat0,
        lon0,
        crs
    )

    x, y, fclim = prepare_ffp_grid(
        FFP,
        x0,
        y0
    )

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])

    transform = from_origin(
        x.min() - dx / 2,
        y.max() + dy / 2,
        dx,
        dy
    )

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    with rasterio.open(

        filename,
        "w",

        driver="GTiff",

        height=fclim.shape[0],
        width=fclim.shape[1],

        count=1,

        dtype="float32",

        crs=crs.to_wkt(),

        transform=transform,

        nodata=-9999.0,

        compress="deflate"

    ) as dst:

        dst.write(
            fclim,
            1
        )

        dst.set_band_description(
            1,
            "FFP footprint density"
        )

    print(
        f"Single GeoTIFF saved: {filename}"
    )


# ============================================================
# 7. Export SINGLE contours
# ============================================================

def export_single_contours(
    FFP,
    filename,
    lat0,
    lon0,
    crs
):

    x0, y0 = get_tower_coordinates(
        lat0,
        lon0,
        crs
    )

    geometries, isopleths = prepare_contours(
        FFP,
        x0,
        y0
    )

    gdf = gpd.GeoDataFrame(
        {
            "isopleth": isopleths
        },
        geometry=geometries,
        crs=crs
    )

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    gdf.to_file(
        filename,
        driver="ESRI Shapefile"
    )

    print(
        f"Single contour SHP saved: {filename}"
    )


# ============================================================
# 8. Export MONTHLY NetCDF
# ============================================================

def export_monthly_netcdf(
    FFP_monthly,
    site_name,
    year,
    filename,
    lat0,
    lon0,
    crs
):

    x0, y0 = get_tower_coordinates(
        lat0,
        lon0,
        crs
    )

    # --------------------------------------------------------
    # Sort months
    # --------------------------------------------------------

    months = sorted(
        FFP_monthly.keys()
    )

    # --------------------------------------------------------
    # Prepare first FFP
    # --------------------------------------------------------

    first_ffp = FFP_monthly[months[0]]

    x, y, first_fclim = prepare_ffp_grid(
        first_ffp,
        x0,
        y0
    )

    ny, nx = first_fclim.shape

    # --------------------------------------------------------
    # Monthly raster
    # --------------------------------------------------------

    fclim_monthly = np.full(
        (
            len(months),
            ny,
            nx
        ),
        np.nan,
        dtype=np.float32
    )

    for m_index, month in enumerate(months):

        _, _, fclim = prepare_ffp_grid(
            FFP_monthly[month],
            x0,
            y0
        )

        if fclim.shape != (ny, nx):

            raise ValueError(
                f"Month {month} has different "
                f"grid dimensions: {fclim.shape}"
            )

        fclim_monthly[
            m_index,
            :, :
        ] = fclim

    # --------------------------------------------------------
    # Isopleths
    # --------------------------------------------------------

    rs = np.asarray(
        first_ffp["rs"],
        dtype=np.float32
    )

    n_rs = len(rs)

    # --------------------------------------------------------
    # Find maximum contour nodes
    # --------------------------------------------------------

    max_nodes = 0

    for month in months:

        FFP = FFP_monthly[month]

        for i in range(n_rs):

            if (
                FFP["xr"][i] is None
                or
                FFP["yr"][i] is None
            ):
                continue

            xr_i = np.asarray(
                FFP["xr"][i]
            )

            yr_i = np.asarray(
                FFP["yr"][i]
            )

            valid = (
                np.isfinite(xr_i)
                &
                np.isfinite(yr_i)
            )

            max_nodes = max(
                max_nodes,
                np.sum(valid)
            )

    # --------------------------------------------------------
    # Contour arrays
    #
    # dimensions:
    #
    # month × isopleth × node
    # --------------------------------------------------------

    contour_x = np.full(
        (
            len(months),
            n_rs,
            max_nodes
        ),
        np.nan,
        dtype=np.float64
    )

    contour_y = np.full(
        (
            len(months),
            n_rs,
            max_nodes
        ),
        np.nan,
        dtype=np.float64
    )

    nodes_count = np.zeros(
        (
            len(months),
            n_rs
        ),
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Fill contours
    # --------------------------------------------------------

    for m_index, month in enumerate(months):

        FFP = FFP_monthly[month]

        for i in range(n_rs):

            xr_i = FFP["xr"][i]
            yr_i = FFP["yr"][i]

            if xr_i is None or yr_i is None:
                continue

            xr_i = np.asarray(
                xr_i,
                dtype=float
            )

            yr_i = np.asarray(
                yr_i,
                dtype=float
            )

            valid = (
                np.isfinite(xr_i)
                &
                np.isfinite(yr_i)
            )

            xr_i = xr_i[valid]
            yr_i = yr_i[valid]

            n = min(
                len(xr_i),
                len(yr_i)
            )

            if n < 2:
                continue

            contour_x[
                m_index,
                i,
                :n
            ] = xr_i[:n] + x0

            contour_y[
                m_index,
                i,
                :n
            ] = yr_i[:n] + y0

            nodes_count[
                m_index,
                i
            ] = n

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    ds = xr.Dataset(

        data_vars={

            "fclim": (
                [
                    "month",
                    "y",
                    "x"
                ],
                fclim_monthly,
                {
                    "long_name":
                        "Monthly normalised "
                        "footprint function",

                    "units":
                        "m-2",

                    "grid_mapping":
                        "spatial_ref"
                }
            ),

            "rs": (
                ["isopleth"],
                rs,
                {
                    "long_name":
                        "Footprint isopleth",

                    "units":
                        "%"
                }
            ),

            "contour_x": (
                [
                    "month",
                    "isopleth",
                    "node"
                ],
                contour_x,
                {
                    "units":
                        "m"
                }
            ),

            "contour_y": (
                [
                    "month",
                    "isopleth",
                    "node"
                ],
                contour_y,
                {
                    "units":
                        "m"
                }
            ),

            "nodes_count": (
                [
                    "month",
                    "isopleth"
                ],
                nodes_count
            )
        },

        coords={

            "month": months,

            "x": (
                ["x"],
                x,
                {
                    "standard_name":
                        "projection_x_coordinate",

                    "units":
                        "m"
                }
            ),

            "y": (
                ["y"],
                y,
                {
                    "standard_name":
                        "projection_y_coordinate",

                    "units":
                        "m"
                }
            ),

            "isopleth":
                np.arange(n_rs),

            "node":
                np.arange(max_nodes)
        },

        attrs={

            "title":
                f"Monthly FFP climatology - {site_name}",

            "site":
                site_name,

            "year":
                str(year),

            "latitude":
                lat0,

            "longitude":
                lon0,

            "crs":
                crs.to_string(),

            "description":
                "Monthly flux footprint climatology",

            "Conventions":
                "CF-1.9"
        }
    )

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    ds["spatial_ref"] = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name":
                "transverse_mercator",

            "epsg_code":
                crs.to_string(),

            "crs_wkt":
                crs.to_wkt()
        }
    )

    # --------------------------------------------------------
    # Tower
    # --------------------------------------------------------

    ds["tower_x"] = xr.DataArray(
        x0,
        attrs={"units": "m"}
    )

    ds["tower_y"] = xr.DataArray(
        y0,
        attrs={"units": "m"}
    )

    ds["tower_latitude"] = xr.DataArray(
        lat0,
        attrs={"units": "degrees_north"}
    )

    ds["tower_longitude"] = xr.DataArray(
        lon0,
        attrs={"units": "degrees_east"}
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    ds.to_netcdf(
        filename,
        format="NETCDF4",
        encoding={
            "fclim": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": -9999.0
            }
        }
    )

    print(
        f"Monthly NetCDF saved: {filename}"
    )


# ============================================================
# 9. Export MONTHLY GeoTIFF
#
# One file
# 12 bands = 12 months
# ============================================================

def export_monthly_geotiff(
    FFP_monthly,
    filename,
    lat0,
    lon0,
    crs
):

    x0, y0 = get_tower_coordinates(
        lat0,
        lon0,
        crs
    )

    months = sorted(
        FFP_monthly.keys()
    )

    # --------------------------------------------------------
    # First grid
    # --------------------------------------------------------

    x, y, first_fclim = prepare_ffp_grid(
        FFP_monthly[months[0]],
        x0,
        y0
    )

    ny, nx = first_fclim.shape

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    dx = abs(
        x[1] - x[0]
    )

    dy = abs(
        y[1] - y[0]
    )

    transform = from_origin(
        x.min() - dx / 2,
        y.max() + dy / 2,
        dx,
        dy
    )

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create 12-band raster
    # --------------------------------------------------------

    with rasterio.open(

        filename,
        "w",

        driver="GTiff",

        height=ny,
        width=nx,

        count=len(months),

        dtype="float32",

        crs=crs.to_wkt(),

        transform=transform,

        nodata=-9999.0,

        compress="deflate"

    ) as dst:

        for band, month in enumerate(
            months,
            start=1
        ):

            _, _, fclim = prepare_ffp_grid(
                FFP_monthly[month],
                x0,
                y0
            )

            if fclim.shape != (
                ny,
                nx
            ):

                raise ValueError(
                    f"Month {month} has "
                    f"different grid dimensions"
                )

            dst.write(
                fclim,
                band
            )

            dst.set_band_description(
                band,
                f"Month {month:02d}"
            )

    print(
        f"Monthly GeoTIFF saved: {filename}"
    )


# ============================================================
# 10. Export MONTHLY contours
#
# One shapefile containing:
#
# January   20 40 60 80
# February  20 40 60 80
# ...
# December  20 40 60 80
# ============================================================

def export_monthly_contours(
    FFP_monthly,
    filename,
    lat0,
    lon0,
    crs
):

    x0, y0 = get_tower_coordinates(
        lat0,
        lon0,
        crs
    )

    months = sorted(
        FFP_monthly.keys()
    )

    geometries = []
    month_values = []
    isopleth_values = []

    # --------------------------------------------------------
    # Loop months
    # --------------------------------------------------------

    for month in months:

        FFP = FFP_monthly[month]

        geometries_month, isopleths = prepare_contours(
            FFP,
            x0,
            y0
        )

        for geometry, isopleth in zip(
            geometries_month,
            isopleths
        ):

            geometries.append(
                geometry
            )

            month_values.append(
                int(month)
            )

            isopleth_values.append(
                float(isopleth)
            )

    # --------------------------------------------------------
    # GeoDataFrame
    # --------------------------------------------------------

    gdf = gpd.GeoDataFrame(
        {
            "month": month_values,

            "isopleth": isopleth_values
        },

        geometry=geometries,

        crs=crs
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    gdf.to_file(
        filename,
        driver="ESRI Shapefile"
    )

    print(
        f"Monthly contour SHP saved: {filename}"
    )


# ============================================================
# 11. Export tower
# ============================================================

def export_tower(
    filename,
    site_name,
    lat0,
    lon0,
    crs
):

    x0, y0 = get_tower_coordinates(
        lat0,
        lon0,
        crs
    )

    gdf = gpd.GeoDataFrame(
        {
            "site": [site_name],

            "latitude": [lat0],

            "longitude": [lon0]
        },

        geometry=[
            Point(x0, y0)
        ],

        crs=crs
    )

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    gdf.to_file(
        filename,
        driver="ESRI Shapefile"
    )

    print(
        f"Tower SHP saved: {filename}"
    )


# ============================================================
# 12. SINGLE MASTER EXPORT
# ============================================================

def export_ffp_single(
    FFP,
    site_name,
    year,
    config,
    output_dir
):

    lat0 = float(
        config["lat_lon"]["lat"]
    )

    lon0 = float(
        config["lat_lon"]["lon"]
    )

    crs = get_utm_crs(
        lat0,
        lon0
    )

    basename = (
        f"FFP_single_"
        f"{site_name}_{year}"
    )

    nc_file = os.path.join(
        output_dir,
        "netcdf",
        basename + ".nc"
    )

    tif_file = os.path.join(
        output_dir,
        "geotiff",
        basename + ".tif"
    )

    shp_file = os.path.join(
        output_dir,
        "shapefile",
        basename + "_contours.shp"
    )

    tower_file = os.path.join(
        output_dir,
        "shapefile",
        basename + "_tower.shp"
    )

    export_single_netcdf(
        FFP,
        site_name,
        year,
        nc_file,
        lat0,
        lon0,
        crs
    )

    export_single_geotiff(
        FFP,
        tif_file,
        lat0,
        lon0,
        crs
    )

    export_single_contours(
        FFP,
        shp_file,
        lat0,
        lon0,
        crs
    )

    export_tower(
        tower_file,
        site_name,
        lat0,
        lon0,
        crs
    )

    return {
        "netcdf": nc_file,
        "geotiff": tif_file,
        "contours": shp_file,
        "tower": tower_file
    }


# ============================================================
# 13. MONTHLY MASTER EXPORT
# ============================================================

def export_ffp_monthly(
    FFP_monthly,
    site_name,
    year,
    config,
    output_dir
):

    lat0 = float(
        config["lat_lon"]["lat"]
    )

    lon0 = float(
        config["lat_lon"]["lon"]
    )

    crs = get_utm_crs(
        lat0,
        lon0
    )

    basename = (
        f"FFP_monthly_"
        f"{site_name}_{year}"
    )

    nc_file = os.path.join(
        output_dir,
        "netcdf",
        "Monthly",
        basename + ".nc"
    )

    tif_file = os.path.join(
        output_dir,
        "geotiff",
        "Monthly" ,
        basename + ".tif"
    )

    shp_file = os.path.join(
        output_dir,
        "shapefile",
        "Monthly" ,
        basename + "_contours.shp"
    )

    tower_file = os.path.join(
        output_dir,
        "shapefile",
        "Monthly",
        basename + "_tower.shp"
    )

    # --------------------------------------------------------
    # NetCDF
    # --------------------------------------------------------

    export_monthly_netcdf(
        FFP_monthly,
        site_name,
        year,
        nc_file,
        lat0,
        lon0,
        crs
    )

    # --------------------------------------------------------
    # GeoTIFF
    # --------------------------------------------------------

    export_monthly_geotiff(
        FFP_monthly,
        tif_file,
        lat0,
        lon0,
        crs
    )

    # --------------------------------------------------------
    # Contours
    # --------------------------------------------------------

    export_monthly_contours(
        FFP_monthly,
        shp_file,
        lat0,
        lon0,
        crs
    )

    # --------------------------------------------------------
    # Tower
    # --------------------------------------------------------

    export_tower(
        tower_file,
        site_name,
        lat0,
        lon0,
        crs
    )

    print("\n================================")
    print("MONTHLY FFP EXPORT COMPLETE")
    print("================================")
    print("NetCDF :", nc_file)
    print("GeoTIFF:", tif_file)
    print("Contours:", shp_file)
    print("Tower   :", tower_file)
    print("================================\n")

    return {
        "netcdf": nc_file,
        "geotiff": tif_file,
        "contours": shp_file,
        "tower": tower_file
    }