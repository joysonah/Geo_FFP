
import os

# from load_database import load_database_data
# from FFP_clim_monthly import monthly_ffp_pipeline, single_ffp_plot
# import calc_footprint_FFP as ffp
# import calc_footprint_FFP_climatology as ffp_climatology

from FFP_module import (
    load_database_data,
    monthly_ffp_pipeline,
    single_ffp_plot,
    
)

from FFP_module.gee_ndvi import (
    init_gee,
    # single_ndvi,
    # monthly_ndvi,
    # monthly_ndvi_timeseries,
   
    # get_ffp_union_geometry,
    #load_monthly_ffp
)


import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pickle


import pyproj as pyproj
import contextily as ctx
import matplotlib.font_manager as fm

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import FancyArrowPatch

# ---------------------------
# Load configuration file
# ---------------------------
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# ---------------------------
# Read values from config
# ---------------------------
base_path = config["database"]["base_path"]
years = config["database"]["years"]
site_name = config["database"]["site_name"]
levels = config["database"]["levels"]
variables_to_read = config["variables"]["variables_to_read"]
z = config["tower_height"][0]

# ---------------------------
# Load database
# ---------------------------
all_data_frames = load_database_data(
    base_path,
    years,
    site_name,
    levels,
    variables_to_read
)

print(f"Data loaded for site: {site_name}, year: {years[0]}")


df = all_data_frames[site_name[0]]


k = 0.4
u = df["WS_1_1_1"].to_numpy(dtype=float)
ustar = df["USTAR"].to_numpy(dtype=float)
z_ref = z

# Avoid division by 0
ustar_safe = np.where(ustar <= 0, np.nan, ustar)

denom = 0.6 + 0.1 * np.exp((k * u) / ustar_safe)
df["Canopy_height"] = z_ref / denom
df["d"] = 0.65 * df["Canopy_height"]
df["Zm"] = z_ref - df["d"]

# Prepare data for FFP
# copy dataframe
df2 = df.copy()

# if Zm is missing, calculate it from z - d
# if d is also missing, and your site has no displacement estimate, you can use z directly
df2["Zm"] = df2["Zm"].fillna(z_ref - df2["d"])
df2["Zm"] = df2["Zm"].fillna(z_ref)

# keep only rows needed for FFP_climatology
cols = ["Zm", "WS_1_1_1", "hpbl", "L", "V_SIGMA", "USTAR", "WD_1_1_1"]
ffp_df = df2[cols].replace([np.inf, -np.inf], np.nan).dropna()

# optional but recommended: remove physically invalid rows
ffp_df = ffp_df[
    (ffp_df["USTAR"] >= 0.1) &
    (ffp_df["WS_1_1_1"] > 0) &
    (ffp_df["hpbl"] > 0) &
    (ffp_df["V_SIGMA"] > 0)
].copy()

print(f"Number of valid rows: {len(ffp_df)}")



#test in first 200 rows
# ffp_df = ffp_df.iloc[:100].copy() # remove this later, this is for testing

# -----------------------------
# Run single cumulative FFP
# -----------------------------
if config["pipeline"].get("run_single_ffp", 0) == 1:
    FFP = single_ffp_plot(ffp_df, z_ref, site_name[0], years[0], config)

# -----------------------------
# Run Monthly FFP
# -----------------------------
if config["pipeline"].get("run_monthly_ffp", 0) == 1:

    monthly_contours = monthly_ffp_pipeline(
        df2, z_ref, site_name[0], years[0], config
    )

else:

    # monthly_contours = load_monthly_ffp(config)
    print("Loading existing monthly FFP contours...")
# -----------------------------
# Step 2: plotting (independent)
# -----------------------------
# if config["pipeline"].get("run_plot_ffp", 0) == 1:
#     print("Plotting monthly FFP maps...")
#     plot_monthly_ffp_maps(ds, config)
    

# -----------------------------
# Run NDVI Extraction
# -----------------------------
if config["pipeline"].get("run_ndvi_extraction", 0) == 1:

    print("Starting NDVI pipeline...")

    init_gee(config)

    # --------------------------------
    # INPUT from Geo_FFP
    # --------------------------------
    # monthly_contours = ffp_results["contours_80"]

    lat = config["lat_lon"]["lat"]
    lon = config["lat_lon"]["lon"]

    # geom = get_ffp_union_geometry(config, monthly_contours, lat, lon)
   

    # --------------------------------
    # 1. Single NDVI
    # --------------------------------
    # img_single = single_ndvi(geom, config)
    # export_tif(img_single, geom, "NDVI_full_period", config)

    # --------------------------------
    # 2. Monthly NDVI
    # --------------------------------
    # monthly = monthly_ndvi(geom, config)

    # for m, img in monthly.items():
    #     export_tif(img, geom, f"NDVI_{m}", config)

    # --------------------------------
    # 3. Time series
    # --------------------------------
    # ts = monthly_ndvi_timeseries(geom, config)

    # df = pd.DataFrame(ts)
    # df.to_csv(config["output"]["ndvi_csv"], index=False)

    # print("NDVI pipeline completed successfully")
# -----------------------------
# Plotting timeseries NDVI
# -----------------------------
    plt.figure(figsize=(10,4))

    plt.plot(df["month"], df["NDVI"], marker="o")

    plt.xticks(rotation=45)
    plt.ylabel("NDVI")
    plt.xlabel("Month")
    plt.title(f"Monthly NDVI Time Series ({site_name[0]}, {years[0]})")
    plt.grid(True)


    plt.savefig(config["output"]["ndvi_plot"], dpi=100, bbox_inches="tight")

    plt.show()

# -----------------------------
# Plotting timeseries NDVI
# -----------------------------
#     plot_ffp_with_ndvi(
#     FFP,
#     img_single,
#     lat,
#     lon,
#     site_name,
#     years[0],
#     config
# )

#OPTIONAL: 
# -----------------------------
# Run Monthly FFP
# -----------------------------
# if config["pipeline"]["run_monthly_ffp"]:
#     print("Running monthly FFP...")
#     ds = run_from_config(df, config)
# else:
#     print("Skipping monthly FFP")

# -----------------------------
# Step 1: compute OR load
# -----------------------------
# if config["pipeline"]["run_monthly_ffp"]:
#     ds = run_from_config(df, config)
# else:
#     print("Loading existing FFP NetCDF...")
#     years = config["database"]["years"]
#     site_name = config["database"]["site_name"]

#     # output_path = config["output"]["ffp_climatology_file"]
#     output_path= os.path.join(
#     config["database"]["base_path"],
#     "FFP_output",
#     site_name[0],
#     str(years[0]),
#     "netcdf",
#     f"FFP_nc_{site_name[0]}_{years[0]}.nc"
#     )
#     ds = xr.open_dataset(output_path)