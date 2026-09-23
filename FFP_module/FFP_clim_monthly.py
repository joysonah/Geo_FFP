from logging import config
import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import netCDF4
import pyproj as pyproj
import matplotlib.pyplot as plt
import contextily as ctx
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import FancyArrowPatch

from FFP_module.calc_footprint_FFP_climatology import FFP_climatology
from FFP_module.ffp_export import (export_ffp_single, export_ffp_monthly, get_utm_crs, get_tower_coordinates)

## FFP single/cumulative Climatology Module
def single_ffp_plot(ffp_df, z_ref, site_name, year, config, save_pickle=True, save_nc=True):
    """
    Compute single FFP, save full object, save single NetCDF, and generate a map.
    """
    
    # -----------------------------
    # 1. Compute single FFP
    # -----------------------------
    FFP = FFP_climatology(
        zm=z_ref,
        z0=None,
        umean=ffp_df["WS_1_1_1"].tolist(),
        h=ffp_df["hpbl"].tolist(),
        ol=ffp_df["L"].tolist(),
        sigmav=ffp_df["V_SIGMA"].tolist(),
        ustar=ffp_df["USTAR"].tolist(),
        wind_dir=ffp_df["WD_1_1_1"].tolist(),
        domain=[-1000, 1000, -1000, 1000],
        nx=1000,
        ny=1000,
        rs=[20., 40., 60., 80.]
    )

    # -----------------------------
    # 2. Save full FFP object (Pickle)
    # -----------------------------
    if save_pickle:
        pickle_path = os.path.join(
            config["database"]["base_path"], "FFP_output", site_name, str(year), "pickle"
        )
        os.makedirs(pickle_path, exist_ok=True)
        pickle_file = os.path.join(pickle_path, f"FFP_full_{site_name}_{year}.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(FFP, f)
        print(f"Full FFP object saved: {pickle_file}")

#     # -----------------------------
#     # 3. Save single NetCDF
#     # -----------------------------
#     if save_nc:
#         nc_path = os.path.join(
#             config["database"]["base_path"], "FFP_output", site_name, str(year), "netcdf"
#         )
#         os.makedirs(nc_path, exist_ok=True)
#         nc_file = os.path.join(nc_path, f"FFP_single_{site_name}_{year}.nc")

#         # convert grid
#         wgs84 = pyproj.CRS('EPSG:4326')
#         mercator = pyproj.CRS('EPSG:3857')
#         proj = pyproj.Transformer.from_crs(wgs84, mercator, always_xy=True)
#         x0, y0 = proj.transform(lon0, lat0)

#         ds = xr.Dataset(
#             {
#                 "fclim": (["y", "x"], FFP["fclim_2d"])
#             },
#             coords={
#                 "x": (["x"], FFP["x_2d"][0, :]+x0),
#                 "y": (["y"], FFP["y_2d"][:, 0]+y0)
#             },
#             attrs={
#                 "description": f"Single footprint grid for {site_name}, {year}",
#                 "units": "m-2"
#             }
#         )

#         ds.rio.write_crs(mercator, inplace=True)
#         ds4326 = ds.rio.reproject(wgs84)
        # # -----------------------------
        # # Save NetCDF
        # # -----------------------------
#         ds4326.to_netcdf(nc_file)
#         print(f"Single FFP NetCDF saved: {nc_file}")

        # # -----------------------------
        # # Save NetCDF. Geotiff and Shapefile contour
        # # -----------------------------
        output_dir = os.path.join(
            config["database"]["base_path"],
            "FFP_output",
            site_name,
            str(year)
        )

        export_ffp_single(
        FFP=FFP,
        site_name=site_name,
        year=year,
        config=config,
        output_dir=output_dir
        )

    # -----------------------------
    # 4. Plot single map
    # -----------------------------
    lat0, lon0 = config["lat_lon"]["lat"], config["lat_lon"]["lon"]

    wgs84 = pyproj.CRS('EPSG:4326')
    mercator = pyproj.CRS('EPSG:3857')
    proj = pyproj.Transformer.from_crs(wgs84, mercator, always_xy=True)
    x0, y0 = proj.transform(lon0, lat0)

    X_abs = FFP['x_2d'] + x0
    Y_abs = FFP['y_2d'] + y0
    Z = FFP['fclim_2d']

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(X_abs, Y_abs, Z, levels=100, cmap='viridis', alpha=0.4)
    plt.colorbar(cs, label='Footprint Weight (m$^{-2}$)')

    for i, rs_val in enumerate(FFP['rs']):
        if FFP['xr'][i] is not None and FFP['yr'][i] is not None:
            xr_arr = np.array(FFP['xr'][i]) + x0
            yr_arr = np.array(FFP['yr'][i]) + y0
            ax.plot(xr_arr, yr_arr, linewidth=1, label=f'{int(rs_val*100)}%')

    ax.scatter(x0, y0, c='red', marker='^', s=80, label='Tower')

    # Zoom to 80% contour
    idx80 = next((i for i, r in enumerate(FFP['rs']) if r in [80, 0.8]), None)
    if idx80 is not None and FFP['xr'][idx80] is not None:
        xr80 = np.array(FFP['xr'][idx80]) + x0
        yr80 = np.array(FFP['yr'][idx80]) + y0
        margin = 50
        ax.set_xlim(xr80.min() - margin, xr80.max() + margin)
        ax.set_ylim(yr80.min() - margin, yr80.max() + margin)

    google_sat = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    ctx.add_basemap(ax, source=google_sat, crs="EPSG:3857")

    fontprops = fm.FontProperties(size=10)
    scalebar = AnchoredSizeBar(ax.transData, 100, '100 m', loc='lower center',
                               pad=0.5, color='white', frameon=False,
                               size_vertical=1, fontproperties=fontprops)
    ax.add_artist(scalebar)

    arrow = FancyArrowPatch((0.95, 0.2), (0.95, 0.35), transform=ax.transAxes,
                            arrowstyle='-|>', color='white', mutation_scale=20)
    ax.add_artist(arrow)
    ax.text(0.95, 0.37, 'N', transform=ax.transAxes,
            ha='center', va='bottom', color='white', fontsize=20, fontweight='bold')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Single Footprint Map ({site_name}, {year})')
    ax.legend()

    map_path = os.path.join(
        config["database"]["base_path"], "FFP_output", site_name, str(year), "map"
    )
    os.makedirs(map_path, exist_ok=True)
    map_file = os.path.join(map_path, f"FP_map_single_{site_name}_{year}.png")
    plt.savefig(map_file, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Single FFP map saved: {map_file}")

    return FFP

# ============================================================
# FFP Monthly Climatology Pipeline
# ============================================================

def monthly_ffp_pipeline(df, z_ref, site_name, year, config):
    """
    Compute monthly FFP climatologies and save:

    1. Individual monthly FFP pickle files
    2. One combined annual/monthly FFP pickle containing all months
    3. One combined NetCDF
    4. One 12-band GeoTIFF
    5. One contour shapefile containing all months
    6. One tower shapefile
    7. Individual monthly PNG maps

    Returns
    -------
    dict
        Dictionary containing monthly FFP objects and output paths.
    """

    # ========================================================
    # 1. Basic information
    # ========================================================

    lat0 = float(config["lat_lon"]["lat"])
    lon0 = float(config["lat_lon"]["lon"])

    base_path = config["database"]["base_path"]

    # Main output directory
    output_dir = os.path.join(
        base_path,
        "FFP_output",
        site_name,
        str(year)
    )

    # Subdirectories
    pickle_dir = os.path.join(output_dir, "pickle")
    netcdf_dir = os.path.join(output_dir, "netcdf")
    geotiff_dir = os.path.join(output_dir, "geotiff")
    shapefile_dir = os.path.join(output_dir, "shapefile")
    map_dir = os.path.join(
        output_dir,
        "map",
        "Monthly_FFP_maps"
    )

    # Create directories
    for directory in [
        pickle_dir,
        netcdf_dir,
        geotiff_dir,
        shapefile_dir,
        map_dir
    ]:
        os.makedirs(directory, exist_ok=True)

    # ========================================================
    # 2. UTM projection
    # ========================================================

    crs_utm = get_utm_crs(lat0, lon0)

    x0, y0 = get_tower_coordinates(
        lat0,
        lon0,
        crs_utm
    )

    print("\n==============================================")
    print("Monthly FFP climatology")
    print("==============================================")
    print(f"Site       : {site_name}")
    print(f"Year       : {year}")
    print(f"Latitude   : {lat0}")
    print(f"Longitude  : {lon0}")
    print(f"Projection : {crs_utm.to_string()}")
    print(f"Output     : {output_dir}")
    print("==============================================\n")

    # ========================================================
    # 3. Container for all monthly FFP objects
    # ========================================================

    FFP_monthly = {}

    # ========================================================
    # 4. Group data by month
    # ========================================================

    months = df.groupby(pd.Grouper(freq="MS"))

    # ========================================================
    # 5. Calculate FFP for each month
    # ========================================================

    for month_start, g in months:

        month_number = month_start.month
        month_str = month_start.strftime("%Y_%m")

        # ----------------------------------------------------
        # Remove invalid meteorological observations
        # ----------------------------------------------------

        required_columns = [
            "WS_1_1_1",
            "hpbl",
            "L",
            "V_SIGMA",
            "USTAR",
            "WD_1_1_1",
            "Zm"
        ]

        g = g.dropna(
            subset=required_columns
        ).copy()

        # # ----------------------------------------------------
        # # Minimum number of observations
        # # ----------------------------------------------------

        # if len(g) < 50:

        #     print(
        #         f"Skipping {month_start:%Y-%m}: "
        #         f"too few valid points ({len(g)})"
        #     )

        #     continue

        # print(
        #     f"Running FFP for {month_start:%Y-%m}, "
        #     f"n={len(g)}"
        # )

        # ====================================================
        # 6. Run FFP climatology
        # ====================================================

        FFP = FFP_climatology(

            zm=float(g["Zm"].median()),

            z0=None,

            umean=g["WS_1_1_1"].tolist(),

            h=g["hpbl"].tolist(),

            ol=g["L"].tolist(),

            sigmav=g["V_SIGMA"].tolist(),

            ustar=g["USTAR"].tolist(),

            wind_dir=g["WD_1_1_1"].tolist(),

            domain=[
                -2000,
                2000,
                -2000,
                2000
            ],

            nx=300,
            ny=300,

            rs=[
                20.,
                40.,
                60.,
                80.
            ],

            smooth_data=1,

            crop=0,

            pulse=100,

            verbosity=0,

            fig=0
        )

        # ====================================================
        # 7. Store FFP object in monthly dictionary
        # ====================================================

        FFP_monthly[month_number] = FFP

        # ====================================================
        # 8. Save individual monthly pickle
        # ====================================================

        monthly_pickle = os.path.join(
            pickle_dir,
            f"FFP_full_{site_name}_{year}_{month_number:02d}.pkl"
        )

        with open(
            monthly_pickle,
            "wb"
        ) as f:

            pickle.dump(
                FFP,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        print(
            f"  Saved pickle: "
            f"{os.path.basename(monthly_pickle)}"
        )

        # ====================================================
        # 9. Create monthly PNG map
        # ====================================================

        X_abs = FFP["x_2d"] + x0
        Y_abs = FFP["y_2d"] + y0

        Z = FFP["fclim_2d"]

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        # ----------------------------------------------------
        # Continuous footprint
        # ----------------------------------------------------

        contourf = ax.contourf(
            X_abs,
            Y_abs,
            Z,
            levels=30
        )

        cbar = fig.colorbar(
            contourf,
            ax=ax
        )

        cbar.set_label(
            "Footprint density (m$^{-2}$)"
        )

        # ----------------------------------------------------
        # FFP isopleth contours
        # ----------------------------------------------------

        rs_values = FFP["rs"]

        contour = ax.contour(
            X_abs,
            Y_abs,
            Z,
            levels=rs_values
        )

        # ----------------------------------------------------
        # Label contours
        # ----------------------------------------------------

        labels = {}

        for rs_val in rs_values:

            if rs_val <= 1:
                labels[rs_val] = (
                    f"{int(rs_val * 100)}%"
                )
            else:
                labels[rs_val] = (
                    f"{int(rs_val)}%"
                )

        fmt = {
            level: labels.get(
                level,
                str(level)
            )
            for level in contour.levels
        }

        ax.clabel(
            contour,
            inline=True,
            fontsize=9,
            fmt=fmt
        )

        # ----------------------------------------------------
        # Tower location
        # ----------------------------------------------------

        ax.scatter(
            x0,
            y0,
            marker="^",
            s=100,
            edgecolor="black",
            zorder=10,
            label="EC tower"
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        ax.set_xlabel(
            "Easting (m)"
        )

        ax.set_ylabel(
            "Northing (m)"
        )

        ax.set_title(
            f"{site_name} FFP Climatology — "
            f"{month_start:%Y-%m}"
        )

        ax.legend()

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        # ----------------------------------------------------
        # Save PNG
        # ----------------------------------------------------

        png_file = os.path.join(
            map_dir,
            f"FP_map_{site_name}_{year}_{month_number:02d}.png"
        )

        plt.tight_layout()

        plt.savefig(
            png_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"  Saved map: "
            f"{os.path.basename(png_file)}"
        )

    # ========================================================
    # 10. Check whether any months were successfully processed
    # ========================================================

    if len(FFP_monthly) == 0:

        print(
            "\nNo valid monthly FFP calculations."
        )

        return None

    # ========================================================
    # 11. Save ALL monthly FFP objects into ONE pickle
    # ========================================================

    combined_pickle = os.path.join(
        pickle_dir,
        f"FFP_monthly_{site_name}_{year}.pkl"
    )

    with open(
        combined_pickle,
        "wb"
    ) as f:

        pickle.dump(
            FFP_monthly,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print("\n==============================================")
    print("Combined monthly FFP pickle saved")
    print("==============================================")
    print(combined_pickle)
    print(
        f"Months included: "
        f"{sorted(FFP_monthly.keys())}"
    )
    print("==============================================\n")

    # ========================================================
    # 12. Export combined GIS products
    # ========================================================

    combined_outputs = export_ffp_monthly(
        FFP_monthly=FFP_monthly,
        site_name=site_name,
        year=year,
        config=config,
        output_dir=output_dir
    )

    # ========================================================
    # 13. Add combined pickle to output dictionary
    # ========================================================

    combined_outputs["pickle"] = combined_pickle

    # ========================================================
    # 14. Print final summary
    # ========================================================

    print("\n==============================================")
    print("FFP MONTHLY PIPELINE COMPLETE")
    print("==============================================")

    print(
        f"Site: {site_name}"
    )

    print(
        f"Year: {year}"
    )

    print(
        f"Months calculated: "
        f"{sorted(FFP_monthly.keys())}"
    )

    print("\nOutputs:")

    for key, path in combined_outputs.items():

        print(
            f"  {key:10s}: {path}"
        )

    print("==============================================\n")

    # ========================================================
    # 15. Return results
    # ========================================================

    return {
        "FFP_monthly": FFP_monthly,
        "outputs": combined_outputs
    }
# # ## FFP Monthly Climatology Module
# def monthly_ffp_pipeline(df, z_ref, site_name, year, config):
#     """
#     Compute FFP climatology for each month and save:
#     - full FFP object (Pickle)
#     - gridded NetCDF
#     - footprint map (PNG)
#     """

#     months = df.groupby(pd.Grouper(freq="MS"))

#     lat0, lon0 = config["lat_lon"]["lat"], config["lat_lon"]["lon"]
#     wgs84 = pyproj.CRS("EPSG:4326")
#     mercator = pyproj.CRS("EPSG:3857")
#     proj = pyproj.Transformer.from_crs(wgs84, mercator, always_xy=True)
#     x0, y0 = proj.transform(lon0, lat0)

#     for month_start, g in months:
#         g = g.dropna(subset=["WS_1_1_1","hpbl","L","V_SIGMA","USTAR","WD_1_1_1"]).copy()
#         if len(g) < 50:
#             print(f"Skipping {month_start:%Y-%m}: too few valid points ({len(g)})")
#             continue

#         print(f"Running FFP for {month_start:%Y-%m}, n={len(g)}")

#         # Compute FFP climatology for the month
#         FFP = FFP_climatology(
#             zm=float(g["Zm"].median()),
#             z0=None,
#             umean=g["WS_1_1_1"].tolist(),
#             h=g["hpbl"].tolist(),
#             ol=g["L"].tolist(),
#             sigmav=g["V_SIGMA"].tolist(),
#             ustar=g["USTAR"].tolist(),
#             wind_dir=g["WD_1_1_1"].tolist(),
#             domain=[-2000, 2000, -2000, 2000],
#             nx=300,
#             ny=300,
#             rs=[20., 40., 60., 80.],
#             smooth_data=1,
#             crop=0,
#             pulse=100,
#             verbosity=0,
#             fig=0
#         )

#         month_str = month_start.strftime("%Y_%m")

#         # -----------------------------
#         # 1. Save full FFP object (Pickle)
#         # -----------------------------
#         pickle_path = os.path.join(config["database"]["base_path"], "FFP_output", site_name, str(year), "pickle")
#         os.makedirs(pickle_path, exist_ok=True)
#         pickle_file = os.path.join(pickle_path, f"FFP_full_{site_name}_{month_str}.pkl")
#         with open(pickle_file, "wb") as f:
#             pickle.dump(FFP, f)
#         print(f"Saved full FFP object: {pickle_file}")

#         # -----------------------------
#         # 2. Save NetCDF
#         # -----------------------------
#         nc_path = os.path.join(config["database"]["base_path"], "FFP_output", site_name, str(year), "netcdf")
#         os.makedirs(nc_path, exist_ok=True)
#         nc_file = os.path.join(nc_path, f"FFP_monthly_{site_name}_{month_str}.nc")

#         # # convert grid
#         # wgs84 = pyproj.CRS('EPSG:4326')
#         # mercator = pyproj.CRS('EPSG:3857')
#         # proj = pyproj.Transformer.from_crs(wgs84, mercator, always_xy=True)
#         # x0, y0 = proj.transform(lon0, lat0)	

#         # ds = xr.Dataset(
#         #     {
#         #         "fclim": (["y","x"], FFP["fclim_2d"])
#         #     },
#         #     coords={
#         #         "x": (["x"], FFP["x_2d"][0,:]+x0),
#         #         "y": (["y"], FFP["y_2d"][:,0]+y0)
#         #     },
#         #     attrs={"description": f"Monthly footprint for {site_name} {month_str}"}
#         # )

#         # ds.rio.write_crs(mercator, inplace=True)
#         # ds4326 = ds.rio.reproject(wgs84)

#         # ds4326.to_netcdf(nc_file)
#         # print(f"Saved NetCDF: {nc_file}")

#         # -----------------------------
#         # 3. Plot single map
#         # -----------------------------
#         X_abs = FFP["x_2d"] + x0
#         Y_abs = FFP["y_2d"] + y0
#         Z = FFP["fclim_2d"]

#         fig, ax = plt.subplots(figsize=(10, 8))
#         cs = ax.contourf(X_abs, Y_abs, Z, levels=100, cmap="viridis", alpha=0.4)
#         plt.colorbar(cs, label="Footprint Weight (m$^{-2}$)")

#         for i, rs_val in enumerate(FFP["rs"]):
#             if FFP["xr"][i] is not None and FFP["yr"][i] is not None:
#                 xr_arr = np.array(FFP["xr"][i]) + x0
#                 yr_arr = np.array(FFP["yr"][i]) + y0
#                 ax.plot(xr_arr, yr_arr, linewidth=1, label=f"{int(rs_val*100)}%")

#         ax.scatter(x0, y0, c="red", marker="^", s=80, label="Tower")

#         # Zoom 80% contour
#         idx80 = next((i for i, r in enumerate(FFP["rs"]) if r in [80,0.8]), None)
#         if idx80 is not None and FFP["xr"][idx80] is not None:
#             xr80 = np.array(FFP["xr"][idx80]) + x0
#             yr80 = np.array(FFP["yr"][idx80]) + y0
#             margin = 50
#             ax.set_xlim(xr80.min() - margin, xr80.max() + margin)
#             ax.set_ylim(yr80.min() - margin, yr80.max() + margin)

#         google_sat = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
#         ctx.add_basemap(ax, source=google_sat, crs="EPSG:3857")

#         ax.set_xlabel("x [m]")
#         ax.set_ylabel("y [m]")
#         ax.set_title(f"Monthly Footprint Map ({site_name}, {month_str})")
#         ax.legend()

#         map_path = os.path.join(config["database"]["base_path"], "FFP_output", site_name, str(year), "map", "Monthly_FFP_maps")
#         os.makedirs(map_path, exist_ok=True)
#         map_file = os.path.join(map_path, f"FP_map_{site_name}_{month_str}.png")
#         plt.savefig(map_file, dpi=100, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Saved map: {map_file}")

#     print("Monthly FFP processing complete.")

# ## FFP Monthly Climatology Module
# def compute_monthly_ffp_climatology(df, config):
#     """
#     Compute monthly footprint climatology and return xarray Dataset.
#     """

#     req = config["ffp"]["required_columns"]

#     domain = config["ffp"]["domain"]
#     nx = config["ffp"]["nx"]
#     ny = config["ffp"]["ny"]
#     rs = config["ffp"]["rs"]
#     min_points = config["ffp"]["min_points"]

#     smooth_data = config["ffp"].get("smooth_data", 1)
#     crop = config["ffp"].get("crop", 0)

#     monthly_maps = []
#     monthly_times = []
#     xr_list = []
#     yr_list = []

#     for month_start, g in df.groupby(pd.Grouper(freq="MS")):
#         g = g[req].dropna().copy()

#         g = g[
#             (g["USTAR"] >= 0.1) &
#             (g["WS_1_1_1"] > 0) &
#             (g["hpbl"] > 0) &
#             (g["V_SIGMA"] > 0)
#         ]

#         if len(g) < min_points:
#             print(f"Skipping {month_start:%Y-%m}: too few valid rows ({len(g)})")
#             continue

#         zm_val = float(g["Zm"].median())

#         print(f"Running {month_start:%Y-%m} with n={len(g)}")

#         FFP = myfootprint.FFP_climatology(
#             zm=zm_val,
#             z0=None,
#             umean=g["WS_1_1_1"].tolist(),
#             h=g["hpbl"].tolist(),
#             ol=g["L"].tolist(),
#             sigmav=g["V_SIGMA"].tolist(),
#             ustar=g["USTAR"].tolist(),
#             wind_dir=g["WD_1_1_1"].tolist(),
#             domain=domain,
#             nx=nx,
#             ny=ny,
#             rs=rs,
#             smooth_data=smooth_data,
#             crop=crop,
#             pulse=100,
#             verbosity=0,
#             fig=0
#         )

#         monthly_maps.append(FFP["fclim_2d"])
#         monthly_times.append(month_start)
#         xr_list.append(FFP["x_2d"][0, :])
#         yr_list.append(FFP["y_2d"][:, 0])

#     # Build dataset
#     x = xr_list[0]
#     y = yr_list[0]

#     fclim_monthly = xr.DataArray(
#         data=np.stack(monthly_maps, axis=0),
#         coords={
#             "time": monthly_times,
#             "y": y,
#             "x": x
#         },
#         dims=("time", "y", "x"),
#         name="fclim_monthly",
#         attrs={
#             "long_name": "Monthly footprint climatology",
#             "units": "m-2"
#         }
#     )

#     ds = xr.Dataset({"fclim_monthly": fclim_monthly})

#     return ds

# def plot_monthly_ffp_maps(ds, config):
#     """
#     Plot and save monthly footprint maps with basemap and 80% contour.
#     """

#     lat0 = config["lat_lon"]["lat"]
#     lon0 = config["lat_lon"]["lon"]

#     zoom_radius = config["plot"].get("zoom_radius", 500)
#     # output_dir = config["output"]["ffp_plot_dir"]
#     years = config["database"]["years"]
#     site_name = config["database"]["site_name"]

#     # output_path = config["output"]["ffp_climatology_file"]
#     output_dir= os.path.join(
#     config["database"]["base_path"],
#     "FFP_output",
#     site_name[0],
#     str(years[0]),
#     "map",
#     "Monthly_FFP_maps",
    
#     )

#     os.makedirs(output_dir, exist_ok=True)

#     # projection
#     proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
#     x0, y0 = proj.transform(lon0, lat0)

#     # convert grid
#     x_abs = ds["x"].values + x0
#     y_abs = ds["y"].values + y0
#     X_abs, Y_abs = np.meshgrid(x_abs, y_abs)

#     google_sat = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

#     for t in ds["time"].values:

#         fig, ax = plt.subplots(figsize=(8, 8))

#         Z = ds["fclim_monthly"].sel(time=t).values

#         # heatmap
#         cf = ax.contourf(
#             X_abs, Y_abs, Z,
#             levels=30,
#             cmap="viridis",
#             alpha=0.45
#         )

#         # ---- 80% contour ----
#         zflat = Z.ravel()
#         zflat = zflat[np.isfinite(zflat)]
#         zflat = zflat[zflat > 0]

#         if len(zflat) > 0:
#             zsort = np.sort(zflat)[::-1]
#             csum = np.cumsum(zsort)
#             csum = csum / csum[-1]

#             idx80 = np.searchsorted(csum, 0.80)
#             if idx80 < len(zsort):
#                 thr80 = zsort[idx80]

#                 ax.contour(
#                     X_abs, Y_abs, Z,
#                     levels=[thr80],
#                     colors="red",
#                     linewidths=1.5
#                 )

#         # tower
#         ax.scatter(x0, y0, c="red", marker="^", s=50)

#         # zoom
#         ax.set_xlim(x0 - zoom_radius, x0 + zoom_radius)
#         ax.set_ylim(y0 - zoom_radius, y0 + zoom_radius)

#         # basemap
#         ctx.add_basemap(ax, source=google_sat, crs="EPSG:3857")

#         # # -----------------------------
#         # Scale bar
#         # -----------------------------
#         fontprops = fm.FontProperties(size=10)
#         scalebar = AnchoredSizeBar(ax.transData,
#                                 100,      # 100 meters
#                                 '100 m', 
#                                 loc='lower center',
#                                 pad=0.5,
#                                 color='white',
#                                 frameon=False,
#                                 size_vertical=1,
#                                 fontproperties=fontprops)
#         ax.add_artist(scalebar)

#         # -----------------------------
#         # North arrow like QGIS
#         # -----------------------------
#         arrow = FancyArrowPatch((0.95, 0.2), (0.95, 0.35), 
#                                 transform=ax.transAxes,
#                                 arrowstyle='-|>', 
#                                 color='white', 
#                                 mutation_scale=20)
#         ax.add_artist(arrow)
#         ax.text(0.95, 0.37, 'N', transform=ax.transAxes,
#                 ha='center', va='bottom',color='white', fontsize=20, fontweight='bold')

#         # title
#         tstr = np.datetime_as_string(t, unit="D")[:7]
#         ax.set_title(f"Footprint {tstr}")

#         ax.set_aspect("equal")
#         ax.set_xlabel("x [m]")
#         ax.set_ylabel("y [m]")
#         ax.legend()

#         # colorbar
#         cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
#         cbar.set_label("Footprint weight (m$^{-2}$)")

#         # save
#         outfile = os.path.join(output_dir, f"FFP_{tstr}.png")
#         plt.savefig(outfile, dpi=300, bbox_inches="tight")
#         plt.close(fig)

#         print(f"Saved: {outfile}")

# def run_from_config(df, config):
#     """
#     Wrapper to compute and save NetCDF.
#     """
#     ds = compute_monthly_ffp_climatology(df, config)
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

#     # Create directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     ds.to_netcdf(output_path)

#     print(f"Saved: {output_path}")

#     return ds

