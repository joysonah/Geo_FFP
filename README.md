# Geo_FFP

An automation for the [Kljun et al. 2015](https://gmd.copernicus.org/articles/8/3695/2015/gmd-8-3695-2015.html) flux footprint (FFP) to generate monthly or cumulative single flux footprint map.

Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2015). A simple two-dimensional parameterisation for Flux Footprint Prediction (FFP). Geoscientific Model Development, 8(11), 3695–3713.

### Project structure
```
Geo_FFP/
│
├── main.py
├── config.yml
├── doc
├── requirements.txt
│
├── ffp_module/
│   ├── __init__.py          
│   ├── calc_footprint_FFP.py
│   ├── calc_footprint_FFP_climatology.py
│   ├── monthly_ffp.py
│   ├── FFP_clim_monthly.py
│   ├── GEE_module.py
│   ├── load_database.py
```
## Setup
1. Clone the Geo_FFP
2. Change directory (cd) to Geo_FFP
3. Create the virtual environment
   
   a) using windows terminal or cmd
   ```python -m venv .venv```

    * *Note* if "py" doesn't work - try "python" or "python3" instead - the call to python may be different depending on your installation

   b) using VScode
      1. Open the Geo_FFP folder in VS Code
      2. Hit ctrl + shift + p > and select "Create Python Environment"
      * Use Venv, not conda
      3. You will be prompted to select dependencies to install
      * Select "requirements.txt" form the menu.  This will automatically install all required packages for you.
  
4. Activate the virtual environment
   ```.\.venv\Scripts\activate```
5. Install dependencies
   ```pip install -r requirements.txt```
   * you dont need to do it again if you are using VScode
   
## Config the yml file and run
1. Open  and edit config.yml
   - Provide database path, years, site name, tower height, lat long
   - You can set 1 to compute single cumulative FFP with _run_single_ffp_ and monthly FFP with _run_monthly_ffp_
   
   **Example**
```
database:
  base_path: "C:/Users/joyso/Matlab/local_database_new1"
  years:
    - 2022
  site_name:
    - "YOUNG"
  levels:
    - "Clean/ThirdStage"
tower_height:
  - 4.08
variables:
  variables_to_read:
    - WS_1_1_1
    - hpbl
    - L
    - V_SIGMA
    - USTAR
    - WD_1_1_1
lat_lon:
  lat: 50.3623 
  lon: -100.2024
# -----------------------------
# PIPELINE CONTROL
# -----------------------------
pipeline:
  run_single_ffp: 1
  run_monthly_ffp: 1
```
2. Run the Geo_FFP
  ``` python main.py ```
3. Output
   This will generate FFP map, netcdf and FFP model output (as pickle) and will store the output in the database folder with FFP_output (e.g. database\FFP_output).
```
  FFP_output/
  ├── Sitename
  │   ├── year  
  │   │   ├── map
  │   │   ├── netcdf
  │   │   ├── pickle
```
