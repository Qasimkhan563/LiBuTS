#!/usr/bin/env python
# ==============================================================
# LiBuTS-X Step 1 — Real Data Retrieval  (single clean NetCDF)
#   • KD490, ADG443, APH443, BBP443  → Copernicus Marine (OLCI, 300 m)
#   • PAR_surface                     → NASA POWER (SWRAD × 0.45)
#   • Depth (GEBCO 2025)              → Local NetCDF, clipped to AOI
# ==============================================================

import os, requests
import numpy as np
import xarray as xr
import rioxarray
from copernicusmarine import open_dataset, login

# --------------------------------------------------------------
# 1️⃣  Credentials  (replace with your own)
# --------------------------------------------------------------
USERNAME = "test@example.com"
PASSWORD = "Your password for copernicus marine"
login(USERNAME, PASSWORD)

# --------------------------------------------------------------
# 2️⃣  Define AOI & period
# --------------------------------------------------------------
AOI   = dict(lon_min=13.3, lon_max=13.7, lat_min=54.0, lat_max=54.4)
START, END = "2024-07-01", "2024-07-31"

# ==============================================================
# Copernicus Marine (Optical variables)
# ==============================================================

print("🔹 Fetching KD490 + optical coefficients from Copernicus Marine …")

# KD490 (Transparency)
kd_ds = open_dataset(
    dataset_id="cmems_obs-oc_bal_bgc-transp_nrt_l3-olci-300m_P1D",
    variables=["KD490"],
    minimum_longitude=AOI["lon_min"], maximum_longitude=AOI["lon_max"],
    minimum_latitude=AOI["lat_min"], maximum_latitude=AOI["lat_max"],
    start_datetime=START, end_datetime=END,
)
kd = kd_ds["KD490"].mean("time").rename("KD490")

# Optical absorption / scattering
optics_ds = open_dataset(
    dataset_id="cmems_obs-oc_bal_bgc-optics_nrt_l3-olci-300m_P1D",
    variables=["ADG443", "APH443", "BBP443"],
    minimum_longitude=AOI["lon_min"], maximum_longitude=AOI["lon_max"],
    minimum_latitude=AOI["lat_min"], maximum_latitude=AOI["lat_max"],
    start_datetime=START, end_datetime=END,
)
adg = optics_ds["ADG443"].mean("time").rename("ADG443")
aph = optics_ds["APH443"].mean("time").rename("APH443")
bbp = optics_ds["BBP443"].mean("time").rename("BBP443")

# ==============================================================
# NASA POWER (PAR_surface)
# ==============================================================

print("🔹 Fetching PAR_surface from NASA POWER …")

def fetch_nasa_power(lat, lon, start=START, end=END):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=ALLSKY_SFC_SW_DWN&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start.replace('-','')}&end={end.replace('-','')}&format=JSON"
    )
    r = requests.get(url, timeout=30)
    data = r.json()["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    return np.mean(list(data.values())) * 0.45  # convert SWRAD→PAR

par_val = fetch_nasa_power(
    (AOI["lat_min"] + AOI["lat_max"]) / 2,
    (AOI["lon_min"] + AOI["lon_max"]) / 2
)
par_surface = xr.full_like(kd, par_val).rename("PAR_surface")
par_surface.attrs["units"] = "E m⁻² d⁻¹"

# ==============================================================
# GEBCO 2025 Bathymetry (local NetCDF)
# ==============================================================

print("🔹 Reading and clipping GEBCO 2025 bathymetry …")
gebco_path = "data/gebco_2025.nc"

depth = (
    xr.open_dataset(gebco_path)["elevation"]
    .rename("depth")
    .rio.write_crs("EPSG:4326")
)

depth = depth.rio.clip_box(
    minx=AOI["lon_min"], miny=AOI["lat_min"],
    maxx=AOI["lon_max"], maxy=AOI["lat_max"]
)

lon_name = [c for c in kd.coords if "lon" in c.lower()][0]
lat_name = [c for c in kd.coords if "lat" in c.lower()][0]

depth = depth.interp(
    lon=kd[lon_name],
    lat=kd[lat_name],
    method="nearest"
)
depth.attrs.update({"units": "m", "long_name": "Seafloor elevation (GEBCO 2025)"})

# Mask land (positive values → NaN)
depth = depth.where(depth < 0)
depth.attrs["comment"] = "Positive values masked (land)"

# ==============================================================
# Merge & Save — single clean NetCDF
# ==============================================================

print("🔹 Merging all layers …")
ds = xr.merge([kd, adg, aph, bbp, par_surface, depth])
ds.attrs.update({
    "AOI": "Greifswalder Bodden",
    "period": "July 2024 (real data)",
    "source": "Copernicus Marine OLCI + NASA POWER + GEBCO 2025",
    "note": "Clean harmonized dataset (land masked, single output)"
})

os.makedirs("outputs", exist_ok=True)
out_nc = "outputs/greifswalder_inputs.nc"
ds.to_netcdf(out_nc)
print(f"✅ Saved clean harmonized dataset → {out_nc}")
