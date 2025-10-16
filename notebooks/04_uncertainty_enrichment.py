# ==============================================================
# LiBuTS Step 4 — Enrich with Physics Drivers & Uncertainty (fixed reshape)
# ==============================================================

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.utils import resample

# ---------------------------------------------------------------------
# 1️⃣ Load dataset
# ---------------------------------------------------------------------
ds = xr.open_dataset("outputs/greifswalder_step3_ml.nc")
lat, lon = ds["lat"], ds["lon"]
shape = ds["SSI"].shape

# ---------------------------------------------------------------------
# 2️⃣ Add synthetic physical drivers
# ---------------------------------------------------------------------
np.random.seed(42)
ds["temp_bottom"] = xr.DataArray(
    10 + 5*np.random.rand(*shape),
    dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
    attrs={"units": "°C", "long_name": "Bottom temperature"}
)
ds["nutrients"] = xr.DataArray(
    np.random.gamma(2, 0.3, shape),
    dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
    attrs={"units": "mmol m⁻³", "long_name": "Nutrient concentration proxy"}
)
ds["shear_stress"] = xr.DataArray(
    np.abs(np.random.normal(0.15, 0.05, shape)),
    dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
    attrs={"units": "Pa", "long_name": "Bottom shear stress proxy"}
)

# ---------------------------------------------------------------------
# 3️⃣ Prepare dataframe for ML classification
# ---------------------------------------------------------------------
df = (
    ds[["KD490","ADG443","APH443","BBP443",
        "depth","temp_bottom","nutrients","shear_stress","SSI"]]
    .to_dataframe()
    .dropna()
)
df["target"] = (df["SSI"] > 0.15).astype(int)
X = df.drop(["SSI","target"], axis=1)
y = df["target"]

# ---------------------------------------------------------------------
# 4️⃣ Cross-validated F1
# ---------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores=[]
for tr, te in kf.split(X):
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1)
    rf.fit(X.iloc[tr], y.iloc[tr])
    scores.append(f1_score(y.iloc[te], rf.predict(X.iloc[te])))
print(f"Mean F1 (5-fold): {np.mean(scores):.3f}")

# ---------------------------------------------------------------------
# 5️⃣ Bootstrap-based uncertainty
# ---------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1)
probs = np.vstack([
    rf.fit(*resample(X, y, random_state=i)).predict_proba(X)[:,1]
    for i in range(20)
])
uncertainty = probs.std(axis=0)
df["uncertainty"] = uncertainty

# ---------------------------------------------------------------------
# 6️⃣ Map uncertainty back to 2D grid safely
# ---------------------------------------------------------------------
# Start with NaN grid
uncert_grid = np.full(shape, np.nan, dtype=float)

# Match DataFrame indices (lat, lon) to DataArray positions
lat_vals = df.index.get_level_values("lat").values
lon_vals = df.index.get_level_values("lon").values
lat_index = {v: i for i,v in enumerate(ds["lat"].values)}
lon_index = {v: i for i,v in enumerate(ds["lon"].values)}

for la, lo, val in zip(lat_vals, lon_vals, df["uncertainty"].values):
    if la in lat_index and lo in lon_index:
        uncert_grid[lat_index[la], lon_index[lo]] = val

ds["uncertainty"] = xr.DataArray(
    uncert_grid,
    dims=("lat","lon"),
    coords={"lat": lat, "lon": lon},
    name="uncertainty",
    attrs={"long_name":"Model uncertainty (bootstrapped std)",
           "units":"(0–1)"}
)

# ---------------------------------------------------------------------
# 7️⃣ Save results
# ---------------------------------------------------------------------
ds.to_netcdf("outputs/greifswalder_step4_physics_uncertainty.nc")
df.to_csv("outputs/greifswalder_uncertainty.csv", index=False)
print("✅ Step 4 completed → enriched physics + uncertainty saved.")
