import xarray as xr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1️⃣ Load datasets
# ---------------------------------------------------------------------
opt = xr.open_dataset("outputs/greifswalder_inputs_clean.nc")[["KD490","ADG443","APH443","BBP443"]]
phy = xr.open_dataset("outputs/greifswalder_step2_physics.nc")[["SSI","depth","Zeu","PAR_bed"]]

# ---------------------------------------------------------------------
# 2️⃣ Coordinate harmonization (safe)
# ---------------------------------------------------------------------
def harmonize_coords(ds):
    # Drop duplicates if both forms exist
    for d in ["lat", "latitude"]:
        if "lat" in ds.coords and "latitude" in ds.coords:
            ds = ds.drop_vars("lat")
    for d in ["lon", "longitude"]:
        if "lon" in ds.coords and "longitude" in ds.coords:
            ds = ds.drop_vars("lon")
    # Rename standard CF names to shorter ones
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    # Remove CRS if exists
    ds = ds.drop_vars(["crs"], errors="ignore")
    return ds

opt = harmonize_coords(opt)
phy = harmonize_coords(phy)

# ---------------------------------------------------------------------
# 3️⃣ Ensure matching grid
# ---------------------------------------------------------------------
phy = phy.interp(lat=opt["lat"], lon=opt["lon"], method="nearest")
ds = xr.merge([opt, phy])

print("✅ Merged dataset dims:", ds.dims)
print("Vars:", list(ds.data_vars))

# ---------------------------------------------------------------------
# 4️⃣ Prepare tabular data for ML
# ---------------------------------------------------------------------
df = ds[["KD490","ADG443","APH443","BBP443","SSI"]].to_dataframe().reset_index().dropna()
print(f"Training samples available: {len(df)}")

X = df[["KD490","ADG443","APH443","BBP443"]].values
y = df["SSI"].values

# ---------------------------------------------------------------------
# 5️⃣ Train Random Forest
# ---------------------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=300, max_depth=10,
    min_samples_leaf=3, random_state=42, n_jobs=-1
)
rf.fit(X, y)
y_pred = rf.predict(X)

print(f"🔹 R² = {r2_score(y, y_pred):.3f}")
print(f"🔹 MAE = {mean_absolute_error(y, y_pred):.3f}")

# ---------------------------------------------------------------------
# 6️⃣ Predict SSI_ML map
# ---------------------------------------------------------------------
features = np.column_stack([
    ds["KD490"].values.ravel(),
    ds["ADG443"].values.ravel(),
    ds["APH443"].values.ravel(),
    ds["BBP443"].values.ravel()
])
mask = ~np.isnan(features).any(axis=1)
ssi_ml = np.full_like(ds["KD490"].values.ravel(), np.nan, dtype=float)
ssi_ml[mask] = rf.predict(features[mask])

ssi_ml = xr.DataArray(
    ssi_ml.reshape(ds["KD490"].shape),
    dims=("lat", "lon"),
    coords={"lat": ds["lat"], "lon": ds["lon"]},
    name="SSI_ML",
    attrs={
        "long_name": "AI-predicted Seagrass Suitability Index",
        "units": "0–1",
        "comment": "Predicted via Random Forest from OLCI optical features"
    }
)

ds_ml = ds.merge(ssi_ml)
ds_ml.to_netcdf("outputs/greifswalder_step3_ml.nc")
print("✅ Step 3 completed → greifswalder_step3_ml.nc")

# ---------------------------------------------------------------------
# 7️⃣ SHAP explainability
# ---------------------------------------------------------------------
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

shap.summary_plot(
    shap_values,
    df[["KD490","ADG443","APH443","BBP443"]],
    show=False, plot_size=(8,5)
)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=300)
plt.show()

# ---------------------------------------------------------------------
# 8️⃣ Compare SSI vs SSI_ML maps
# ---------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12,4))
ds["SSI"].plot(ax=axs[0], cmap="YlGn", vmin=0, vmax=1)
axs[0].set_title("Physics-based SSI")
ssi_ml.plot(ax=axs[1], cmap="YlGn", vmin=0, vmax=1)
axs[1].set_title("AI-predicted SSI (RF)")
plt.tight_layout()
plt.show()
