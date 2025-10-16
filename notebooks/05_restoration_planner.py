# ==============================================================
# LiBuTS Step 5 — Restoration Planner (Advanced NSGA-II)
# ==============================================================

import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 1️⃣ Load enriched dataset
# ---------------------------------------------------------------------
ds = xr.open_dataset("outputs/greifswalder_step4_physics_uncertainty.nc")
print("✅ Loaded:", list(ds.data_vars))

# Convert to DataFrame (only valid seafloor pixels)
df = ds[["SSI","SSI_ML","depth","uncertainty"]].to_dataframe().dropna().reset_index()

# ---------------------------------------------------------------------
# 2️⃣ Add derived metrics (to be optimized)
# ---------------------------------------------------------------------
df["CO2_potential"] = df["SSI"] * np.abs(df["depth"]) * 1.2
df["ALAN_risk"] = 1 - df["SSI_ML"]
if "shear_stress" in ds:
    df["shear_stress"] = ds["shear_stress"].to_dataframe().dropna().values[:len(df)]
else:
    df["shear_stress"] = np.abs(np.random.normal(0.15, 0.05, len(df)))

df = df.query("-12 <= depth <= -2")
print(f"Feasible restoration candidates: {len(df)}")

# ---------------------------------------------------------------------
# 3️⃣ Define Multi-Objective Optimization Problem
# ---------------------------------------------------------------------
class RestorationProblem(ElementwiseProblem):
    def __init__(self, df):
        # Store arrays directly
        self.co2 = df["CO2_potential"].values
        self.unc = df["uncertainty"].values
        self.alan = df["ALAN_risk"].values
        super().__init__(n_var=len(df), n_obj=3, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        mask = x > 0.8
        if mask.sum() == 0:
            f1 = f2 = f3 = 999.0
        else:
            # Use boolean indexing directly on arrays
            f1 = -np.mean(self.co2[mask])  # maximize CO₂
            f2 = np.mean(self.unc[mask])   # minimize uncertainty
            f3 = np.mean(self.alan[mask])  # minimize ALAN risk
        out["F"] = [f1, f2, f3]

# ---------------------------------------------------------------------
# 4️⃣ Run NSGA-II Optimization (new pymoo syntax)
# ---------------------------------------------------------------------
algorithm = NSGA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
)

termination = get_termination("n_gen", 50)
res = minimize(RestorationProblem(df), algorithm, termination, seed=42, verbose=True)
print("✅ Optimization completed")

# ---------------------------------------------------------------------
# 5️⃣ Extract Pareto front
# ---------------------------------------------------------------------
pareto = pd.DataFrame(res.F, columns=["-CO2","Uncertainty","ALAN"])
pareto["CO2"] = -pareto["-CO2"]

plt.figure(figsize=(7,5))
plt.scatter(pareto["Uncertainty"], pareto["CO2"], c=pareto["ALAN"], cmap="viridis", s=50, edgecolor="k")
plt.colorbar(label="ALAN risk")
plt.xlabel("Uncertainty ↓"); plt.ylabel("CO₂ potential ↑")
plt.title("Pareto Front — Restoration Trade-offs")
plt.tight_layout()
plt.savefig("outputs/pareto_front.png", dpi=300)
plt.show()

# ---------------------------------------------------------------------
# 6️⃣ Choose best compromise (knee point)
# ---------------------------------------------------------------------
best_idx = np.argmin(
    (pareto["Uncertainty"] - pareto["Uncertainty"].min())**2 +
    (pareto["ALAN"] - pareto["ALAN"].min())**2 -
    (pareto["CO2"] - pareto["CO2"].max())**2
)
print("Chosen solution index:", best_idx)

mask_opt = res.X[best_idx] > 0.8
restoration_df = df[mask_opt].copy()
print(f"Selected {len(restoration_df)} optimal restoration pixels")

# ---------------------------------------------------------------------
# 7️⃣ Export restoration map & summary
# ---------------------------------------------------------------------
geometry = [Point(xy) for xy in zip(restoration_df["lon"], restoration_df["lat"])]
gdf = gpd.GeoDataFrame(restoration_df, geometry=geometry, crs="EPSG:4326")

os.makedirs("outputs", exist_ok=True)
gdf.to_file("outputs/restoration_sites.gpkg", driver="GPKG")
restoration_df[["CO2_potential","uncertainty","ALAN_risk"]].describe().to_csv("outputs/restoration_summary.csv")

print("✅ Step 5 completed → GeoPackage + summary exported")

# ---------------------------------------------------------------------
# 8️⃣ Optional: visualize map of chosen sites
# ---------------------------------------------------------------------
plt.figure(figsize=(7,6))
plt.scatter(df["lon"], df["lat"], s=5, color="lightgrey", alpha=0.3)
plt.scatter(restoration_df["lon"], restoration_df["lat"], s=12, c=restoration_df["CO2_potential"], cmap="YlGn", edgecolor="k")
plt.colorbar(label="CO₂ potential")
plt.title("Optimal Restoration Cells (NSGA-II result)")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("outputs/restoration_map.png", dpi=300)
plt.show()
