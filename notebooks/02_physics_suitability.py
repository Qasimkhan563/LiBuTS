import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Load cleaned dataset
ds = xr.open_dataset("outputs/greifswalder_inputs.nc")

# ---------------------------------------------------------------------
# 1️⃣ Compute Euphotic Depth (Zeu)
# ---------------------------------------------------------------------
Zeu = (4.6 / ds["KD490"]).clip(min=0, max=30)
Zeu.name = "Zeu"
Zeu.attrs["units"] = "m"
Zeu.attrs["long_name"] = "Euphotic depth (1% light level)"

# ---------------------------------------------------------------------
# 2️⃣ Compute PAR at seabed
# ---------------------------------------------------------------------
PAR_bed = ds["PAR_surface"] * np.exp(ds["KD490"] * ds["depth"])
PAR_bed.name = "PAR_bed"
PAR_bed.attrs["units"] = "E m⁻² d⁻¹"
PAR_bed.attrs["long_name"] = "Photosynthetically Active Radiation at seabed"

# ---------------------------------------------------------------------
# 3️⃣ Normalize key layers (0–1) and compute Suitability Index (SSI)
# ---------------------------------------------------------------------
def normalize(da):
    return (da - da.min()) / (da.max() - da.min())

ssi = (
    0.5 * normalize(PAR_bed) +
    0.3 * normalize(Zeu) -
    0.2 * normalize(abs(ds["depth"]))
)
ssi = ssi.clip(min=0, max=1)
ssi.name = "SSI"
ssi.attrs["long_name"] = "Seagrass Suitability Index (0–1)"
ssi.attrs["comment"] = "0=unsuitable, 1=highly suitable"

# ---------------------------------------------------------------------
# 4️⃣ Combine & save
# ---------------------------------------------------------------------
out = xr.Dataset({
    "KD490": ds["KD490"],
    "depth": ds["depth"],
    "Zeu": Zeu,
    "PAR_bed": PAR_bed,
    "SSI": ssi
})
out.attrs.update(ds.attrs)
out.attrs["step"] = "Physics-based seagrass suitability"
out.to_netcdf("outputs/greifswalder_step2_physics.nc")

print("✅ Step 2 completed → greifswalder_step2_physics.nc")

# ---------------------------------------------------------------------
# 5️⃣ Visualize key outputs
# ---------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
Zeu.plot(ax=axs[0], cmap="viridis")
axs[0].set_title("Euphotic Depth (m)")
PAR_bed.plot(ax=axs[1], cmap="plasma")
axs[1].set_title("PAR at Bed (E m⁻² d⁻¹)")
ssi.plot(ax=axs[2], cmap="YlGn")
axs[2].set_title("Seagrass Suitability Index")
plt.tight_layout()
plt.show()
