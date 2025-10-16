import os, xarray as xr

def test_uncertainty_layer():
    fn = "outputs/greifswalder_step4_physics_uncertainty.nc"
    assert os.path.exists(fn), "Uncertainty step output missing."
    ds = xr.open_dataset(fn)
    assert "uncertainty" in ds, "Uncertainty variable not found."
    mean_unc = float(ds["uncertainty"].mean())
    assert 0 <= mean_unc <= 1, "Uncertainty values out of expected range (0–1)."
    print("✅ Mean uncertainty:", round(mean_unc,3))
