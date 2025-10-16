import xarray as xr, numpy as np, os

def test_physics_suitability():
    fn = "outputs/greifswalder_step2_physics.nc"
    assert os.path.exists(fn), "Physics step output missing."
    ds = xr.open_dataset(fn)
    assert "SSI" in ds, "SSI variable missing."
    assert np.isfinite(ds["SSI"].mean()), "SSI contains invalid values."
    print("✅ Physics SSI mean:", float(ds["SSI"].mean()))
