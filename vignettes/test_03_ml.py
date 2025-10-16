import os, xarray as xr

def test_ml_rf_results():
    fn = "outputs/greifswalder_step3_ml.nc"
    assert os.path.exists(fn), "ML step output missing."
    ds = xr.open_dataset(fn)
    assert "SSI_ML" in ds, "Predicted SSI_ML missing."
    import numpy as np
    ssi = ds["SSI"].values.flatten()
    ssi_ml = ds["SSI_ML"].values.flatten()
    mask = np.isfinite(ssi) & np.isfinite(ssi_ml)
    corr = float(np.corrcoef(ssi[mask], ssi_ml[mask])[0, 1])

    assert corr > 0.7, f"Low correlation between SSI and SSI_ML ({corr:.2f})"
    print(f"✅ ML correlation: {corr:.2f}")
