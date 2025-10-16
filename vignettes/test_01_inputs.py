import os, xarray as xr

def test_inputs_retrieval():
    assert os.path.exists("outputs/greifswalder_inputs.nc"), \
        "Expected clean dataset not found."

    ds = xr.open_dataset("outputs/greifswalder_inputs.nc")
    expected_vars = {"KD490","ADG443","APH443","BBP443","PAR_surface","depth"}
    assert expected_vars.issubset(ds.data_vars), "Missing variables in dataset."
    assert (ds["depth"] < 0).any(), "Depth masking seems incorrect."
    print("✅ Inputs retrieval verified:", list(ds.data_vars))
