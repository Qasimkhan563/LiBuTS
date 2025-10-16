import os, geopandas as gpd, pandas as pd

def test_restoration_outputs():
    gpkg = "outputs/restoration_sites.gpkg"
    csv = "outputs/restoration_summary.csv"
    assert os.path.exists(gpkg), "Restoration sites file missing."
    assert os.path.exists(csv), "Restoration summary CSV missing."

    gdf = gpd.read_file(gpkg)
    df = pd.read_csv(csv)
    assert len(gdf) > 0, "No restoration sites found."
    assert "CO2_potential" in df.columns, "Missing CO₂ potential column."
    print("✅ Restoration sites:", len(gdf))
