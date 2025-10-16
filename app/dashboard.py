#!/usr/bin/env python
# ============================================================
# 🌊 LiBuTS Pro++ — Light-Based Underwater Terrain Twin
# (Copernicus + NASA + GEBCO + Physics + ML + NSGA-II + SHAP)
# Enhanced: Plotly 3D, basemaps, export, MaterialTemplate
# ============================================================

import os, time, numpy as np, pandas as pd, xarray as xr
import panel as pn, hvplot.xarray, hvplot.pandas, holoviews as hv, geoviews as gv
import matplotlib.pyplot as plt, shap, plotly.graph_objects as go
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor

pn.extension('tabulator', 'plotly', 'floatpanel', 'echarts', sizing_mode="stretch_width")

# ------------------------------------------------------------
# 🌍 Load Data
# ------------------------------------------------------------
ds = xr.open_dataset("outputs/greifswalder_step3_ml.nc")
print("✅ Loaded:", list(ds.data_vars))

restoration_csv = "outputs/restoration_summary.csv"
pareto_csv = "outputs/pareto_front.csv"
has_restoration = os.path.exists(restoration_csv)
if has_restoration:
    restoration = pd.read_csv(restoration_csv)
    pareto = pd.read_csv(pareto_csv) if os.path.exists(pareto_csv) else None

# ------------------------------------------------------------
# 🎨 Theme & Branding
# ------------------------------------------------------------
ACCENT = "#00b4d8"
BG = "#ffffff"         # lighter background
TEXT = "#000000"       # black text

style = {"color": TEXT, "font-family": "Inter, sans-serif", "font-size": "14px"}

pn.config.raw_css.append(f"""
body {{
  background-color: {BG};
  color: {TEXT};
  font-family: 'Inter', sans-serif;
}}
:root {{
  --accent-color: {ACCENT};
}}
@keyframes sonar {{
  0% {{ box-shadow: 0 0 0 0 rgba(0,180,216,0.4); }}
  70% {{ box-shadow: 0 0 0 25px rgba(0,180,216,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(0,180,216,0); }}
}}
.sonar {{
  height: 120px; width: 120px; margin: auto;
  border-radius: 50%; background: {ACCENT};
  animation: sonar 2s infinite;
}}
""")

# ------------------------------------------------------------
# 📋 Sidebar Summary
# ------------------------------------------------------------
sidebar = pn.Column(
    pn.pane.Markdown(f"""
# 🌊 **LiBuTS Pro++**
### *Light-Based Underwater Terrain Twin*  

**Data Sources**
🛰️ *Copernicus Marine – OLCI Optics*  
☀️ *NASA POWER – Surface PAR*  
🌎 *GEBCO 2025 – Bathymetry*

**Model**
⚙️ Physics + AI (Random Forest)  
🧠 SHAP Explainability  
⚖️ NSGA-II Optimization  
📍 *Greifswalder Bodden – Jul 2024*
""", styles=style, width=260),
    pn.layout.Divider(),
)

# ------------------------------------------------------------
# 🗺 Spatial Explorer (with basemap)
# ------------------------------------------------------------
var_select = pn.widgets.Select(name="Variable", options=list(ds.data_vars), value="SSI")
import geoviews as gv
import cartopy.crs as ccrs
from holoviews.operation.datashader import regrid

@pn.depends(var_select)
def map_view(var):
    da = ds[var]

    # --- CRS and extent ---
    crs = ccrs.PlateCarree()
    lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
    lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
    extent = (lon_min, lon_max, lat_min, lat_max)

    # --- GeoViews image (1D coords are fine) ---
    img = gv.Image(
        da,
        kdims=["lon", "lat"],
        crs=crs
    ).opts(
        cmap="viridis",
        colorbar=True,
        tools=["hover", "wheel_zoom", "pan"],
        active_tools=["wheel_zoom"],
        frame_width=850,
        frame_height=600,
        projection=crs,
        global_extent=False,
        xlim=(lon_min, lon_max),
        ylim=(lat_min, lat_max),
        title=f"🗺️ {var} — Spatial Distribution"
    )

    # --- Base map overlay ---
    base = gv.tile_sources.EsriImagery.opts(alpha=0.6)

    # --- Combine ---
    return (base * regrid(img)).opts(framewise=True)

tab1 = pn.Column(
    pn.pane.Markdown("## 🗺️ Spatial Layers Overview", styles=style),
    var_select, map_view,
    pn.pane.Markdown("_Toggle variable and explore overlayed on basemap._", styles=style)
)

# ------------------------------------------------------------
# 📈 Cross-Section Explorer (time-aware)
# ------------------------------------------------------------
lat_slider = pn.widgets.FloatSlider(
    name="Latitude", start=float(ds.lat.min()), end=float(ds.lat.max()),
    step=0.01, value=float(ds.lat.mean()), width=400
)

time_slider = None
if "time" in ds.dims:
    time_slider = pn.widgets.DiscreteSlider(name="Time", options=list(map(str, ds.time.values)))

@pn.depends(lat_slider)
def ssi_profile(lat):
    cut = ds.sel(lat=lat, method="nearest")
    return cut.hvplot.line(
        x="lon", y="SSI", color=ACCENT, line_width=3,
        title=f"📈 SSI cross-section at {lat:.3f}° N"
    )

tab2 = pn.Column(
    pn.pane.Markdown("## 📈 SSI Cross-Section", styles=style),
    lat_slider, ssi_profile,
    pn.pane.Markdown("_Observe suitability gradients longitudinally._", styles=style)
)

# ------------------------------------------------------------
# 🔬 Correlation + Summary Statistics
# ------------------------------------------------------------
def correlation_heatmap():
    df = ds.to_dataframe().dropna()
    corr = df.corr().stack().reset_index()
    corr.columns = ["x", "y", "correlation"]
    return corr.hvplot.heatmap(
        x="x", y="y", C="correlation", cmap="coolwarm", clim=(-1, 1),
        width=500, height=450, title="🔬 Variable Correlation Matrix", tools=["hover"]
    )

def summary_table():
    df = ds.to_dataframe().describe().T.reset_index().rename(columns={'index': 'Variable'})
    return pn.widgets.Tabulator(df, height=350, theme='fast', layout='fit_data_stretch')

tab3 = pn.Row(
    pn.Column(correlation_heatmap),
    pn.Column("### 📊 Summary Statistics", summary_table()),
)

# ------------------------------------------------------------
# 🧠 SHAP Explainability
# ------------------------------------------------------------
df = ds[["KD490", "ADG443", "APH443", "BBP443"]].to_dataframe().dropna()
y = ds["SSI"].to_dataframe().reindex(df.index).fillna(0)
rf = RandomForestRegressor(n_estimators=150, random_state=42)
rf.fit(df, y.values.ravel())

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(df)
mean_abs = np.abs(shap_values).mean(axis=0)
ranking = pd.DataFrame({'Variable': df.columns, 'Mean |SHAP|': mean_abs}).sort_values('Mean |SHAP|', ascending=False)

shap_var = pn.widgets.Select(name="Variable", options=list(df.columns), value="KD490")

@pn.depends(shap_var)
def shap_dependence(var):
    plt.figure(figsize=(6,4))
    shap.dependence_plot(var, shap_values, df, show=False)
    buf = BytesIO(); plt.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close()
    return pn.pane.PNG(buf.getvalue(), height=400)

ranking_table = pn.widgets.Tabulator(ranking, height=200, theme='fast')

tab4 = pn.Column(
    pn.pane.Markdown("## 🧠 SHAP Explainability", styles=style),
    shap_var, shap_dependence,
    pn.pane.Markdown("### Mean |SHAP| Variable Importance", styles=style),
    ranking_table
)

# ------------------------------------------------------------
# 🌱 Restoration Planner (auto coords + export)
# ------------------------------------------------------------
if has_restoration:
    if not any(col.lower() in ["lon","longitude","x"] for col in restoration.columns):
        np.random.seed(42)
        restoration["lon"] = np.random.uniform(13.2,14.3,len(restoration))
        restoration["lat"] = np.random.uniform(53.9,54.6,len(restoration))

    co2_slider = pn.widgets.FloatSlider(name="Min CO₂ (kt eq)", start=0,
        end=float(restoration["CO2_potential"].max()), value=0)
    risk_slider = pn.widgets.FloatSlider(name="Max ALAN Risk", start=0,
        end=float(restoration["ALAN_risk"].max()), value=float(restoration["ALAN_risk"].max()))
    unc_slider = pn.widgets.FloatSlider(name="Max Uncertainty", start=0,
        end=float(restoration["uncertainty"].max()), value=float(restoration["uncertainty"].max()))

    @pn.depends(co2_slider, risk_slider, unc_slider)
    def filtered_sites(co2_min, alan_max, unc_max):
        f = restoration.query("CO2_potential>=@co2_min & ALAN_risk<=@alan_max & uncertainty<=@unc_max")
        if f.empty:
            return pn.pane.Markdown("❌ No sites match filters.", styles={"color":"#ff4d6d"})

        map_points = f.hvplot.points("lon","lat",color="CO2_potential",cmap="viridis",size=9,
                                     tools=["hover"],width=650,height=450,
                                     title=f"🌱 {len(f)} Restoration Sites")

        # --- CSV export helper ---
        def _make_csv():
            return f.to_csv(index=False)

        download = pn.widgets.FileDownload(
            filename="filtered_restoration.csv",
            label="⬇️ Download CSV",
            button_type="success",
        )
        download.callback = _make_csv   # ✅ correct pattern for Panel>=1.3


        metrics = pn.Row(
            pn.indicators.Number(name="Total CO₂ (kt)", value=f["CO2_potential"].sum(),
                                 format="{value:.2f}", default_color=ACCENT),
            pn.indicators.Number(name="Mean Risk", value=f["ALAN_risk"].mean(),
                                 format="{value:.2f}", default_color="#ffb703"),
            pn.indicators.Number(name="Mean Uncertainty", value=f["uncertainty"].mean(),
                                 format="{value:.2f}", default_color="#fb8500")
        )
        return pn.Column(map_points, metrics, download)

    pareto_plot = (
        pareto.hvplot.scatter(x="Uncertainty", y="CO2_potential", c="ALAN_risk",
            cmap="plasma", colorbar=True, size=60, width=550, height=450,
            title="⚖️ Pareto Front — CO₂ vs Uncertainty")
        if pareto is not None else hv.Curve([])
    )

    tab5 = pn.Row(
        pn.Column("## 🌱 Restoration Planner", co2_slider, risk_slider, unc_slider, filtered_sites),
        pn.Column("## ⚖️ Trade-offs", pareto_plot)
    )
else:
    tab5 = pn.Column("### 🌱 Restoration Planner",
                     pn.pane.Markdown("_No restoration data available yet._", styles=style))

# ------------------------------------------------------------
# 🌊 3D Bathymetry Viewer (Plotly)
# ------------------------------------------------------------
if "depth" in ds.data_vars:
    z = ds["depth"].values
    fig = go.Figure(data=[go.Surface(z=z, colorscale="Viridis")])
    fig.update_layout(title="3D Bathymetry (Depth)", autosize=True, height=500,
                      scene=dict(zaxis_title='Depth', xaxis_title='Lon', yaxis_title='Lat'))
    tab6 = pn.Column(pn.pane.Plotly(fig))
else:
    tab6 = pn.pane.Markdown("🧭 No 'depth' variable found.")

# ------------------------------------------------------------
# 🧩 Combine Tabs in Material Template
# ------------------------------------------------------------
tabs = pn.Tabs(
    ("🗺️ Spatial", tab1),
    ("📈 Cross-section", tab2),
    ("🔬 Correlation", tab3),
    ("🧠 Explainability", tab4),
    ("🌱 Restoration", tab5),
    ("🌊 3D Bathymetry", tab6)
)

dashboard = pn.template.MaterialTemplate(
    title="🌊 LiBuTS — Seagrass Restoration Digital Twin",
    sidebar=[sidebar],
    main=[tabs],
)

# Apply styling safely for Panel 1.8.2
dashboard.header_background = ACCENT
dashboard.sidebar_width = 280
dashboard.main_max_width = "90%"

# Add gentle padding/centering
for item in dashboard.main:
    if hasattr(item, 'margin'):
        item.margin = (10, 25)


# ------------------------------------------------------------
# 🚀 Launch Local Web App
# ------------------------------------------------------------
if __name__ == "__main__":
    pn.serve(dashboard, show=True, port=5015)

