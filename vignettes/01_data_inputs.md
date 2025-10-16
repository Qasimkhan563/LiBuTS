# Vignette 1 — Data Retrieval & Harmonization

This vignette documents the first LiBuTS module, which collects and harmonizes
multi-source inputs describing the underwater light environment.

---

## 🎯 Objective
To build a consistent spatiotemporal dataset for the Greifswalder Bodden
combining:
- **Copernicus Marine OLCI optical properties** (300 m)
- **NASA POWER solar radiation** converted to PAR
- **GEBCO 2025 bathymetry**

---

## 🌍 Datasets & Access

| Source | Variables | URL |
|--------|------------|-----|
| Copernicus Marine | KD490, ADG443, APH443, BBP443 | <https://marine.copernicus.eu> |
| NASA POWER | ALLSKY_SFC_SW_DWN | <https://power.larc.nasa.gov> |
| GEBCO 2025 | Elevation | <https://www.gebco.net> |

The region of interest spans **13.3–13.7 E, 54.0–54.4 N** for **July 2024**.

---

## ⚙️ Methods

1. **KD490 and optical coefficients** retrieved via the `copernicusmarine` API  
   and averaged over July 2024.
2. **Surface PAR** derived from NASA POWER short-wave radiation using  
   `PAR = SWRAD × 0.45`.
3. **Bathymetry** clipped from GEBCO and interpolated onto the OLCI grid.
4. **Merge** all variables into a harmonized NetCDF file with attributes
   describing coordinate reference (EPSG:4326) and period metadata.

---

## 📦 Output

outputs/greifswalder_inputs.nc


This contain five data variables: KD490, ADG443, APH443, BBP443, PAR_surface, depth.

---

## 💬 Interpretation
This dataset forms the foundation for all subsequent physical and AI modeling,
representing the mean optical conditions for mid-summer 2024 in the southern
Baltic Sea.
