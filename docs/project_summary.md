# 🌊 LiBuTS — Light-Budget Twin for Seagrass Restoration

## Overview
**LiBuTS** (*Light-Budget Twin for Seagrass Restoration*) is a physics-guided, explainable-AI modeling framework designed to evaluate seagrass habitat suitability and restoration potential under varying underwater light and turbidity conditions.  
The project integrates multi-source datasets — **Copernicus Marine (OLCI)** optical products, **NASA POWER** radiation data, and **GEBCO 2025** bathymetry — to create a harmonized digital twin of coastal ecosystems, starting with the **Greifswalder Bodden** in the southern Baltic Sea.

---

## Scientific Motivation
Seagrass meadows play a critical role in **blue carbon storage**, **biodiversity maintenance**, and **coastal resilience**, yet they are declining globally due to light limitation, eutrophication, and changing underwater lightscapes.  
Understanding and predicting how physical–biogeochemical and optical variables interact to support seagrass growth is essential for targeted restoration and climate mitigation. LiBuTS bridges this gap by coupling **physics-based euphotic depth modeling** with **machine learning explainability** and **multi-objective optimization**.

---

## Core Methodological Modules

| Step | Component | Description | Key Output |
|------|------------|--------------|-------------|
| **1. Data Retrieval** | Copernicus OLCI + NASA POWER + GEBCO | Extracts KD490, ADG443, APH443, BBP443, PAR, and bathymetry data harmonized to 300 m | `greifswalder_inputs_clean.nc` |
| **2. Physics Model** | Light Budget Model | Derives euphotic depth (Zeu), bottom PAR, and Seagrass Suitability Index (SSI) | `greifswalder_step2_physics.nc` |
| **3. Machine Learning** | Random Forest + SHAP | Learns SSI patterns from optical predictors and interprets key drivers | `greifswalder_step3_ml.nc`, `shap_summary.png` |
| **4. Uncertainty Quantification** | Bootstrap + Physical Drivers | Adds temperature, nutrients, and shear-stress proxies to estimate uncertainty | `greifswalder_step4_physics_uncertainty.nc` |
| **5. Restoration Optimization** | NSGA-II (multi-objective) | Identifies restoration zones maximizing CO₂ gain while minimizing uncertainty and ALAN risk | `restoration_sites.gpkg`, `pareto_front.csv` |
| **6. Interactive Dashboard** | Panel + Plotly + GeoViews | Explorable visualization of maps, cross-sections, SHAP results, and Pareto front | `dashboard.py` |

---

## Novelty
- 🔹 Integrates **real optical satellite data** with **AI explainability** and **restoration optimization**.  
- 🔹 Provides a **transparent, reproducible light-budget workflow** ready for transfer to other coastal basins.  
- 🔹 Enables **policy-ready visual outputs** aligned with blue carbon and nature-based climate solutions.  
- 🔹 Embeds **interactive analytics** for scientists, NGOs, and marine planners.

---

## Study Region
**Greifswalder Bodden**, a shallow lagoon in the southern Baltic Sea (Germany), represents an ecologically sensitive and restoration-priority area.  
LiBuTS leverages high-resolution bathymetry and OLCI water-quality data to assess the interplay between turbidity, depth, and photosynthetically active radiation (PAR).

---

## Key Outcome
LiBuTS delivers a reproducible framework and digital twin for:
- Mapping seagrass habitat suitability and light limitation;
- Estimating CO₂ storage potential;
- Quantifying uncertainty;
- Prioritizing restoration zones through multi-objective optimization;
- Visualizing results via an interactive open-source dashboard.

---

**Principal Developer:** Muhammad Qasim (2025)  
**Affiliations:** Independent Researcher / Environmental Data Scientist  
**License:** MIT  
**Keywords:** Seagrass, Light Budget, Explainable AI, NSGA-II, Restoration, Blue Carbon, Copernicus Marine
