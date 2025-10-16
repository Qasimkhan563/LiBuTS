# 🌊 LiBuTS — Light-Budget Twin for Seagrass Restoration

A **physics-guided, explainable-AI digital twin** that integrates **Copernicus Marine**, **NASA POWER**, and **GEBCO 2025** data to map and optimize **seagrass restoration zones** under light- and uncertainty-driven scenarios.

---

## 🧠 Concept
LiBuTS fuses satellite optics, physics-based PAR modeling, and explainable ML to reveal where seagrass can thrive despite turbidity and artificial-light pressures in the Baltic Sea.

---

## ⚙️ Pipeline Overview

| Step | Script | Purpose | Output |
|:--|:--|:--|:--|
| 1️⃣ | `01_inputs_retrieval.py` | Fetch KD490 + ADG443 + APH443 + BBP443 (OLCI) + PAR (NASA) + Depth (GEBCO) | `greifswalder_inputs_clean.nc` |
| 2️⃣ | `02_physics_suitability.py` | Compute Euphotic Depth & PAR at Bed → SSI | `greifswalder_step2_physics.nc` |
| 3️⃣ | `03_ml_rf_shap.py` | Random Forest + SHAP Explainability | `greifswalder_step3_ml.nc` |
| 4️⃣ | `04_uncertainty_enrichment.py` | Add drivers + bootstrap uncertainty | `greifswalder_step4_physics_uncertainty.nc` |
| 5️⃣ | `05_restoration_planner.py` | NSGA-II multi-objective restoration planner | `restoration_sites.gpkg` |
| 💻 | `app/dashboard.py` | Interactive digital-twin dashboard | Web app on port 5016 |

---

## 🚀 Quick Start
```bash
cd env
conda env create -f environment.yml
conda activate libuts
make all
make app



## ✅ Validation & Testing
To verify the entire LiBuTS pipeline, run:

```bash
pytest -v vignettes/
