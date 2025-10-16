# Vignette 4 — Physical Drivers & Uncertainty Quantification

---

## 🎯 Objective
To enrich the model with additional physical variables and quantify prediction
uncertainty using bootstrap ensembles.

---

## ⚙️ Added Variables
- Bottom temperature (°C)
- Nutrient proxy (mmol m⁻³)
- Shear stress (Pa)

---

## ⚙️ Methodology
1. Create binary suitability target (SSI > 0.15).
2. Perform 5-fold cross-validation with `RandomForestClassifier`.
3. Use 20 bootstrap iterations to estimate predictive standard deviation.
4. Map uncertainty back to 2-D grid.

---

## 📊 Results

| Metric | Mean F1 Score |
|---------|---------------:|
|                       | 0.83 |

![](../outputs/uncertainty_map.png)

---

## 💬 Interpretation
High uncertainty zones coincide with intermediate depths where physical and
optical gradients overlap, guiding future data collection for calibration.
