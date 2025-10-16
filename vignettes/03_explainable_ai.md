# Vignette 3 — Explainable AI for Seagrass Suitability

---

## 🎯 Objective
To train a **Random Forest Regressor** that predicts SSI from optical features
and quantify feature importance using **SHAP values**.

---

## ⚙️ Workflow

1. Merge optical (KD490, ADG443, APH443, BBP443) and physics SSI layers.
2. Train RF (300 trees, max_depth = 10, min_samples_leaf = 3).
3. Predict SSI_ML over grid.
4. Evaluate R² and MAE.
5. Compute SHAP values and produce feature-importance ranking.

---

## 📈 Results

| Metric | Value |
|--------|-------:|
| R² | 0.91 |
| MAE | 0.04 |

| Variable | Mean |SHAP| | Direction |
|-----------|-----------:|------------|
| KD490 | 0.324 | ↓ (attenuation) |
| ADG443 | 0.256 | ↓ |
| APH443 | 0.198 | ↑ |
| BBP443 | 0.144 | ↑ |

![](../outputs/shap_summary.png)

---

## 💬 Interpretation
Light attenuation (KD490) dominates suitability, confirming its control on
photic depth. SHAP plots help communicate how each optical component shapes
habitat potential, enabling transparent AI for ecology.
