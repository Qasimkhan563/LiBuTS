# Vignette 2 — Physics-Based Light Budget & Suitability Index

---

## 🎯 Objective
To estimate underwater light penetration and compute a physics-driven
**Seagrass Suitability Index (SSI)**.

---

## ⚙️ Equations

- **Euphotic Depth**  
  \[
  Z_{eu} = \frac{4.6}{K_{D490}}
  \]
- **PAR at Seabed**  
  \[
  PAR_{bed} = PAR_{surface} \times e^{(K_{D490} \times depth)}
  \]
- **Suitability Index**  
  \[
  SSI = 0.5·norm(PAR_{bed}) + 0.3·norm(Z_{eu}) - 0.2·norm(|depth|)
  \]
  clipped to \[0, 1\].

---

## ⚙️ Implementation
1. Compute `Zeu`, `PAR_bed`, and normalized terms using `xarray`.
2. Combine into `SSI` and export `outputs/greifswalder_step2_physics.nc`.
3. Visualize with Matplotlib color maps (`viridis`, `YlGn`).

---

## 📊 Example Outputs

- `Zeu`: Euphotic Depth (0–30 m)
- `PAR_bed`: Light at bed (E m⁻² d⁻¹)
- `SSI`: Seagrass Suitability Index (0–1)

![](../outputs/example_physics_maps.png)

---

## 💬 Interpretation
High SSI zones cluster in shallow regions (−3 to −7 m) where attenuation is low.
This provides a physics-only baseline before introducing machine-learning
refinements.
