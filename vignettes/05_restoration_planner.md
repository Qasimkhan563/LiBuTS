# Vignette 5 — Restoration Planner via NSGA-II Optimization

---

## 🎯 Objective
To identify optimal restoration cells balancing:
- **CO₂ potential** ↑
- **Uncertainty** ↓
- **ALAN risk** ↓ (Artificial Light at Night)

---

## ⚙️ Optimization Formulation

Each pixel _i_ has three objectives \(F₁,F₂,F₃\):

\[
\begin{aligned}
F_1 &= -CO2_i \quad \text{(maximize CO₂ gain)} \\
F_2 &= Uncertainty_i \\
F_3 &= ALAN_i
\end{aligned}
\]

Optimization uses **NSGA-II** (pop = 100, 50 generations)
implemented via `pymoo`.

---

## 📈 Outputs

- Pareto front (Trade-offs CO₂ vs Uncertainty)
- GeoPackage of optimal sites
- Summary statistics

![](../outputs/pareto_front.png)
![](../outputs/restoration_map.png)

---

## 📦 Files

outputs/pareto_front.csv
outputs/restoration_sites.gpkg
outputs/restoration_summary.csv


---

## 💬 Interpretation
Optimal sites cluster where SSI is high and ALAN risk is low, mainly along
sheltered inner basins (−4 to −8 m).  
These trade-offs help managers prioritize restoration for maximal carbon
benefit with low model risk.
