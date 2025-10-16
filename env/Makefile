# ============================================================
# LiBuTS — Reproducible Pipeline Makefile
# ============================================================

PYTHON = python

.PHONY: all preprocess physics ml uncertainty optimize app clean

all: preprocess physics ml uncertainty optimize app

preprocess:
	@echo "🔹 Step 1: Fetching inputs..."
	$(PYTHON) notebooks/01_inputs_retrieval.py

physics:
	@echo "🔹 Step 2: Physics SSI..."
	$(PYTHON) notebooks/02_physics_suitability.py

ml:
	@echo "🔹 Step 3: ML + SHAP..."
	$(PYTHON) notebooks/03_ml_rf_shap.py

uncertainty:
	@echo "🔹 Step 4: Uncertainty enrichment..."
	$(PYTHON) notebooks/04_uncertainty_enrichment.py

optimize:
	@echo "🔹 Step 5: NSGA-II optimization..."
	$(PYTHON) notebooks/05_restoration_planner.py

app:
	@echo "🌊 Launching dashboard..."
	cd app && $(PYTHON) dashboard.py

clean:
	@echo "🧹 Cleaning outputs..."
	rm -rf outputs/*
