# ==============================================================
# LiBuTS — Vignette 06
# Dashboard Runtime Validation (Non-blocking)
# ==============================================================

import os
import importlib.util

def test_dashboard_exists():
    """Check if the dashboard file is present."""
    path = "app/dashboard.py"
    assert os.path.exists(path), "Dashboard script missing."

def test_dashboard_importable():
    """Verify that dashboard.py can be imported successfully."""
    spec = importlib.util.spec_from_file_location("dashboard", "app/dashboard.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise AssertionError(f"Dashboard failed to import: {e}")

    # Check for a dashboard object
    assert hasattr(module, "dashboard"), "No dashboard object found in dashboard.py"
    dash_obj = module.dashboard

    # Minimal smoke test
    assert "LiBuTS" in str(dash_obj.title), "Dashboard title mismatch"
