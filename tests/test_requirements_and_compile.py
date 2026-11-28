"""Checks the new requirements file and that the notebook export compiles."""

import py_compile
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements.txt"
SCRIPT = ROOT / "Model" / "fraud_detection_in_transactions.py"

# These pins were copied from the notebook's Colab install logs, except
# kagglehub==1.0.3, which is the PyPI release checked on 2026-11-27.
EXPECTED_PINS = {
    "pandas==2.2.2",
    "numpy==2.0.2",
    "scikit-learn==1.6.1",
    "xgboost==3.0.4",
    "imbalanced-learn==0.14.0",
    "joblib==1.5.2",
    "plotly==5.24.1",
    "kagglehub==1.0.3",
}

# Map imported top-level names to the requirement package that provides them.
IMPORT_TO_PACKAGE = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "imblearn": "imbalanced-learn",
    "joblib": "joblib",
    "plotly": "plotly",
    "kagglehub": "kagglehub",
}


class RequirementsTests(unittest.TestCase):
    def test_requirements_has_logged_pins(self):
        self.assertTrue(REQ.is_file())
        pins = {
            line.strip()
            for line in REQ.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(pins, EXPECTED_PINS)

    def test_requirements_cover_script_imports(self):
        ## IPython is a notebook-runtime import and is not pinned here.
        text = SCRIPT.read_text()
        names = set()
        for line in text.splitlines():
            s = line.strip()
            m = re.match(r"^(?:import|from)\s+([A-Za-z0-9_]+)", s)
            if m:
                names.add(m.group(1))
        packages = {IMPORT_TO_PACKAGE[n] for n in names if n in IMPORT_TO_PACKAGE}
        pinned = {pin.split("==")[0] for pin in EXPECTED_PINS}
        self.assertEqual(packages, pinned)
        self.assertIn("IPython", names)

    def test_script_compiles(self):
        ## Do not execute the script: it would download the Kaggle dataset.
        ## Compile only, to catch leftover notebook syntax.
        py_compile.compile(str(SCRIPT), doraise=True)


if __name__ == "__main__":
    unittest.main()
