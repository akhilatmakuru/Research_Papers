# XAI for MRI-based Alzheimer’s Disease Prediction

This repository contains Python implementations of explainable AI (XAI) techniques applied to MRI-derived features for predicting cognitive stages in Alzheimer’s Disease. The repository includes feature importance analysis using **SHAP** and sensitivity analysis using **Sobol, Morris, and FAST** methods.

## Repository Structure

```
/repo-root
│
├─ SHAP.py          # SHAP-based feature importance
├─ SOBOL.py         # Sobol sensitivity analysis
├─ MORRIS.py        # Morris sensitivity analysis
├─ FAST.py          # FAST sensitivity analysis
└─ requirements.txt          # Python dependencies
```

## Setup

1. Clone the repository:

```bash
git clone <your_repo_link>
cd <repo-folder>
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your MRI dataset CSVs in the `data/` folder.

The code expects the MRI dataset:

```
MRI_rois_alldiseases.csv
```

> *The dataset used in this thesis (MRI ROI data) is licensed and cannot be redistributed.
> Users must obtain access independently through ADNI/AIBL/IXI and place the CSV file in the following directory:*
>
> ```
> /data/MRI_rois_alldiseases.csv
> ```

## Running the Code

### 1. SHAP Analysis

* File: `shap_analysis.py`
* Trains a neural network classifier and computes SHAP values for all features.

```bash
python shap_analysis.py
```

* Outputs CSV and Excel files with SHAP-based feature importance.

### 2. Sensitivity Analysis (SALib)

* Files: `sobol_analysis.py`, `morris_analysis.py`, `fast_analysis.py`
* Uses pre-trained or simple neural models to perform sensitivity analysis with different methods.
* Key difference is in the `.analyze()` function used:

  * **Sobol:** `sobol.analyze()`
  * **Morris:** `morris.analyze()`
  * **FAST:** `fast.analyze()`

```bash
python sobol_analysis.py
python morris_analysis.py
python fast_analysis.py
```

* Outputs CSV/Excel files containing feature sensitivity indices.

## Notes on Reproducibility

* Random seeds are set for reproducibility where applicable.
* Ensure the input datasets have the same preprocessing and column order as used in the scripts.
* Early stopping is used for neural networks to stabilize training.
* All outputs (feature importance, sensitivity indices) are saved as CSV and Excel for easy analysis.

## Requirements

See `requirements.txt` for the full list of Python packages needed (e.g., `tensorflow`, `keras`, `pandas`, `numpy`, `shap`, `SALib`, `scikit-learn`, `matplotlib`, `seaborn`).
