# Feature Selection for MRI-based Alzheimer’s Disease Prediction

This repository contains Python implementations of two feature selection methodologies applied to MRI-derived features for predicting cognitive stages in Alzheimer’s Disease:

1. **Correlation-based Greedy Neighbourhood Feature Selection (CGN-FS)**
2. **Region and Clustering-based Heuristic Feature Selection (RCH-FSC)**

---

## Repository Structure

```
/repo-root
│
├─ CGN_FS                   # Contains both Selection and Evaluation of Correlation-based Greedy Neighbourhood Feature Selection
├─ RCH_FSC                  # Contains both Selection and Evaluation of Region and Clustering-based Heuristic Feature Selection
├─ data/                    # Place your MRI dataset CSVs here
└─ requirements.txt         # Python dependencies
```

---

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

---

## Running the Code

### 1. CGN-FS (Correlation-based Greedy Neighbourhood Feature Selection)

* File: `Selection_CGN_FS.py`
* Removes highly correlated features based on a threshold (threshold could be modified).
* Steps:

  1. Load dataset CSV.
  2. Drop irrelevant columns (`age, research_category, gender`).
  3. Compute correlation matrix and absolute values.
  4. Sort features by correlation count and mark correlated neighbors for removal.
* Outputs: Excel files containing features marked as **Keep** or **Remove**.

```bash
python Selection_CGN_FS.py
```

---

### 2. RCH-FSC (Region and Clustering-based Heuristic Feature Selection)

* File: `Selection_RCH_FSC.py`
* Uses clustering (K-Medoids) on PCoA-transformed distance matrix for selecting representative features (medoids).
* Steps:

  1. Load dataset CSV.
  2. Encode categorical labels and drop unnecessary columns.
  3. Compute correlation-based distance matrix.
  4. Standardize features and perform Principal Coordinate Analysis (PCoA).
  5. Apply K-Medoids clustering and compute silhouette scores for different numbers of clusters.
  6. Select features corresponding to cluster medoids.
* Outputs: Excel/CSV files listing selected features per cluster.

```bash
python Selection_RCH_FSC.py
```

---

# 3. 📊 Dataset Requirements

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

## Notes on Reproducibility

* Random seeds are set for reproducibility.
* Ensure the dataset columns match the scripts (especially for label columns).
* Excel and CSV outputs provide feature selection results for further analysis or downstream modeling.
