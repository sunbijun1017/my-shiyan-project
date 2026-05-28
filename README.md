# MFFNet — Multi-branch Feature Fusion Network for Time-Series Classification

A deep learning experiment that uses a three-branch architecture (MFFNet) to classify longitudinal patient examination data. The model performs 5-fold cross-validation and outputs per-fold metrics, ROC curves, confusion matrices, and SHAP feature-importance plots.

---

## Project Structure

```
├── MFFNet.py           # Main experiment script
├── config.py           # Your local column configuration (not tracked by Git)
├── config_template.py  # Template for config.py — copy and edit this
├── requirements.txt    # Python dependencies
└── data.xlsx           # Your input data file (not included in the repo)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your config file

```bash
# Windows
copy config_template.py config.py

# Linux / macOS
cp config_template.py config.py
```

Open `config.py` and fill in your actual column names:

```python
TARGET = 'your_target_column_name'   # binary label column (0 / 1)

FEATURES = [
    'feature_col_1',
    'feature_col_2',
    # ... all feature columns
]
```

> **Note:** `config.py` is listed in `.gitignore` and will never be committed.  
> Always reference `config_template.py` to understand the expected structure.

### 4. Prepare your data file

Place your Excel file in the project root and name it **`data.xlsx`**.

The file must contain:

| Column | Description |
|--------|-------------|
| `ID`   | Patient identifier (used to group visits) |
| *(feature columns)* | All columns listed in `FEATURES` inside `config.py` |
| *(target column)*   | The column named in `TARGET` inside `config.py`; binary (0 / 1) |

Each row represents a single visit. Patients with fewer than 3 visits are excluded automatically.

### 5. Run the experiment

```bash
python MFFNet.py
```

---

## Model Architecture

MFFNet extracts complementary information through three parallel branches before feature fusion

## Output

For each experiment run the script produces:

- Per-fold console metrics: Accuracy, Sensitivity, Specificity, PPV, NPV, F1, ROC-AUC, PR-AUC, Brier Score
- Mean ± std summary table
- 5-fold confusion matrix subplot figure
- Mean ROC curve with ±1 std band
- SHAP feature importance bar chart
- SHAP beeswarm summary plot

---

## Dependencies

| Package | Minimum Version |
|---------|----------------|
| numpy | 1.23.0 |
| pandas | 1.5.0 |
| matplotlib | 3.6.0 |
| shap | 0.42.0 |
| scikit-learn | 1.1.0 |
| tensorflow | 2.10.0 |
| openpyxl | 3.0.10 |

