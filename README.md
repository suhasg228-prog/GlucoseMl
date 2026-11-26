# GlucoseML – Diabetes Classification Project

A Streamlit-based educational machine-learning project with:
- Dataset overview and missing-value checks
- Exploratory data analysis and descriptive statistics
- Feature distribution graph
- Correlation heatmap
- Logistic Regression and Random Forest models
- Accuracy, precision, recall, F1-score
- Confusion matrix, classification report and ROC curve
- Interactive sample prediction form

> **Important:** This is an educational demonstration, not a medical diagnostic tool. Do not use predictions for healthcare decisions.

## Requirements
- Python 3.9 or newer
- Internet connection for the default dataset download, or a local CSV upload

## Setup in VS Code

1. Extract the ZIP file.
2. Open the `GlucoseML` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment (recommended):

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

5. Install packages:
```bash
python -m pip install -r requirements.txt
```

6. Run the app:
```bash
python -m streamlit run app.py
```

7. Streamlit will show a local URL, usually `http://localhost:8501`.

## Dataset
By default, the app loads the Pima Indians Diabetes dataset from:
https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

You can also upload your own CSV from the sidebar. The target column should be named `Outcome`, `Diabetes`, or `target` (case-insensitive variants are supported), and should contain two classes such as 0 and 1. Feature columns must be numeric.

## Troubleshooting

### `No module named streamlit`
Run this in the same terminal/environment used to start the app:
```bash
python -m pip install streamlit
```

### Dataset does not load
Check your internet connection or upload a compatible CSV using the sidebar.

### Wrong Python environment
Run:
```bash
python -m pip show streamlit
python --version
```
