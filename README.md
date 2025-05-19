# Bank Customer Churn Prediction

## Overview

This project aims to predict bank customer churn using machine learning models. It includes data preprocessing, model training, evaluation, and logging with MLflow. The dataset used is `Churn_Modelling.csv`, which contains customer information and whether they exited the bank (churned).

The project implements three models: Random Forest, Logistic Regression, and Support Vector Classifier (SVC). It uses MLflow to track experiments, log metrics, and store artifacts like ROC curves and confusion matrices.

## Features

- **Data Preprocessing**: Handles class imbalance using resampling, encodes categorical variables, and scales numerical features.
- **Model Training**: Trains Random Forest, Logistic Regression, and SVC models.
- **Evaluation**: Calculates accuracy, F1-score, AUC, and plots ROC curves and confusion matrices.
- **MLflow Integration**: Logs models, metrics, and artifacts to MLflow for experiment tracking.

## Installation

### Prerequisites

- Python 3.8+
- MLflow server running at `http://localhost:5000` (default tracking URI)

### Steps

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd MLOPS-COURSE-LABS
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes packages like `pandas`, `scikit-learn`, `mlflow`, `matplotlib`, `colorama`, and `joblib`.

4. Ensure the MLflow server is running:

   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

## Usage

1. Place the `Churn_Modelling.csv` dataset in the `dataset/` directory.

2. Run the main script to start the experiment:

   ```bash
   python main.py
   ```

3. The script will:

   - Preprocess the data.
   - Train Random Forest, Logistic Regression, and SVC models.
   - Log metrics, parameters, and artifacts to MLflow.
   - Save outputs (e.g., ROC curves, confusion matrices, and transformers) in the `output/` directory.

4. View the experiment results in the MLflow UI:

   - Open `http://localhost:5000` in your browser.
   - Navigate to the `Churn_Prediction_exp` experiment to see the logged models, metrics, and artifacts.

## File Structure

```
MLOPS-COURSE-LABS/
├── dataset/
│   └── Churn_Modelling.csv      # Input dataset
├── output/
│   ├── ROC_curve.png            # ROC curve plot
│   ├── confusion_matrix.png     # Confusion matrix plot
│   └── transformer.joblib       # Saved ColumnTransformer
├── src/
│   ├── data_preprocessing.py    # Preprocessing functions
│   ├── evaluation.py            # Evaluation metrics and plotting
│   ├── mlflow_logging.py        # MLflow logging utilities
│   └── models.py                # Model training functions
├── main.py                      # Main script to run the experiment
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
```

## Project Workflow

1. **Data Preprocessing** (`data_preprocessing.py`):
   - Balances the dataset using `resample` to address class imbalance.
   - Applies `OneHotEncoder` to categorical features and `StandardScaler` to numerical features.
   - Splits the data into training and test sets.
2. **Model Training** (`models.py`):
   - Trains three models: Random Forest, Logistic Regression, and SVC with predefined hyperparameters.
3. **Evaluation** (`evaluation.py`):
   - Computes accuracy, F1-score, and AUC.
   - Generates and saves ROC curves and confusion matrices.
4. **MLflow Logging** (`mlflow_logging.py`):
   - Sets up an MLflow experiment and logs models, metrics, and artifacts.
5. **Main Script** (`main.py`):
   - Orchestrates the entire workflow, from data loading to logging results.

## Dependencies

The project relies on the following Python libraries (listed in `requirements.txt`):

- `pandas`
- `scikit-learn`
- `mlflow`
- `matplotlib`
- `colorama`
- `joblib`

## Output

- **Metrics**: Accuracy, F1-score, AUC, and score for each model.
- **Plots**: ROC curves and confusion matrices saved in the `output/` directory.
- **Artifacts**: Logged to MLflow, including the trained models, transformers, and plots.
- **Transformer**: The `ColumnTransformer` used for preprocessing is saved as `transformer.joblib`.

## Notes

- Ensure the MLflow server is running before executing the script.
- The dataset `Churn_Modelling.csv` must be present in the `dataset/` directory.
- Adjust the `tracking_uri` in `setup_mlflow_experiment` if your MLflow server is hosted elsewhere.