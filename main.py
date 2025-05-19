from src.data_preprocessing import preprocess
from src.models import train_random_forest, train_logistic_regression, train_SVC
from src.mlflow_logging import log_model_with_mlflow, setup_mlflow_experiment
from pathlib import Path
import logging
from colorama import Fore, Style
import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=f"{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - {Fore.BLUE}%(levelname)s{Style.RESET_ALL} - %(message)s"
    )

def main():
    setup_logging()
    logging.info("Starting Prediction Experiment...")
    experiment_id = setup_mlflow_experiment("Churn_Prediction_exp")
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "dataset/Churn_Modelling.csv"
    output_dir = BASE_DIR / "output"

    df = pd.read_csv(data_path)
    
    transf,X_train, X_test, y_train, y_test = preprocess(df)
    
    rf_model = train_random_forest(X_train, y_train)
    log_model_with_mlflow(rf_model, transf, X_test, y_test, "RandomForestClassifier", experiment_id, output_dir)
    
    lr_model = train_logistic_regression(X_train, y_train)
    log_model_with_mlflow(lr_model, transf, X_test, y_test, "LogisticRegression", experiment_id, output_dir)
    
    dt_model = train_SVC(X_train, y_train)
    log_model_with_mlflow(dt_model, transf, X_test, y_test, "SVC_classifier", experiment_id, output_dir)
    
    logging.info("Experiment completed successfully!")

if __name__ == "__main__":
    main()