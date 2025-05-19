import mlflow
import mlflow.data
import mlflow.models
import mlflow.sklearn
import joblib
from pathlib import Path
from .evaluation import eval_metrics, plot_roc_curve
import logging
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "http://localhost:5000") -> str:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    return exp.experiment_id

def log_model_with_mlflow(model, col_transf, X_test, y_test, model_name: str, exp_id: str, output_dir: Path):
    with mlflow.start_run(experiment_id=exp_id, run_name=model_name) as run:
        logging.info(f"Logging {model_name} to MLflow...")

        mlflow.set_tag("model", model_name)

        pred = model.predict(X_test)
        accuracy, f1, auc = eval_metrics(y_test, pred)
        plot_roc_curve(y_test, pred, output_dir)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "score": model.score(X_test, y_test),
            "Accuracy": accuracy,
            "f1-score": f1,
            "AUC": auc
        })

        mlflow.log_artifact(str(output_dir / "ROC_curve.png"), artifact_path="plots")

        conf_mat = confusion_matrix(y_test, pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )        
        conf_mat_disp.plot()
        # Log the image as an artifact in MLflow
        fig_path = "output/confusion_matrix.png"
        conf_mat_disp.figure_.savefig(fig_path)
        mlflow.log_artifact(fig_path, artifact_path="plots")

        pd_dataset = mlflow.data.from_pandas(X_test, name="Testing Dataset")
        mlflow.log_input(pd_dataset, context="Testing")

        joblib.dump(model, output_dir / f"{model_name}.pkl")
        # Save transformer locally
        joblib.dump(col_transf, output_dir / "transformer.pkl")
        # Log the transformer as an artifact
        mlflow.log_artifact(str(output_dir / "transformer.pkl"), artifact_path="transformer")

        signature = mlflow.models.infer_signature(X_test, y_test)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=X_test.iloc[[0]])