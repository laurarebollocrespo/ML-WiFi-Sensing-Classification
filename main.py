'''
main.py
'''
import click
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import yaml
from src.preprocessing import *
from src.training import MLTrainer
from sklearn.preprocessing import RobustScaler


MODEL_REGISTRY = {
    "lda": LinearDiscriminantAnalysis,
    "qda": QuadraticDiscriminantAnalysis,
    "knn": KNeighborsClassifier,
    "logistic": LogisticRegression,
    "svm": SVC,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "mlp": MLPClassifier
}   

def get_model(model_name: str) -> BaseEstimator:
    """Retrieves and initializes the model from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class()


def run_training(config_path: str) -> None:

    #1-Loading
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print(f"Loaded config from {config_path}")
    print(f"Model: {config['model_name']}")
    print(config)
    
    #2-Preprocessing
    data = preprocess_data(
        filepath_train=config["data_path"],
        filepath_test=config["test_data_path"],
        target_col=config["target_col"]
    )
    
    #3-Modeling
    model = get_model(config["model_name"])

    trainer = MLTrainer(
        model=model,
        param_grid=config["param_grid"],
        cv_folds=config["cv_folds"],
        scoring=config["scoring"],
        data=config["data_path"]
    )

    trainer.fit_and_search(data["X_train"], data["y_train"])

    #4-Evaluating
    trainer.evaluate(data["X_val"], data["y_val"])

    trainer.save()

    # Refit on full data AFTER evaluation (metrics already logged)
    trainer.refit_full(data["X_train_full_final"], data["y_train_full"])

    #5-Testing
    trainer.save_submission(X_test=data["X_test"])

    print(f"Training completed. Model saved")  


@click.group()
def main():
    """Command-line tool for training ML models."""
    pass

@main.command("train")
@click.argument("config_path", type=click.Path(exists=True))

def train_cmd(config_path):
    """
    Train a model using a YAML configuration file.
    """
    run_training(config_path)


if __name__ == "__main__":
    main()
