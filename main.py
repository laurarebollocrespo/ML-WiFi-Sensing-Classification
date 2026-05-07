'''
main.py
'''
import click
import yaml
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.preprocessing import apply_pca, load_test_data, load_train_data, scale_data
from src.training import MLTrainer

MODEL_REGISTRY = {
    "lda": LinearDiscriminantAnalysis,
    "qda": QuadraticDiscriminantAnalysis,
    "knn": KNeighborsClassifier,
    "logistic": LogisticRegression,
    "svm": SVC,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "mlp": MLPClassifier,
}


def get_model(model_name: str) -> BaseEstimator:
    """Builds a fresh estimator instance from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class()


def run_training(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print(f"Loaded config from {config_path}")
    print(f"Model: {config['model_name']}")
    print(config)

    X, y = load_train_data(
        filepath=config["data_path"],
        target_col=config["target_col"],
    )
    X_test_raw, test_ids = load_test_data(config["test_data_path"])

    test_size = float(config.get("test_size", 0.2))
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=123,
    )

    # Fit scaler on train split only; same transformation for val, full train, and test.
    X_train_scaled, fitted_scaler = scale_data(X_train_raw, scaler=None)
    X_val_scaled, _ = scale_data(X_val_raw, scaler=fitted_scaler)
    X_full_scaled, _ = scale_data(X, scaler=fitted_scaler)
    X_test_scaled, _ = scale_data(X_test_raw, scaler=fitted_scaler)

    if config.get("use_pca", False):
        X_train_final, fitted_pca = apply_pca(X_train_scaled, pca=None)
        X_val_final, _ = apply_pca(X_val_scaled, pca=fitted_pca)
        X_full_final, _ = apply_pca(X_full_scaled, pca=fitted_pca)
        X_test_final, _ = apply_pca(X_test_scaled, pca=fitted_pca)
    else:
        X_train_final = X_train_scaled
        X_val_final = X_val_scaled
        X_full_final = X_full_scaled
        X_test_final = X_test_scaled

    model = get_model(config["model_name"])

    trainer = MLTrainer(
        model=model,
        param_grid=config["param_grid"],
        cv_folds=config["cv_folds"],
        scoring=config["scoring"],
        data=config["data_path"],
    )

    trainer.fit_and_search(X_train_final, y_train)
    trainer.evaluate(X_val_final, y_val)
    trainer.refit_on_full_data(X_full_final, y)
    trainer.note_submission_refit()

    trainer.save()
    trainer.save_submission(X_test=X_test_final, test_ids=test_ids)

    print("Training completed. Artifacts stored under outputs/.")


@click.group()
def main():
    """Command-line tool for training ML models."""
    pass


@main.command("train")
@click.argument("config_path", type=click.Path(exists=True))
def train_cmd(config_path):
    """Train a model using a YAML configuration file."""
    run_training(config_path)


if __name__ == "__main__":
    main()
