'''
main.py
'''
import click
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import yaml
from src.preprocessing import load_and_split_data
from src.training import MLTrainer

MODEL_REGISTRY = {
    "lda": LinearDiscriminantAnalysis,
    "random_forest": RandomForestClassifier,
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

    #2-Splitting
    X_train, X_val, y_train, y_val = load_and_split_data(
        filepath=config["data_path"],
        target_col=config["target_col"],
        test_size=config["test_size"]
    )
    #3-Preprocessing
    #TODO

    #
    X_train_scaled, fitted_scaler = scale_data(X_train, scaler=None)
    X_val_scaled, _ = scale_data(X_val, scaler=fitted_scaler)

    #4-Modeling
    model = get_model(config["model_name"])

    trainer = MLTrainer(
        model=model,
        param_grid=config["param_grid"],
        cv_folds=config["cv_folds"],
        scoring=config["scoring"]
    )

    trainer.fit_and_search(X_train, y_train)
    trainer.evaluate(X_val, y_val)

    trainer.save()

    #Testing
    X_test_raw = load_test_data(config["test_data_path"])
    X_test_scaled, _ = scale_data(X_test_raw, scaler=fitted_scaler)

    trainer.save_submission(X_test=X_test_scaled)

    print(f"Training completed. Model saved to {config['save_path']}")
    


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
