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

    # #2-Splitting
    # X_train, X_val, y_train, y_val = load_and_split_data(
    #     filepath=config["data_path"],
    #     target_col=config["target_col"],
    #     test_size=config["test_size"]
    # )
    
    # X_test_raw = load_test_data(config["test_data_path"])

    # #3-Preprocessing
    # #Scaling

    # X_train_scaled, fitted_scaler = scale_data(X_train, scaler=None)
    # X_val_scaled, _ = scale_data(X_val, scaler=fitted_scaler)
    # X_test_scaled, _ = scale_data(X_test_raw, scaler=fitted_scaler)

    # # 3.5 Apply PCA (If specified in config)
    # if config.get("use_pca", False):
    #     X_train_final, fitted_pca = apply_pca(X_train_scaled, pca=None)
    #     X_val_final, _ = apply_pca(X_val_scaled, pca=fitted_pca)
    #     X_test_final, _ = apply_pca(X_test_scaled, pca=fitted_pca)
    # else:
    #     X_train_final, X_val_final, X_test_final = X_train_scaled, X_val_scaled, X_test_scaled
    

    # #4-Modeling
    # model = get_model(config["model_name"])

    # trainer = MLTrainer(
    #     model=model,
    #     param_grid=config["param_grid"],
    #     cv_folds=config["cv_folds"],
    #     scoring=config["scoring"],
    #     data=config["data_path"]
    # )

    # trainer.fit_and_search(X_train_final, y_train)
    # trainer.evaluate(X_val_final, y_val)

    # trainer.save()

    # #Testing
    # trainer.save_submission(X_test=X_test_final)

    # print(f"Training completed. Model saved")

    # # 1. Update your imports to use the new load_train_data function!

# ... inside run_training(config_path): ...

    # 1. Load FULL Training Data (No 80/20 split!)
    X_train, y_train = load_train_data(
        filepath=config["data_path"],
        target_col=config["target_col"]
    )
    
    X_test_raw = load_test_data(config["test_data_path"])

    # 3. Scale all data
    X_train_scaled, fitted_scaler = scale_data(X_train, scaler=None)
    X_test_scaled, _ = scale_data(X_test_raw, scaler=fitted_scaler)

    # 3.5 Apply PCA (If specified in config)
    if config.get("use_pca", False):
        X_train_final, fitted_pca = apply_pca(X_train_scaled, pca=None)
        X_test_final, _ = apply_pca(X_test_scaled, pca=fitted_pca)
    else:
        X_train_final, X_test_final = X_train_scaled, X_test_scaled

    # 4. Initialize Model & Trainer
    model = get_model(config["model_name"])

    trainer = MLTrainer(
        model=model,
        param_grid=config["param_grid"],
        cv_folds=config["cv_folds"],
        scoring=config["scoring"],
        data=config["data_path"]
    )

    # 5. Train & Search (This will run CV on 100% of the training data!)
    trainer.fit_and_search(X_train_final, y_train)

    # 6. Define Experiment based on CV results (Replaces trainer.evaluate)
    trainer.evaluate(X_val_final, y_val)

    # 7. Save Model, JSON, and Kaggle Submission
    trainer.save()
    trainer.save_submission(X_test=X_test_final)

    print("NEW : Training completed. Artifacts safely stored in outputs/!")

    
    


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
