'''
main.py
'''
import click
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE

# XGBoost (if installed)
from xgboost import XGBClassifier
import yaml
from src.preprocessing_act import *
from src.training import *

def build_stacking():
    '''
    Returns a StackingClassifier with Random Forest, SVM, and XGBoost as base estimators and Logistic Regression as the final estimator.
    Options for base estimators: 
    '''
    return StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()),
            ('svm', SVC(probability=True)), # Must keep probability=True to allow stacking!
            ('xgb', XGBClassifier())
        ],
        final_estimator=LogisticRegression(),
        passthrough=True,
        n_jobs=-1,
        cv=3
    )


def build_voting():
    return VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()),
            ('svm', SVC(probability=True)),
            ('xgb', XGBClassifier())
        ],
        voting='soft'
    )


def build_rfe():
    return RFE(
        estimator=RandomForestClassifier(n_jobs=1),
    )


MODEL_REGISTRY = {
    # Classical
    "lda": LinearDiscriminantAnalysis,
    "qda": QuadraticDiscriminantAnalysis,
    "knn": KNeighborsClassifier,
    "logistic": LogisticRegression,
    "svm": SVC,

    # Trees & Ensembles
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "ada_boost": AdaBoostClassifier,
    "bagging": BaggingClassifier,

    # Boosting (external)
    "xgboost": XGBClassifier,

    # Neural Nets
    "mlp": MLPClassifier,

    # Meta-models (factory functions)
    "stacking": build_stacking,
    "voting": build_voting,
    "rfe": build_rfe,

    # Clustering
    "kmeans": lambda: SupervisedClusteringWrapper(model_name='kmeans'),
    "gmm": lambda: SupervisedClusteringWrapper(model_name='gmm'),
    
    # Deep Learning 
    # "cnn_2d": CNNWrapper,
    # "df_cnn": DFCNNWrapper # (You would build this similarly, splitting the inputs to 2 and 2)
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
    
    #2-Preprocessing
    if config.get("is_clean_data", False) == True:
        data = load_clean_data(
            filepath_train=config["data_path"],
            filepath_test=config["test_data_path"],
            target_col=config["target_col"]
        )
    else:
        data = preprocess_data(
            filepath_train=config["data_path"],
            filepath_test=config["test_data_path"],
            target_col=config["target_col"],
            outlier_method=config.get("outlier_method", "none"),  # "none" | "iqr" | "lof"
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

    if config["model_name"] == "rfe":
        print("\n--- RFE FEATURE EXTRACTION ---")
        survivor_mask = trainer.best_model.support_
        
        # Filter the train and test DataFrames
        X_train_clean = data["X_train_full_final"].loc[:, survivor_mask]
        X_test_clean = data["X_test"].loc[:, survivor_mask]
        
        # Re-attach the target column (y) to the training data
        train_clean_final = X_train_clean.copy()
        train_clean_final[config["target_col"]] = data["y_train_full"]
        
        train_clean_final.to_csv("data/processed/train_rfe_clean.csv", index=False)
        X_test_clean.to_csv("data/processed/test_rfe_clean.csv", index=False)
        
        print(f"SUCCESS: Extracted {sum(survivor_mask)} features.")
        print("Clean datasets saved as 'train_rfe_clean.csv' and 'test_rfe_clean.csv'!")
        print(f"Winning columns: {list(X_train_clean.columns)}")


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



"""
Y en el YAML añade la línea:
    outlier_method: lof    # o "iqr" o "none"

Ejemplo de YAML completo:
--------------------------
model_name: svm

data_path: data/raw/train_nt.csv
test_data_path: data/raw/test_nolabels_nt.csv
target_col: position

is_clean_data: false
outlier_method: lof       # <-- nuevo parámetro

param_grid:
  C: [1, 10, 100]
  kernel: [rbf]
  gamma: [scale, auto]

cv_folds: 5
scoring: f1_weighted
"""
