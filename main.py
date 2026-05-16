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

# # XGBoost / LightGBM (if installed)
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
import yaml
from src.training import *

def build_stacking():
    '''
    Returns a StackingClassifier with Random Forest, SVM, and XGBoost as base estimators and Logistic Regression as the final estimator.
    Options for base estimators: 
    '''
    return StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)),
            ('svm', SVC(probability=True, random_state=42)), # Must keep probability=True to allow stacking!
            # ('xgb', XGBClassifier(random_state=42))
        ],
        final_estimator=LogisticRegression(random_state=42),
        passthrough=True,
        n_jobs=-1,
        cv=3
    )


def build_voting():
    return VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)),
            ('svm', SVC(probability=True, random_state=42)),
            # ('xgb', XGBClassifier(random_state=42))
        ],
        voting='soft',
        n_jobs=-1
    )


def build_rfe(n_features=None):
    return RFE(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        n_features_to_select=n_features,  # None = select half, or specify number
        step=1
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
    # "xgboost": XGBClassifier,
    # "xgboost_gpu": lambda **params: XGBClassifier(
    #     tree_method="gpu_hist",
    #     predictor="gpu_predictor",
    #     gpu_id=0,
    #     **params
    # ),
    # "lightgbm": lambda: LGBMClassifier(
    #     objective="multiclass",
    #     random_state=42,
    #     verbose=-1,
    #     n_jobs=1,
    # ),

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


def load_processed_data(train_path, val_path, test_path, y_train_path, y_val_path):
    # Use semicolon separator to match how data is saved in datasets.py
    X_train = pd.read_csv(train_path, sep=',')
    X_val   = pd.read_csv(val_path, sep=',')
    X_test  = pd.read_csv(test_path, sep=',')

    y_train = pd.read_csv(y_train_path, sep=',')
    y_val   = pd.read_csv(y_val_path, sep=',')
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_train_full_final": pd.concat([X_train, X_val], axis=0, ignore_index=True),
        "y_train_full": pd.concat([y_train, y_val], axis=0, ignore_index=True),
        "X_test": X_test,
    }



def run_training(config_path: str) -> None:


    # Load config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print(f"Loaded config from {config_path}")
    print(f"Model: {config['model_name']}")

    # Load processed CSVs
    data = load_processed_data(
        train_path=config["train_path"],
        val_path=config["val_path"],
        test_path=config["test_path"],
        y_train_path=config["y_train_path"],
        y_val_path=config["y_val_path"]
    )

    # Build model
    model = get_model(config["model_name"])

    trainer = MLTrainer(
        model=model,
        param_grid=config["param_grid"],
        cv_folds=config["cv_folds"],
        scoring=config["scoring"],
        data=config["train_path"]
    )

    # Fit + search
    trainer.fit_and_search(data["X_train"], data["y_train"])

    #evaluate
    trainer.evaluate(data["X_val"], data["y_val"])

    trainer.save()

    # Refit on full data AFTER evaluation (metrics already logged)
    trainer.refit_full(data["X_train_full_final"], data["y_train_full"])

    #5-Testing
    trainer.save_submission(X_test=data["X_test"], test_ids=data["test_ids"])

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
