import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import wandb
from loguru import logger
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score
)


class SklearnTrainer:
    def __init__(self, model, X_train, y_train, config, project="wifi-sensing-classification", run_name="sklearn-trainer"):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.project = project
        self.run_name = run_name

    def run(self):
        if self.config["use_wandb"]:
            wandb.init(project=self.project, name=self.run_name, config=self.config)

        # 1. Setup Hyperparameter Search
        param_grid = self.config.get("param_grid", {})
        cv_folds = self.config.get("cv_folds", 5)
        scoring = self.config.get("scoring", "f1_macro")

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )

        logger.info(f"Starting GridSearch CV with {cv_folds} folds...")
        grid_search.fit(self.X_train, self.y_train)
        
        # Update self.model to the best performing one
        self.model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

        print("best score:", best_score)
        print("best params:", best_params)


        if self.config["use_wandb"]:
            wandb.finish()

    def save_model(self, path):
        logger.info(f"Saving model to {path}")
        joblib.dump(self.model, path)


all_models = {
    "random_forest": RandomForestClassifier(random_state=42),
    "rbf_svm": SVC(kernel='rbf', random_state=42),
}


if __name__ == "__main__":
    import yaml


    df = pd.read_csv("data/raw/train_nt.csv", sep=";")
    df = df.drop(columns=["seq_ctrl"])
    X, y = df.drop(columns=["position"]), df["position"]

    model_name = "rbf_svm"

    config = {
        "use_wandb": True,
        # "param_grid": yaml.safe_load(open(f"config/{model_name}.yaml"))["param_grid"]
    }
    model = all_models.get(model_name)


    trainer = SklearnTrainer(model, X, y, config)
    trainer.run()





