'''
training.py
'''
import json
from datetime import datetime
import os

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.parallel import Parallel, delayed
import joblib
import torch
import torch
import wandb
import matplotlib.pyplot as plt

class MLTrainer:
    '''
    A trainer class that encapsulates the training, hyperparameter search, evaluation, and saving logic for a scikit-learn model.
    '''
    model: BaseEstimator # any scikit-learn model
    param_grid: dict[str, list]
    cv_folds: int
    scoring: str
    data: str

    best_model: BaseEstimator | None
    best_params: dict
    best_cv_score: float

    experiment_results: dict

    def __init__(self, model: BaseEstimator, param_grid: dict[str, list], cv_folds: int, scoring: str, data: str) -> None:
        '''
        Initializes the trainer with the model, hyperparameter grid, CV settings, and scoring metric.
        Args:
            model: A scikit-learn model
            param_grid: A dictionary specifying the hyperparameters to search over
            cv_folds: Number of cross-validation folds
            scoring: The metric to optimize during hyperparameter search (e.g., 'f1_macro')
        '''
        self.model = model
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.data = data

        self.best_model = None
        self.best_params = {}
        self.best_cv_score = 0.0

        self.experiment_results = {}

    def fit_and_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        '''
        Performs hyperparameter search using GridSearchCV and fits the best model on the training data.
        Args:
            X_train: Training features
            y_train: Training labels
        '''
        print(f"Searching for best hyperparameters using {self.cv_folds}-fold Stratified CV...")
        
        cv_strategy = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=1)
        
        search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=cv_strategy,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=3
        )

        search.fit(X_train, y_train)
        
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_cv_score = search.best_score_

        print(f"Best CV Score: {self.best_cv_score:.4f}")
        print(f"Best Parameters: {self.best_params}")

    def refit_full(self, X_full: pd.DataFrame, y_full: pd.Series) -> None:
        """Refit best model on train+val combined before submission."""

        print("Refitting best model on full training data (train + val)...")
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit_and_search() first.")
        
        self.best_model.set_params(**self.best_params)
        self.best_model.fit(X_full, y_full)

    def compute_metrics(self, y_real: pd.Series, y_pred: pd.Series) -> list[float]:
        # By default it will compute the binary recall of class 1, we can specify which class do we want by using this parameter 
        #recall_class_1 =recall_score(y_real,y_pred, pos_label=1)
        #f1_class_1 =f1_score(y_real,y_pred, pos_label=1)
        accuracy = accuracy_score(y_real,y_pred)
        f1_macro =f1_score(y_real,y_pred, average='macro')
        precision_macro =precision_score(y_real,y_pred,  average='macro')
        recall_macro =recall_score(y_real,y_pred,  average='macro')
        classif_report = classification_report(y_real, y_pred)
        return [accuracy, f1_macro, precision_macro, recall_macro, classif_report]

    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        '''
        Evaluates the best model on the validation data and defines the experiment.
        ''' 
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit_and_search() first.")
        
        print("Evaluating best model on unseen validation data...")

        y_pred = self.best_model.predict(X_val)

        self.define_experiment(self.compute_metrics(y_val, y_pred), y_val, y_pred)


    def define_experiment(self, metrics: list[float], y_val: pd.Series, y_pred: pd.Series) -> None:
        
        accuracy, f1_macro, precision_macro, recall_macro, classif_report = metrics
        
        self.experiment_results = {
            "model_name": type(self.best_model).__name__,
            "best_hyperparameters": self.best_params,
            "validation_metrics": {"accuracy": accuracy, "f1_macro": f1_macro}
        }

        print("Initializing Weights & Biases run...")
        try:        
            wandb.init(
                project="AA1",
                entity="laura-rebollo-crespo-universitat-polit-cnica-de-catalunya",
                name=f"{type(self.best_model).__name__}_f1-{f1_macro:.4f}",
                config={
                    "model_name": type(self.best_model).__name__,
                    "cv_folds": self.cv_folds,
                    "scoring_metric": self.scoring,
                    **self.best_params
                }
            )
        except Exception as e:
            print(f"Warning: W&B initialization failed ({e}). Continuing without W&B logging.")

        fig, ax = plt.subplots(figsize=(10, 8))
        
        disp = ConfusionMatrixDisplay.from_predictions(
            y_val, 
            y_pred, 
            ax=ax, 
            cmap='viridis',
            colorbar=False
        )
        plt.title(f'Confusion Matrix: {type(self.best_model).__name__}', fontsize=16, pad=15)
        plt.tight_layout()

        # LOG IT TO W&B 
        wandb.log({
            "val_accuracy": accuracy,
            "val_f1_macro": f1_macro,
            "val_precision_macro": precision_macro,
            "val_recall_macro": recall_macro,
            "classification_report": wandb.Html(f"<pre>{classif_report}</pre>"),
            
            "scikit_learn_matrix": wandb.Image(fig) 
        })
        
        plt.close(fig)


    def generate_base_filename(self) -> str:

        if not self.experiment_results:
            raise ValueError("No experiment results found. Run evaluate() first.")

        model_name = self.experiment_results["model_name"]
        params = self.experiment_results["best_hyperparameters"]
        metrics = self.experiment_results["validation_metrics"]
        
        param_string = "_".join([f"{k}-{str(v)}" for k, v in params.items()])
        if len(param_string) > 50:
            param_string = "complex_ensemble_params"

        metric_string = f"f1-{metrics['f1_macro']:.4f}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_used = os.path.basename(self.data).replace(".csv", "")

        return f"{model_name}__{param_string}__{metric_string}__{data_used}__{timestamp}"
     
    def save(self) -> None:
        """
        Saves the best model locally and uploads it to W&B as an artifact, then cleans up the local file.
        """
        models_dir = "outputs/models"
        os.makedirs(models_dir, exist_ok=True)
        base_name = self.generate_base_filename()

        # Save the Model locally first so W&B can grab it
        model_filepath = f"{models_dir}/{base_name}.pkl"
        print(f"Saving model locally to {model_filepath}...")
        joblib.dump(self.best_model, model_filepath)

        # --- 3. UPLOAD MODEL TO W&B ---
        print("Uploading model to W&B Cloud...")
        model_artifact = wandb.Artifact(
            name=f"{type(self.best_model).__name__}_model",
            type="model",
            description="Trained scikit-learn model"
        )
        model_artifact.add_file(model_filepath)
        wandb.log_artifact(model_artifact)

        if os.path.exists(model_filepath):
            os.remove(model_filepath)
            print("Deleted:", model_filepath)
        else:
            print("File not found:", model_filepath)


    def save_submission(self, X_test: pd.DataFrame) -> None:
        """
        Generates Kaggle predictions and uploads the CSV to W&B.
        """
        submissions_dir = "outputs/submissions"
        os.makedirs(submissions_dir, exist_ok=True)
        
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")

        base_name = self.generate_base_filename()
        output_path = f"{submissions_dir}/{base_name}.csv"

        print("Generating Kaggle predictions...")
        y_pred = self.best_model.predict(X_test)

        submission = pd.DataFrame({
            "ID": range(len(y_pred)),
            "POSITION": y_pred.astype(int)
        })

        submission.to_csv(output_path, index=False)
        print(f"Submission saved locally to: {output_path}")

        # UPLOAD CSV TO W&B ---
        print("Uploading Kaggle submission to W&B Cloud...")
        csv_artifact = wandb.Artifact(
            name=f"{type(self.best_model).__name__}_submission",
            type="predictions"
        )
        csv_artifact.add_file(output_path)
        wandb.log_artifact(csv_artifact)

        # --- 5. CLOSE THE W&B RUN ---
        wandb.finish()


class SupervisedClusteringWrapper(BaseEstimator, ClassifierMixin):
    '''
    
    '''
    model_name: str
    n_clusters: int
    model: BaseEstimator
    label_map: dict[int, int] # Maps unsupervised cluster IDs to actual class labels based on majority vote

    def __init__(self, model_name='kmeans', n_clusters=10):
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.label_map = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SupervisedClusteringWrapper':
        '''
        
        '''
        if self.model_name == 'kmeans':
            self.model = make_pipeline(
                PCA(n_components=20), 
                KMeans(n_clusters=self.n_clusters, max_iter=300),
            )
        elif self.model_name == 'gmm':
            self.model = make_pipeline(
                PCA(n_components=20),
                GaussianMixture(n_components=self.n_clusters, max_iter=300)
            )
            
        clusters = self.model.fit_predict(X)
        
        # Map unsupervised clusters to actual labels (y) based on majority vote
        for cluster_id in np.unique(clusters):
            true_labels = y[clusters == cluster_id]
            if len(true_labels) > 0:
                self.label_map[cluster_id] = true_labels.mode()[0]
            else:
                self.label_map[cluster_id] = 0
        return self
        
    def predict(self, X: pd.DataFrame):
        '''
        
        '''
        clusters = self.model.predict(X)
        # Map them back to correct labels
        return np.array([self.label_map[c] for c in clusters])
    