'''
training.py
'''
import json
from datetime import datetime
import os

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import joblib

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

    def fit_and_search(self, X_train, y_train) -> None:
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

    def refit_full(self, X_full, y_full) -> None:
        """Refit best model on train+val combined before submission."""

        print("Refitting best model on full training data (train + val)...")
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit_and_search() first.")
        
        self.best_model.set_params(**self.best_params)
        self.best_model.fit(X_full, y_full)

    def compute_metrics(self, y_real: list, y_pred: list) -> list[float]:
        # By default it will compute the binary recall of class 1, we can specify which class do we want by using this parameter 
        #recall_class_1 =recall_score(y_real,y_pred, pos_label=1)
        #f1_class_1 =f1_score(y_real,y_pred, pos_label=1)
        accuracy = accuracy_score(y_real,y_pred)
        f1_macro =f1_score(y_real,y_pred, average='macro')
        precision_macro =precision_score(y_real,y_pred,  average='macro')
        recall_macro =recall_score(y_real,y_pred,  average='macro')
        classif_report = classification_report(y_real, y_pred)
        return [accuracy, f1_macro, precision_macro, recall_macro, classif_report]

    def evaluate(self, X_val, y_val) -> None:
        '''
        Evaluates the best model on the validation data and defines the experiment.
        ''' 
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit_and_search() first.")
        
        print("Evaluating best model on unseen validation data...")

        y_pred = self.best_model.predict(X_val)

        print("hola")
        self.define_experiment(self.compute_metrics(y_val, y_pred))

    def define_experiment(self, metrics: list[float]) -> None:
        
        accuracy, f1_macro, precision_macro, recall_macro, classif_report = metrics
        self.experiment_results = {
            "model_name": type(self.best_model).__name__,

            "best_hyperparameters": self.best_params,

            "cross_validation": {
                "scoring_metric": self.scoring,
                "cv_folds": self.cv_folds,
                "best_cv_score": self.best_cv_score
            },

            "validation_metrics": {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro
            },

            "classification_report": classif_report
        }

    
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
        Saves both the model (.pkl) and the experiment (.json) using the exact same base name.
        """
        models_dir, experiments_dir = "outputs/models", "outputs/experiments"

        base_name = self.generate_base_filename()

        #Save the Model
        model_filepath = f"{models_dir}/{base_name}.pkl"
        print(f"Saving model to {model_filepath}...")
        joblib.dump(self.best_model, model_filepath)

        #Save the Experiment JSON
        exp_filepath = f"{experiments_dir}/{base_name}.json"
        print(f"Saving experiment to {exp_filepath}...")
        with open(exp_filepath, "w") as f:
            json.dump(self.experiment_results, f, indent=4)

    def save_submission(self, X_test: pd.DataFrame) -> None:
        """
        Generates a Kaggle submission CSV using the same base filename 
        as the model and experiment JSON.
        """
        submissions_dir: str = "outputs/submissions"
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
        print(f"Submission saved to: {output_path}")

