from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, f1_score
import joblib

class MLTrainer:
    '''
    A trainer class that encapsulates the training, hyperparameter search, evaluation, and saving logic for a scikit-learn model.
    '''
    model: BaseEstimator # any scikit-learn model
    param_grid: dict[str, list]
    cv_folds: int
    scoring: str
    best_model: BaseEstimator | None

    def __init__(self, model: BaseEstimator, param_grid: dict[str, list], cv_folds: int, scoring: str) -> None:
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
        self.best_model = None

    def fit_and_search(self, X_train, y_train) -> None:
        '''
        Performs hyperparameter search using GridSearchCV and fits the best model on the training data.
        Args:
            X_train: Training features
            y_train: Training labels
        '''
        print(f"Searching for best hyperparameters using {self.cv_folds}-fold CV...")
        search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=-1 
        )

        search.fit(X_train, y_train)
        
        self.best_model = search.best_estimator_
        print(f"Best CV Score: {search.best_score_:.4f}")
        print(f"Best Parameters Found: {search.best_params_}")

    def evaluate(self, X_val, y_val) -> float:
        '''
        Evaluates the best model on the validation set and prints the F1-score.
        Args:
            X_val: Validation features
            y_val: Validation labels
        Returns:
            The F1-score on the validation set.
        ''' 

        print("Evaluating best model on unseen validation data...")
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit_and_search() first.")
        y_pred = self.best_model.predict(X_val)

        print("Classification Report:")
        print(classification_report(y_val, y_pred))


    #     # TODO: hacer un dataframe o algo como en classe
    #     def compute_metrics(y_real,y_pred):
    # # By default it will compute the binary recall of class 1, we can specify which class do we want by using this parameter 
    #         recall_class_1 =recall_score(y_real,y_pred, pos_label=1)
    #         f1_class_1 =f1_score(y_real,y_pred, pos_label=1)
    #         accuracy = accuracy_score(y_real,y_pred)
    #         f1_macro =f1_score(y_real,y_pred, average='macro')
    #         precison_macro =precision_score(y_real,y_pred,  average='macro')
    #         recall_macro =recall_score(y_real,y_pred,  average='macro')
    #         return [recall_class_1, f1_class_1, accuracy,f1_macro,precison_macro,recall_macro ]
    #     results_heart = pd.DataFrame(index=[], columns= ['Recall class 1', 'F1 class 1','Accuracy', 'F1 Macro', 'Precision Macro', 'Recall Macro'])

    #     results_heart.loc['LDA', :] = compute_metrics(y_val, y_val_pred_lda)
    #     results_heart


        score = f1_score(y_val, y_pred, average='macro')
        print(f"Final Validation F1-Score: {score:.4f}")
        return score

    def save(self, filepath) -> None:
        '''
        Saves the best model to disk using joblib.
        Args:
            filepath: The path where the model should be saved
        '''
        print(f"Saving model to {filepath}...")
        joblib.dump(self.best_model, filepath)