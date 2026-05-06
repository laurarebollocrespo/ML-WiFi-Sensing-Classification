"""Model registry for sklearn estimators used by the trainer."""

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_REGISTRY: dict[str, type] = {
    "lda": LinearDiscriminantAnalysis,
    "qda": QuadraticDiscriminantAnalysis,
    "gaussian_nb": GaussianNB,
    "logistic_regression": LogisticRegression,
    "softmax_regression": LogisticRegression,
    "knn": KNeighborsClassifier,
    "svm_rbf": SVC,
    "svm_linear": SVC,
    "svm_poly": SVC,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "adaboost": AdaBoostClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def get_model(name: str, **kwargs):
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)
