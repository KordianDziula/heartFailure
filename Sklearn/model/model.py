from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score, classification_report

class Model():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def train(self):
        grid = GridSearchCV(
            estimator = LGBMClassifier(objective = "binary", random_state = 1),
            param_grid = {
                "boosting_type": ["gbdt", "dart"],
                "num_leaves": [4, 5, 10, 15, 20],
                "max_depth": [2, 4, 6, 8, 10],
                "learning_rate": [0.15, 0.1, 0.05, 0.01],
                "n_estimators": [40, 45, 50, 60]
            },
            scoring = make_scorer(precision_score, zero_division = 0),
            cv = StratifiedKFold(n_splits = 3),
            n_jobs = -1,
            verbose = 0
        )

        grid.fit(self.X_train, self.y_train)
        self.best_estimator = grid.best_estimator_
        self.classification_report = classification_report(grid.best_estimator_.predict(self.X_test), self.y_test)

        return self

        