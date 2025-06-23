import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.exception import CustomException

# Save object using pickle
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# Load object using pickle
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# Evaluate classification models
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params[model_name]

            gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)

            report[model_name] = {
                "best_model": best_model,
                "accuracy": test_accuracy,
                "best_params": gs.best_params_
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
