import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent variables from train and test array")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "SVM": SVC()
            }

            params = {
                "SVM": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                }
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=lambda x: model_report[x]["accuracy"])
            best_model = model_report[best_model_name]["best_model"]
            best_accuracy = model_report[best_model_name]["accuracy"]

            logging.info(f"Best Model: {best_model_name} with Accuracy: {best_accuracy}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model, best_accuracy

        except Exception as e:
            raise CustomException(e, sys)
