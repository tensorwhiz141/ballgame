"""
Machine learning model training for EEG blink detection.
Trains multiple models and selects the best one for deployment.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
from data_processing import EEGDataProcessor

class BlinkDetectionTrainer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
    def initialize_models(self):
        """Initialize different ML models to try."""
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=42,
                class_weight='balanced',
                probability=True
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            )
        }
        
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate their performance."""
        print("\nTraining models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, pos_label='blink')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            
            # Update best model based on F1 score (better for imbalanced data)
            if f1 > self.best_score:
                self.best_score = f1
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} with F1-score: {self.best_score:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning on the best model."""
        print(f"\nPerforming hyperparameter tuning on {self.best_model_name}...")
        
        if self.best_model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            
        elif self.best_model_name == 'GradientBoosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            base_model = GradientBoostingClassifier(random_state=42)
            
        elif self.best_model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
            base_model = SVC(random_state=42, class_weight='balanced', probability=True)
            
        elif self.best_model_name == 'LogisticRegression':
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            base_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
            
        else:  # KNN
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            base_model = KNeighborsClassifier()
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def evaluate_final_model(self, X_test, y_test):
        """Evaluate the final tuned model."""
        print(f"\nFinal evaluation of {self.best_model_name}...")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label='blink')
        
        print(f"Final Accuracy: {accuracy:.4f}")
        print(f"Final F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            print("\nTop 10 Feature Importances:")
            feature_names = ['mean', 'std', 'min', 'max', 'median', 'range', 'skewness', 
                           'kurtosis', 'rms', 'variance', 'slope', 'energy', 'dominant_freq', 'spectral_energy']
            importances = self.best_model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for feature, importance in feature_importance[:10]:
                print(f"  {feature}: {importance:.4f}")
        
        return accuracy, f1
    
    def save_model(self, scaler):
        """Save the trained model and scaler."""
        model_path = os.path.join(self.models_dir, 'blink_detector.joblib')
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        
        # Save model
        joblib.dump(self.best_model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save scaler
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'best_score': self.best_score,
            'feature_names': ['mean', 'std', 'min', 'max', 'median', 'range', 'skewness', 
                            'kurtosis', 'rms', 'variance', 'slope', 'energy', 'dominant_freq', 'spectral_energy']
        }
        
        metadata_path = os.path.join(self.models_dir, 'model_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        print(f"Metadata saved to: {metadata_path}")
        
        return model_path, scaler_path
    
    def train_complete_pipeline(self, window_size=10, tune_hyperparameters=True):
        """Run the complete training pipeline."""
        print("Starting complete model training pipeline...")
        
        # Load and process data
        processor = EEGDataProcessor(data_dir="data")
        X_train, X_test, y_train, y_test, scaler = processor.process_all(window_size=window_size)
        
        # Initialize models
        self.initialize_models()
        
        # Train all models
        results = self.train_models(X_train, y_train, X_test, y_test)
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            self.hyperparameter_tuning(X_train, y_train)
        
        # Final evaluation
        accuracy, f1 = self.evaluate_final_model(X_test, y_test)
        
        # Save the best model
        model_path, scaler_path = self.save_model(scaler)
        
        print(f"\nTraining pipeline completed!")
        print(f"Best model: {self.best_model_name}")
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Final F1-score: {f1:.4f}")
        
        return self.best_model, scaler, accuracy, f1

if __name__ == "__main__":
    # Train the model
    trainer = BlinkDetectionTrainer()
    model, scaler, accuracy, f1 = trainer.train_complete_pipeline()
