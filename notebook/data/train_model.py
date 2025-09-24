import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. DATA LOADING ---
print("Step 1: Loading raw data...")
try:
    # This path assumes your script is in the root of 'emg_ball_game_project/'
    file_path = 'data/emg_sensor3.csv'
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"\nERROR: The file '{file_path}' was not found.")
    exit()

# --- 2. ROLLING WINDOW FEATURE ENGINEERING ---
print("\nStep 2: Engineering features using a rolling window approach...")
window_size = 50
original_features = ['ch1_voltage', 'ch2_voltage', 'ch3_voltage']
new_features_df = df.copy()

for col in original_features:
    new_features_df[f'{col}_roll_mean'] = new_features_df[col].rolling(window=window_size).mean()
    new_features_df[f'{col}_roll_std'] = new_features_df[col].rolling(window=window_size).std()
    new_features_df[f'{col}_roll_max'] = new_features_df[col].rolling(window=window_size).max()
    new_features_df[f'{col}_roll_min'] = new_features_df[col].rolling(window=window_size).min()

new_features_df.dropna(inplace=True)
print(f"Rolling features created. New dataset shape: {new_features_df.shape}\n")

# --- 3. PREPARE FINAL DATA FOR MODELING ---
X = new_features_df.drop('label', axis=1)
y = new_features_df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Final training set shape: {X_train.shape}")
print(f"Final testing set shape: {X_test.shape}\n")

# --- 4. DATA SCALING ---
print("Step 4: Scaling the new features...")
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling complete.\n")

# --- 5. TRAIN THE FINAL MODEL ---
print("Step 5: Training CatBoost on the enriched dataset...")
X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

cat_model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=5,
    verbose=500,
    early_stopping_rounds=100,
    random_state=42
)
cat_model.fit(
    X_train_new, y_train_new,
    eval_set=(X_val, y_val)
)

# --- 6. FINAL EVALUATION AND SAVING ---
predictions = cat_model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, predictions)
print(f"\nFinal Model Accuracy: {final_accuracy:.4f}\n")

# Save the model and scaler to the 'models' subfolder
print("Saving model and scaler...")
joblib.dump(cat_model, 'models/catboost_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("All files saved successfully to the 'models' folder!")