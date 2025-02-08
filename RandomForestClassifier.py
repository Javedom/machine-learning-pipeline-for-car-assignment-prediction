import os
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# Additional imports for encoding detection and Bayesian Optimization
# -------------------------------------------------------------------------
import chardet
import optuna

# -------------------------------------------------------------------------
# Scikit-Learn Imports
# -------------------------------------------------------------------------
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------------------------------------------------------
# 1) Data Loading with Fallback (.csv -> .xlsx)
# -----------------------------------------------------------------------------
def load_data(base_name):
    """
    Attempts to load data from either base_name.csv or base_name.xlsx.
    Priority:
      1) Check if 'base_name.csv' exists; if so, load it with chardet-based encoding detection.
      2) Else, check if 'base_name.xlsx' exists; if so, load it as Excel.
      3) If neither exists, raise FileNotFoundError.
    
    Returns a tuple: (df, file_format, encoding)
      - df: the loaded DataFrame
      - file_format: 'csv' or 'xlsx'
      - encoding: encoding for CSV, or None if XLSX
    """
    csv_path = f"{base_name}.csv"
    xlsx_path = f"{base_name}.xlsx"
    
    if os.path.exists(csv_path):
        with open(csv_path, 'rb') as f:
            rawdata = f.read(10000)
            result = chardet.detect(rawdata)
            guessed_encoding = result['encoding']
        
        print(f"Detected encoding for {csv_path}: {guessed_encoding}")
        df = pd.read_csv(csv_path, encoding=guessed_encoding)
        return df, 'csv', guessed_encoding
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
        return df, 'xlsx', None
    else:
        raise FileNotFoundError(
            f"Neither '{csv_path}' nor '{xlsx_path}' found in {os.getcwd()}."
        )

# -----------------------------------------------------------------------------
# 2) Data Preprocessing
# -----------------------------------------------------------------------------
def _safe_transform_schedule(encoder, schedule_series):
    """
    Safely transform schedule values using an already-fitted LabelEncoder.
    Any unseen values become NaN (so we can drop them).
    """
    known_classes = set(encoder.classes_)

    def encode_if_known(val):
        if val in known_classes:
            # Encode the single value (returns array, so take [0])
            return encoder.transform([val])[0]
        else:
            return np.nan

    return schedule_series.apply(encode_if_known)

def preprocess_data(data1, data2):
    """
    Prepares the datasets for model training and prediction.
    
    - Fits LabelEncoders on 'Schedule' and 'Passenger' using historical data (data2).
    - Adds encoded columns to data2 and transforms 'Schedule' in data1.
    - For data1, any unseen schedule values become NaN and are dropped.
    - Leaves 'Passenger' in data1 untransformed (due to potential new passengers).
    """
    encoder_schedule = LabelEncoder()
    encoder_passenger = LabelEncoder()
    
    # Fit encoders on historical data (data2)
    data2['Schedule_Encoded'] = encoder_schedule.fit_transform(data2['Schedule'])
    data2['Passenger_Encoded'] = encoder_passenger.fit_transform(data2['Passenger'])
    
    # Safely transform data1's Schedule (unseen => NaN)
    data1['Schedule_Encoded'] = _safe_transform_schedule(encoder_schedule, data1['Schedule'])
    # Drop rows with NaN schedules
    initial_len = len(data1)
    data1.dropna(subset=['Schedule_Encoded'], inplace=True)
    dropped = initial_len - len(data1)
    if dropped > 0:
        print(f"⚠️  Dropped {dropped} rows from data1 due to unseen schedule labels.")
    
    return data1, data2, encoder_schedule, encoder_passenger

# -----------------------------------------------------------------------------
# 3) Cross-Validation (Optional Step)
# -----------------------------------------------------------------------------
def cross_validate_model(data, n_splits=5):
    """
    Performs K-Fold cross-validation on the dataset using a RandomForestClassifier
    with class_weight='balanced'.
    Returns the array of accuracy scores and prints the mean.
    """
    X = data[['Passenger_Encoded', 'Schedule_Encoded']]
    y = data['Car']
    
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    print(f"K-Fold CV accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f}")
    return cv_scores

# -----------------------------------------------------------------------------
# 4) Hyperparameter Tuning (Grid Search)
# -----------------------------------------------------------------------------
def grid_search_tuning(X, y):
    """
    Performs a Grid Search for hyperparameter tuning on a RandomForestClassifier.
    Returns the best model found.
    """
    model = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 20, 40],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting Grid Search...")
    grid_search.fit(X, y)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_:.2f}")
    
    best_model = grid_search.best_estimator_
    return best_model

# -----------------------------------------------------------------------------
# (NEW) 4b) Hyperparameter Tuning with Optuna (Bayesian Optimization)
# -----------------------------------------------------------------------------
def tune_rf_with_optuna(X, y, n_trials=30):
    """
    Uses Optuna to perform Bayesian Optimization on a RandomForestClassifier.
    Tries 'n_trials' different hyperparameter combinations, guided by past results.
    Returns the best model found.
    """
    def objective(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        max_depth = trial.suggest_categorical("max_depth", [None, 10, 20, 40])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
        
        # Create the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',
            random_state=42
        )
        
        # Cross-validation for scoring
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        return scores.mean()  # maximize accuracy
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n[Optuna] Number of finished trials: {len(study.trials)}")
    print(f"[Optuna] Best trial parameters: {study.best_trial.params}")
    print(f"[Optuna] Best trial CV accuracy: {study.best_trial.value:.4f}")
    
    # Build final model using the best parameters
    best_params = study.best_params
    best_model = RandomForestClassifier(
        **best_params,
        class_weight='balanced',
        random_state=42
    )
    best_model.fit(X, y)
    
    return best_model

# -----------------------------------------------------------------------------
# 5) Train a Model
# -----------------------------------------------------------------------------
def train_model(data2, model=None):
    """
    Trains a (default) Random Forest model or a provided model using historical data (data2).
    
    Features: Passenger_Encoded, Schedule_Encoded
    Target: Car
    """
    X = data2[['Passenger_Encoded', 'Schedule_Encoded']]
    y = data2['Car']
    
    # Basic train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # If no model is provided, use a default RandomForest
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"✅ Model Training Complete!")
    print(f"Training Accuracy: {train_acc:.2f}")
    print(f"Test Accuracy:     {test_acc:.2f}")
    
    return model, X_test, y_test

# -----------------------------------------------------------------------------
# 6) Additional Metrics & Confusion Matrix
# -----------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Prints precision, recall, F1 for each class and overall (weighted).
    Also plots and returns a confusion matrix.
    """
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Weighted Precision: {precision:.2f}")
    print(f"Weighted Recall:    {recall:.2f}")
    print(f"Weighted F1 Score:  {f1:.2f}")

    # Union of both sets of labels to avoid mismatch
    all_labels = np.union1d(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    print("\nConfusion Matrix:")
    print(cm)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(cmap='Blues')
    disp.ax_.set_title("Confusion Matrix")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

# -----------------------------------------------------------------------------
# 7) Predict Car Numbers (Day 2)
# -----------------------------------------------------------------------------
def predict_car_numbers(data1, model, encoder_passenger):
    """
    For each record in data1:
      - If the passenger exists in the historical data (encoder_passenger.classes_),
        transform the passenger and use the model to predict the car.
      - Otherwise, assign NaN for Car_Predicted.
    """
    predictions = []
    for idx, row in data1.iterrows():
        passenger = row['Passenger']
        if passenger in encoder_passenger.classes_:
            passenger_encoded = encoder_passenger.transform([passenger])[0]
            schedule_encoded = row['Schedule_Encoded']
            X_new = pd.DataFrame([[passenger_encoded, schedule_encoded]],
                                 columns=['Passenger_Encoded', 'Schedule_Encoded'])
            pred = model.predict(X_new)[0]
            predictions.append(pred)
        else:
            predictions.append(np.nan)
    
    data1['Car_Predicted'] = predictions
    return data1

# -----------------------------------------------------------------------------
# 8) Main Execution: Putting it all together
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load Data (base names only; function tries CSV then XLSX)
    data1, data1_format, data1_encoding = load_data("data1_complex")
    data2, data2_format, data2_encoding = load_data("data2_complex")
    
    # 2. Preprocess Data (with safe schedule transform)
    data1, data2, encoder_schedule, encoder_passenger = preprocess_data(data1, data2)
    
    # 3. (Optional) Baseline Cross-Validation
    print("=== Cross-Validation on Historical Data (Baseline RF) ===")
    cross_validate_model(data2, n_splits=5)
    
    # 4. Prepare X, y for Tuning (from data2)
    X_all = data2[['Passenger_Encoded', 'Schedule_Encoded']]
    y_all = data2['Car']
    
    # 5a. Grid Search (Optional)
    print("\n=== Hyperparameter Tuning with Grid Search ===")
    best_model_grid = grid_search_tuning(X_all, y_all)
    
    # 5b. Bayesian Optimization (Optuna)
    print("\n=== Hyperparameter Tuning with Optuna (Bayesian Optimization) ===")
    best_model_optuna = tune_rf_with_optuna(X_all, y_all, n_trials=60)
    
    # 6. Train Final Model (choose which best model you want to proceed with)
    print("\n=== Training Final Model with Best Hyperparameters (Optuna) ===")
    final_model, X_test, y_test = train_model(data2, model=best_model_optuna)
    
    # 7. Evaluate
    print("\n=== Model Evaluation on Test Set ===")
    eval_results = evaluate_model(final_model, X_test, y_test)
    
    # 8. Predict Car Assignments for Day 2
    updated_data1 = predict_car_numbers(data1, final_model, encoder_passenger)
    print("\n🚗 Updated Data Set 1 - Assigned Car Numbers (Using Tuned RF):")
    print(updated_data1)
    
    # 9. Save Output
    if data1_format == 'csv':
        save_encoding = data1_encoding if data1_encoding else 'utf-8'
        output_file = "updated_data1_ml_results.csv"
        updated_data1.to_csv(output_file, index=False, encoding=save_encoding)
        print(f"\n✅ Results saved to: {output_file} (encoding='{save_encoding}')")
    else:
        output_file = "updated_data1_ml_results.xlsx"
        updated_data1.to_excel(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file} (Excel format)")
