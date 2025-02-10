
🚀 MACHINE LEARNING PIPELINE FOR CAR ASSIGNMENT PREDICTION
============================================================

This project provides a complete machine learning pipeline for predicting car assignments based on passenger and schedule data. The pipeline includes data loading, preprocessing, cross-validation, hyperparameter tuning (Grid Search and Optuna), and model evaluation.

------------------------------------------------------------
📂 TABLE OF CONTENTS
------------------------------------------------------------
1. Features
2. Installation
3. Usage
4. Pipeline Steps
5. Example
6. License

------------------------------------------------------------
✨ FEATURES
------------------------------------------------------------
- Dynamic Data Loading: Supports both .csv and .xlsx files with automatic encoding detection.
- Safe Label Encoding: Handles unseen values without errors.
- Cross-Validation: Uses K-Fold cross-validation to assess model performance.
- Hyperparameter Tuning: Implements Grid Search and Optuna for model optimization.
- Model Evaluation: Detailed metrics and confusion matrix visualization.
- Prediction Module: Predicts car assignments for new data while handling unseen values.

------------------------------------------------------------
📦 INSTALLATION
------------------------------------------------------------
Clone this repository and install the necessary dependencies.

```bash
git clone https://github.com/your-username/car-assignment-pipeline.git
cd car-assignment-pipeline
pip install -r requirements.txt
```

Required Python packages:
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Chardet
- Optuna
- Matplotlib

------------------------------------------------------------
🔧 USAGE
------------------------------------------------------------
1. **Prepare Your Data**
   Ensure you have two datasets:
   - `data1`: Dataset for which car predictions will be made.
   - `data2`: Historical dataset for training.

   Columns required:
   - `Passenger`: Unique passenger identifier.
   - `Schedule`: Schedule information (example 9:00).
   - `day_of_week`: Day of week as integer (1-7) (in `data2` only).
   - `Car`: Assigned car number (in `data2` only).

2. **Run the Script**
   Execute the main script:
   ```bash
   python main.py
   ```

3. **Check the Output**
   The output file with predicted car assignments will be saved as:
   - `updated_data1_ml_results.csv` or `updated_data1_ml_results.xlsx`

------------------------------------------------------------
🛠 PIPELINE STEPS
------------------------------------------------------------
1. **Data Loading**
   - Detects and loads `.csv` or `.xlsx` files.
   - Uses `chardet` to determine the encoding for `.csv` files.

2. **Data Preprocessing**
   - Applies label encoding with safe transformation for unseen values.

3. **Cross-Validation**
   - Uses K-Fold cross-validation with `RandomForestClassifier`.

4. **Hyperparameter Tuning**
   - Grid Search for exhaustive search.
   - Optuna for Bayesian Optimization with faster, smarter tuning.

5. **Model Training**
   - Trains the best model found during hyperparameter tuning.

6. **Model Evaluation**
   - Calculates precision, recall, F1-score, and plots a confusion matrix.

7. **Prediction**
   - Predicts car assignments for new data, handling unseen passengers gracefully.

------------------------------------------------------------
🔍 EXAMPLE OUTPUT
------------------------------------------------------------
```
=== Cross-Validation on Historical Data (Baseline RF) ===
K-Fold CV accuracy scores: [0.85, 0.87, 0.84, 0.88, 0.86]
Mean CV accuracy: 0.86

=== Hyperparameter Tuning with Optuna (Bayesian Optimization) ===
[Optuna] Best trial parameters: {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 2}
[Optuna] Best trial CV accuracy: 0.89

✅ Model Training Complete!
Training Accuracy: 0.92
Test Accuracy: 0.88
```

------------------------------------------------------------
📜 LICENSE
------------------------------------------------------------
This project is licensed under the CC0 1.0 Universal license.

