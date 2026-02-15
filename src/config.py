# config.py

# Data paths
DATASET_PATH = "/data/ACME-HappinessSurvey2020.csv"
SAVED_MODEL_DIR = "models"
SAVED_MODEL_FILENAME = "happiness_prediction_model.joblib"
SAVED_MODEL_PATH = f"{SAVED_MODEL_DIR}/{SAVED_MODEL_FILENAME}"

# Column names
TARGET_COLUMN = "Y"
ALL_FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6"]
REDUCED_FEATURES = ["X1", "X2", "X3", "X5"] # Used for the best performing RF model

# Training/Test Split
TEST_SIZE_INITIAL = 0.30 # For initial logistic regression
TEST_SIZE_FINAL = 0.20 # For reduced feature models
RANDOM_STATE_INITIAL = 1 # For initial splits and models
RANDOM_STATE_FINAL = 0 # For the best performing RF model

# Model Hyperparameters
# Decision Tree original tuning grid
DT_PARAM_GRID = {
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(2, 11)),
    'min_samples_leaf': list(range(5, 21))
}

# Best performing Random Forest hyperparameters (for reduced features)
RF_BEST_PARAMS = {
    'n_estimators': 30,
    'criterion': 'entropy',
    'max_depth': 14,
    'min_samples_split': 8,
    'random_state': RANDOM_STATE_FINAL
}

# Thresholds
LOGISTIC_REGRESSION_OPTIMAL_THRESHOLD = 0.4581833349224797 # From initial Logistic Regression
LOGISTIC_REGRESSION_REDUCED_OPTIMAL_THRESHOLD = 0.2407839808014219 # From reduced Logistic Regression

# Evaluation metric target
TARGET_ACCURACY = 0.73
