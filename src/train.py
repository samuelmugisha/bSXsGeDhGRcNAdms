# train.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import configurations
from .config import (
    DATASET_PATH, TARGET_COLUMN, ALL_FEATURES, REDUCED_FEATURES,
    TEST_SIZE_INITIAL, TEST_SIZE_FINAL, RANDOM_STATE_INITIAL, RANDOM_STATE_FINAL,
    DT_PARAM_GRID, RF_BEST_PARAMS, LOGISTIC_REGRESSION_OPTIMAL_THRESHOLD,
    LOGISTIC_REGRESSION_REDUCED_OPTIMAL_THRESHOLD, SAVED_MODEL_DIR, SAVED_MODEL_PATH
)

# =============================================================================
# Utility Functions (copied from notebook, for model training specific tasks)
# =============================================================================
def treat_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    Lower_Whisker = Q1 - 1.5 * IQR
    Upper_Whisker = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], Lower_Whisker, Upper_Whisker)
    return df

def treat_outliers_all(df, col_list):
    for c in col_list:
        df = treat_outliers(df, c)
    return df

def model_performance_classification_statsmodels(model, predictors, target, threshold=0.5):
    pred_temp = model.predict(predictors) > threshold
    pred = np.round(pred_temp)
    acc = accuracy_score(target, pred)
    recall = recall_score(target, pred)
    precision = precision_score(target, pred)
    f1 = f1_score(target, pred)
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )
    return df_perf

def model_performance_classification_sklearn(model, predictors, target):
    pred = model.predict(predictors)
    acc = accuracy_score(target, pred)
    recall = recall_score(target, pred)
    precision = precision_score(target, pred)
    f1 = f1_score(target, pred)
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )
    return df_perf

def confusion_matrix_statsmodels(model, predictors, target, threshold=0.5, title="Confusion Matrix"):
    y_pred = model.predict(predictors) > threshold
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())] for item in cm.flatten()]
    ).reshape(2, 2)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.show()

def confusion_matrix_sklearn(model, predictors, target, title="Confusion Matrix"):
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())] for item in cm.flatten()]
    ).reshape(2, 2)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.show()


def train_models():
    # Load Data
    pdata = pd.read_csv(DATASET_PATH)
    df = pdata.copy()

    # Outlier Treatment
    numerical_col = df.select_dtypes(include=np.number).columns.tolist()
    df = treat_outliers_all(df, numerical_col)

    # Prepare data for initial models (with 'const' for statsmodels)
    X = df.drop(TARGET_COLUMN, axis=1)
    Y = df[TARGET_COLUMN]
    X = sm.add_constant(X) # Add constant for statsmodels

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE_INITIAL, random_state=RANDOM_STATE_INITIAL
    )

    # =============================================================================
    # Logistic Regression (Initial)
    # =============================================================================
    print("\n==============================")
    print("Training Initial Logistic Regression")
    print("==============================")
    logit = sm.Logit(y_train, X_train.astype(float))
    lg = logit.fit(disp=False)
    print(lg.summary())

    y_train_pred_proba_lg = lg.predict(X_train)
    precision, recall, thresholds = precision_recall_curve(y_train, y_train_pred_proba_lg)
    f1_scores = np.divide((2 * precision * recall), (precision + recall), out=np.zeros_like(precision), where=(precision + recall) != 0)
    f1_scores = np.nan_to_num(f1_scores)
    optimal_idx_pr_curve = np.argmax(f1_scores)
    optimal_threshold_pr_curve = thresholds[optimal_idx_pr_curve]

    print("\nTraining performance (Initial LG):")
    lg_train_metrics = model_performance_classification_statsmodels(lg, X_train, y_train, threshold=optimal_threshold_pr_curve)
    print(lg_train_metrics)
    confusion_matrix_statsmodels(lg, X_train, y_train, threshold=optimal_threshold_pr_curve, title="Initial LG Training Confusion Matrix")

    print("\nTest set performance (Initial LG):")
    lg_test_metrics = model_performance_classification_statsmodels(lg, X_test, y_test, threshold=optimal_threshold_pr_curve)
    print(lg_test_metrics)
    confusion_matrix_statsmodels(lg, X_test, y_test, threshold=optimal_threshold_pr_curve, title="Initial LG Test Confusion Matrix")

    # =============================================================================
    # Feature Selection for Logistic Regression
    # =============================================================================
    selected_features_lg = X_train.columns.tolist()
    max_p_value = 1
    while len(selected_features_lg) > 0 and max_p_value > 0.05:
        X_train_aux = X_train[selected_features_lg]
        model = sm.Logit(y_train, X_train_aux.astype(float)).fit(disp=False)
        p_values = model.pvalues
        max_p_value = p_values.max()
        feature_with_p_max = p_values.idxmax()
        if max_p_value > 0.05:
            selected_features_lg.remove(feature_with_p_max)
        else:
            break
    print(f"\nSelected features for Logistic Regression after p-value elimination: {selected_features_lg}")

    # =============================================================================
    # Logistic Regression (Selected Features)
    # =============================================================================
    print("\n======================================")
    print("Training Logistic Regression (Selected Features)")
    print("======================================")
    X_train_lg_selected = X_train[selected_features_lg]
    X_test_lg_selected = X_test[selected_features_lg]
    logit_selected = sm.Logit(y_train, X_train_lg_selected.astype(float))
    lg_selected = logit_selected.fit(disp=False)
    print(lg_selected.summary())

    y_train_pred_proba_lg_selected = lg_selected.predict(X_train_lg_selected)
    precision_lg_selected, recall_lg_selected, thresholds_lg_selected = precision_recall_curve(y_train, y_train_pred_proba_lg_selected)
    f1_scores_lg_selected = np.divide((2 * precision_lg_selected * recall_lg_selected), (precision_lg_selected + recall_lg_selected), out=np.zeros_like(precision_lg_selected), where=(precision_lg_selected + recall_lg_selected) != 0)
    f1_scores_lg_selected = np.nan_to_num(f1_scores_lg_selected)
    optimal_idx_lg_selected = np.argmax(f1_scores_lg_selected)
    optimal_threshold_lg_selected = thresholds_lg_selected[optimal_idx_lg_selected]

    print("\nTraining performance (LG Selected):")
    lg_selected_train_metrics = model_performance_classification_statsmodels(lg_selected, X_train_lg_selected, y_train, threshold=optimal_threshold_lg_selected)
    print(lg_selected_train_metrics)
    confusion_matrix_statsmodels(lg_selected, X_train_lg_selected, y_train, threshold=optimal_threshold_lg_selected, title="LG Selected Training Confusion Matrix")

    print("\nTest set performance (LG Selected):")
    lg_selected_test_metrics = model_performance_classification_statsmodels(lg_selected, X_test_lg_selected, y_test, threshold=optimal_threshold_lg_selected)
    print(lg_selected_test_metrics)
    confusion_matrix_statsmodels(lg_selected, X_test_lg_selected, y_test, threshold=optimal_threshold_lg_selected, title="LG Selected Test Confusion Matrix")

    # Prepare data for sklearn models (no 'const' column)
    X_sk = df[ALL_FEATURES]
    y_sk = df[TARGET_COLUMN]
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
        X_sk, y_sk, test_size=TEST_SIZE_INITIAL, random_state=RANDOM_STATE_INITIAL
    )

    # =============================================================================
    # Decision Tree (Untuned)
    # =============================================================================
    print("\n==============================")
    print("Training Untuned Decision Tree")
    print("==============================")
    dtree = DecisionTreeClassifier(random_state=RANDOM_STATE_INITIAL)
    dtree.fit(X_train_sk, y_train_sk)

    print("\nTraining performance (Untuned DT):")
    dt_train_metrics = model_performance_classification_sklearn(dtree, X_train_sk, y_train_sk)
    print(dt_train_metrics)
    confusion_matrix_sklearn(dtree, X_train_sk, y_train_sk, title="Untuned DT Training Confusion Matrix")

    print("\nTest set performance (Untuned DT):")
    dt_test_metrics = model_performance_classification_sklearn(dtree, X_test_sk, y_test_sk)
    print(dt_test_metrics)
    confusion_matrix_sklearn(dtree, X_test_sk, y_test_sk, title="Untuned DT Test Confusion Matrix")

    # =============================================================================
    # Decision Tree (Tuned)
    # =============================================================================
    print("\n==============================")
    print("Tuning Decision Tree")
    print("==============================")
    dtree_classifier = DecisionTreeClassifier(random_state=RANDOM_STATE_INITIAL)
    grid_search = GridSearchCV(
        estimator=dtree_classifier,
        param_grid=DT_PARAM_GRID,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train_sk, y_train_sk)
    dtree_tuned = grid_search.best_estimator_
    print(f"Best estimator from GridSearchCV: {dtree_tuned}")

    print("\nTest set performance (Tuned DT):")
    tuned_dt_test_metrics = model_performance_classification_sklearn(dtree_tuned, X_test_sk, y_test_sk)
    print(tuned_dt_test_metrics)
    confusion_matrix_sklearn(dtree_tuned, X_test_sk, y_test_sk, title="Tuned DT Test Confusion Matrix")

    # =============================================================================
    # Random Forest (Initial)
    # =============================================================================
    print("\n==============================")
    print("Training Initial Random Forest")
    print("==============================")
    rf_classifier = RandomForestClassifier(random_state=RANDOM_STATE_INITIAL)
    rf_classifier.fit(X_train_sk, y_train_sk)

    print("\nTest set performance (Initial RF):")
    rf_test_metrics = model_performance_classification_sklearn(rf_classifier, X_test_sk, y_test_sk)
    print(rf_test_metrics)
    confusion_matrix_sklearn(rf_classifier, X_test_sk, y_test_sk, title="Initial RF Test Confusion Matrix")

    # =============================================================================
    # SVM (Initial)
    # =============================================================================
    print("\n==============================")
    print("Training Initial SVM")
    print("==============================")
    svm_classifier = SVC(random_state=RANDOM_STATE_INITIAL)
    svm_classifier.fit(X_train_sk, y_train_sk)

    print("\nTest set performance (Initial SVM):")
    svm_test_metrics = model_performance_classification_sklearn(svm_classifier, X_test_sk, y_test_sk)
    print(svm_test_metrics)
    confusion_matrix_sklearn(svm_classifier, X_test_sk, y_test_sk, title="Initial SVM Test Confusion Matrix")

    # =============================================================================
    # Naive Bayes (Initial)
    # =============================================================================
    print("\n==============================")
    print("Training Initial Naive Bayes")
    print("==============================")
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train_sk, y_train_sk)

    print("\nTest set performance (Initial Naive Bayes):")
    nb_test_metrics = model_performance_classification_sklearn(nb_classifier, X_test_sk, y_test_sk)
    print(nb_test_metrics)
    confusion_matrix_sklearn(nb_classifier, X_test_sk, y_test_sk, title="Initial Naive Bayes Test Confusion Matrix")

    # =============================================================================
    # Models with Reduced Features
    # =============================================================================
    print("\n================================")
    print("Training Models with Reduced Features")
    print("================================")
    X_reduced = df[REDUCED_FEATURES]
    y_reduced = df[TARGET_COLUMN]
    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
        X_reduced, y_reduced, test_size=TEST_SIZE_FINAL, random_state=RANDOM_STATE_FINAL
    )

    # Random Forest (Reduced Features - Best Model)
    print("\n-------------------------------------")
    print("Training Random Forest (Reduced Features)")
    print("-------------------------------------")
    rf_classifier_reduced = RandomForestClassifier(**RF_BEST_PARAMS)
    rf_classifier_reduced.fit(X_train_reduced, y_train_reduced)
    print("\nTest set performance (RF Reduced):")
    rf_reduced_metrics = model_performance_classification_sklearn(rf_classifier_reduced, X_test_reduced, y_test_reduced)
    print(rf_reduced_metrics)
    confusion_matrix_sklearn(rf_classifier_reduced, X_test_reduced, y_test_reduced, title="RF Reduced Test Confusion Matrix")

    # Decision Tree (Reduced Features)
    print("\n-------------------------------------")
    print("Training Decision Tree (Reduced Features)")
    print("-------------------------------------")
    dtree_reduced = DecisionTreeClassifier(random_state=RANDOM_STATE_INITIAL)
    dtree_reduced.fit(X_train_reduced, y_train_reduced)
    print("\nTest set performance (DT Reduced):")
    dt_reduced_metrics = model_performance_classification_sklearn(dtree_reduced, X_test_reduced, y_test_reduced)
    print(dt_reduced_metrics)
    confusion_matrix_sklearn(dtree_reduced, X_test_reduced, y_test_reduced, title="DT Reduced Test Confusion Matrix")

    # Tuned Decision Tree (Reduced Features)
    print("\n-------------------------------------")
    print("Tuning Decision Tree (Reduced Features)")
    print("-------------------------------------")
    dtree_classifier_reduced = DecisionTreeClassifier(random_state=RANDOM_STATE_INITIAL)
    grid_search_reduced = GridSearchCV(
        estimator=dtree_classifier_reduced,
        param_grid=DT_PARAM_GRID,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search_reduced.fit(X_train_reduced, y_train_reduced)
    dtree_tuned_reduced_features = grid_search_reduced.best_estimator_
    print(f"Best estimator from GridSearchCV (Reduced Features): {dtree_tuned_reduced_features}")
    print("\nTest set performance (Tuned DT Reduced):")
    tuned_dt_reduced_metrics = model_performance_classification_sklearn(dtree_tuned_reduced_features, X_test_reduced, y_test_reduced)
    print(tuned_dt_reduced_metrics)
    confusion_matrix_sklearn(dtree_tuned_reduced_features, X_test_reduced, y_test_reduced, title="Tuned DT Reduced Test Confusion Matrix")

    # Logistic Regression (Reduced Features)
    print("\n-----------------------------------------")
    print("Training Logistic Regression (Reduced Features)")
    print("-----------------------------------------")
    X_train_logit_reduced = sm.add_constant(X_train_reduced)
    X_test_logit_reduced = sm.add_constant(X_test_reduced)
    logit_reduced = sm.Logit(y_train_reduced, X_train_logit_reduced.astype(float))
    lg_reduced = logit_reduced.fit(disp=False)
    print(lg_reduced.summary())

    y_train_pred_proba_lg_reduced = lg_reduced.predict(X_train_logit_reduced)
    precision_lg_reduced, recall_lg_reduced, thresholds_lg_reduced = precision_recall_curve(y_train_reduced, y_train_pred_proba_lg_reduced)
    f1_scores_lg_reduced = np.divide((2 * precision_lg_reduced * recall_lg_reduced), (precision_lg_reduced + recall_lg_reduced), out=np.zeros_like(precision_lg_reduced), where=(precision_lg_reduced + recall_lg_reduced) != 0)
    f1_scores_lg_reduced = np.nan_to_num(f1_scores_lg_reduced)
    optimal_idx_lg_reduced = np.argmax(f1_scores_lg_reduced)
    optimal_threshold_lg_reduced = thresholds_lg_reduced[optimal_idx_lg_reduced]

    print("\nTest set performance (LG Reduced):")
    lg_reduced_test_metrics = model_performance_classification_statsmodels(lg_reduced, X_test_logit_reduced, y_test_reduced, threshold=optimal_threshold_lg_reduced)
    print(lg_reduced_test_metrics)
    confusion_matrix_statsmodels(lg_reduced, X_test_logit_reduced, y_test_reduced, threshold=optimal_threshold_lg_reduced, title="LG Reduced Test Confusion Matrix")

    # SVM (Reduced Features)
    print("\n-------------------------------------")
    print("Training SVM (Reduced Features)")
    print("-------------------------------------")
    svm_classifier_reduced = SVC(random_state=RANDOM_STATE_INITIAL)
    svm_classifier_reduced.fit(X_train_reduced, y_train_reduced)
    print("\nTest set performance (SVM Reduced):")
    svm_reduced_metrics = model_performance_classification_sklearn(svm_classifier_reduced, X_test_reduced, y_test_reduced)
    print(svm_reduced_metrics)
    confusion_matrix_sklearn(svm_classifier_reduced, X_test_reduced, y_test_reduced, title="SVM Reduced Test Confusion Matrix")

    # Naive Bayes (Reduced Features)
    print("\n-------------------------------------")
    print("Training Naive Bayes (Reduced Features)")
    print("-------------------------------------")
    nb_classifier_reduced = GaussianNB()
    nb_classifier_reduced.fit(X_train_reduced, y_train_reduced)
    print("\nTest set performance (Naive Bayes Reduced):")
    nb_reduced_metrics = model_performance_classification_sklearn(nb_classifier_reduced, X_test_reduced, y_test_reduced)
    print(nb_reduced_metrics)
    confusion_matrix_sklearn(nb_classifier_reduced, X_test_reduced, y_test_reduced, title="Naive Bayes Reduced Test Confusion Matrix")

    # Save the best model
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    joblib.dump(rf_classifier_reduced, SAVED_MODEL_PATH)
    print(f"\nBest performing model (Random Forest with reduced features) saved to {SAVED_MODEL_PATH}")

if __name__ == '__main__':
    train_models()
