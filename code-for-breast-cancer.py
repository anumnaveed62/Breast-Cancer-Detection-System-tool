# ============================================================
# BREAST CANCER CLASSIFICATION
# Wisconsin Diagnostic Breast Cancer Dataset
# CPU-ONLY MACHINE LEARNING PROJECT
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "breast_cancer_dataset.csv"

MODEL_PATH = "best_breast_cancer_model.pkl"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("BREAST CANCER CLASSIFICATION")
print("=" * 70)

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Make sure breast_cancer_dataset.csv is in the "
        "same folder as this Python file."
    )


data = pd.read_csv(DATA_PATH)


print("\nDataset loaded successfully.")

print("\nDataset information:")
print(data.info())

print("\nDataset shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing_values = data.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print(
    "\nTotal missing values:",
    total_missing
)


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    data["diagnosis"].value_counts()
)


# ============================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================

X = data.drop(
    columns=["diagnosis"]
)

y = data["diagnosis"]


# Convert classes to numerical values
#
# BENIGN    = 0
# MALIGNANT = 1

y = y.map({
    "BENIGN": 0,
    "MALIGNANT": 1
})


if y.isnull().any():

    raise ValueError(
        "Unexpected class found in diagnosis column."
    )


print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE
            )
        )
    ]),


    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            )
        )
    ]),


    "Random Forest": RandomForestClassifier(

        n_estimators=300,

        random_state=RANDOM_STATE,

        class_weight="balanced"
    ),


    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),


    "Decision Tree": DecisionTreeClassifier(

        random_state=RANDOM_STATE,

        class_weight="balanced"
    )
}


# ============================================================
# 8. TRAIN MODELS AND EVALUATE
# ============================================================

results = []

trained_models = {}

roc_data = {}


print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)


for model_name, model in models.items():

    print(
        f"\nTraining: {model_name}"
    )

    # Train
    model.fit(
        X_train,
        y_train
    )


    # Predict classes
    y_pred = model.predict(
        X_test
    )


    # Probability for ROC-AUC
    if hasattr(
        model,
        "predict_proba"
    ):

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

    else:

        y_probability = model.decision_function(
            X_test
        )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    tn, fp, fn, tp = cm.ravel()


    # Specificity
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    # Sensitivity = Recall
    sensitivity = recall


    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"Sensitivity : {sensitivity:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print(
        f"F1-score    : {f1:.4f}"
    )

    print(
        f"ROC-AUC     : {roc_auc:.4f}"
    )


    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "Sensitivity": sensitivity,

        "Specificity": specificity,

        "F1-score": f1,

        "ROC-AUC": roc_auc
    })


    trained_models[
        model_name
    ] = model


    # ROC data
    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )


    roc_data[
        model_name
    ] = (
        fpr,
        tpr,
        roc_auc
    )


# ============================================================
# 9. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# Save results

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)


print(
    "\nResults saved to:",
    "model_comparison_results.csv"
)


# ============================================================
# 10. BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best model:",
    best_model_name
)

print(
    "Best ROC-AUC:",
    f"{results_df.iloc[0]['ROC-AUC']:.4f}"
)


# ============================================================
# 11. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_PATH
)


print(
    "\nBest model saved as:",
    MODEL_PATH
)


# ============================================================
# 12. DETAILED CLASSIFICATION REPORT
# ============================================================

best_predictions = best_model.predict(
    X_test
)


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        best_predictions,

        target_names=[
            "BENIGN",
            "MALIGNANT"
        ],

        zero_division=0
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)


print("\nConfusion Matrix:")
print(cm)


plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    [0, 1],
    ["BENIGN", "MALIGNANT"]
)

plt.yticks(
    [0, 1],
    ["BENIGN", "MALIGNANT"]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],

            ha="center",

            va="center"
        )


plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. ROC CURVES
# ============================================================

plt.figure(
    figsize=(9, 7)
)


for model_name, (
    fpr,
    tpr,
    auc
) in roc_data.items():

    plt.plot(
        fpr,
        tpr,

        label=f"{model_name} (AUC={auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],

    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves - Breast Cancer Classification"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "roc_curves.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

if best_model_name == "Random Forest":

    importance = best_model.feature_importances_

    feature_importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": importance

    })


    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )


    print("\n" + "=" * 70)
    print("TOP 10 IMPORTANT FEATURES")
    print("=" * 70)

    print(
        feature_importance.head(10).to_string(
            index=False
        )
    )


    plt.figure(
        figsize=(10, 7)
    )


    top_features = feature_importance.head(10)

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )


    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 10 Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 16. PREDICT A NEW PATIENT
# ============================================================

def predict_patient(values):

    """
    values must contain exactly 30 feature values
    in the same order as the CSV columns.
    """

    if len(values) != len(X.columns):

        raise ValueError(
            f"Expected {len(X.columns)} features, "
            f"but received {len(values)}."
        )


    new_data = pd.DataFrame(
        [values],
        columns=X.columns
    )


    prediction = best_model.predict(
        new_data
    )[0]


    probability = best_model.predict_proba(
        new_data
    )[0]


    if prediction == 0:

        diagnosis = "BENIGN"

    else:

        diagnosis = "MALIGNANT"


    print("\n" + "=" * 70)
    print("NEW PATIENT PREDICTION")
    print("=" * 70)

    print(
        "Predicted class:",
        diagnosis
    )

    print(
        f"Benign probability: "
        f"{probability[0] * 100:.2f}%"
    )

    print(
        f"Malignant probability: "
        f"{probability[1] * 100:.2f}%"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)

print(
    "\nGenerated files:"
)

print(
    "1. model_comparison_results.csv"
)

print(
    "2. best_breast_cancer_model.pkl"
)

print(
    "3. confusion_matrix.png"
)

print(
    "4. roc_curves.png"
)

if best_model_name == "Random Forest":

    print(
        "5. feature_importance.png"
    )

print("\nAll processing completed successfully.")