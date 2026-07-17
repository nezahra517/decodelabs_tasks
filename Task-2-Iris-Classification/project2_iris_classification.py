"""
DecodeLabs - Project 2: Data Classification Using AI
======================================================

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42  # fixed seed so results are reproducible


# ----------------------------------------------------------------------
# STEP 1: LOAD AND UNDERSTAND THE DATASET  ("Raw Material: The Iris Benchmark")
# ----------------------------------------------------------------------
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    class_names = iris.target_names  # ['setosa' 'versicolor' 'virginica']

    print("=" * 60)
    print("STEP 1: DATASET OVERVIEW")
    print("=" * 60)
    print(f"Samples: {X.shape[0]}   Features: {X.shape[1]}   Classes: {len(class_names)}")
    print(f"Class names: {list(class_names)}")
    print("\nFirst 5 rows:")
    print(X.head())
    print("\nClass balance (should be ~50 each, i.e. balanced):")
    print(y.value_counts().sort_index())
    print()
    return X, y, class_names


# ----------------------------------------------------------------------
# STEP 2: TRAIN / TEST SPLIT  ("Structural Integrity: The Split")
# ----------------------------------------------------------------------
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,        # randomize before splitting to remove order bias
        stratify=y,          # keep class proportions equal in train & test
    )
    print("=" * 60)
    print("STEP 2: TRAIN/TEST SPLIT (80/20)")
    print("=" * 60)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples\n")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# STEP 3: FEATURE SCALING  ("The Gatekeeper Rule: Scaling")
# ----------------------------------------------------------------------
def scale_features(X_train, X_test):
    scaler = StandardScaler()          # mean = 0, variance = 1
    X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on training data
    X_test_scaled = scaler.transform(X_test)          # transform test using same scaler

    print("=" * 60)
    print("STEP 3: FEATURE SCALING (StandardScaler)")
    print("=" * 60)
    print("Why: KNN is distance-based. Unscaled features (e.g. petal length in cm")
    print("vs. a feature in mm) would unfairly dominate distance calculations.\n")
    return X_train_scaled, X_test_scaled, scaler


# ----------------------------------------------------------------------
# STEP 4: CHOOSE OPTIMAL K  ("Tuning the Engine: Choosing K")
# ----------------------------------------------------------------------
def find_best_k(X_train_scaled, y_train, X_test_scaled, y_test, k_range=range(1, 21)):
    error_rates = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        error_rates.append(np.mean(preds != y_test))

    best_k = list(k_range)[int(np.argmin(error_rates))]

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), error_rates, marker="o", linestyle="--", color="#1f4e79")
    plt.axvline(best_k, color="orange", linestyle=":", label=f"Best K = {best_k}")
    plt.title("Error Rate vs. K Value (Finding the Elbow)")
    plt.xlabel("K Value")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("k_selection_plot.png", dpi=150)
    plt.close()

    print("=" * 60)
    print("STEP 4: CHOOSING K (K-Nearest Neighbors)")
    print("=" * 60)
    print(f"Tested K = 1 to {max(k_range)}. Optimal K = {best_k} (lowest error rate)")
    print("Plot saved -> k_selection_plot.png\n")
    return best_k


# ----------------------------------------------------------------------
# STEP 5: INSTANTIATE -> FIT -> PREDICT  ("The Workflow: Scikit-Learn")
# ----------------------------------------------------------------------
def train_and_predict(X_train_scaled, y_train, X_test_scaled, best_k):
    model = KNeighborsClassifier(n_neighbors=best_k)  # INSTANTIATE
    model.fit(X_train_scaled, y_train)                 # FIT (memorize the map)
    predictions = model.predict(X_test_scaled)          # PREDICT (apply logic)

    print("=" * 60)
    print(f"STEP 5: MODEL TRAINED (KNN, K={best_k})")
    print("=" * 60 + "\n")
    return model, predictions


# ----------------------------------------------------------------------
# STEP 6: EVALUATE  ("Output Validation" + "The Diagnostic Tool" + "Strategic Trade-offs")
# ----------------------------------------------------------------------
def evaluate_model(y_test, predictions, class_names):
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    cm = confusion_matrix(y_test, predictions)

    print("=" * 60)
    print("STEP 6: OUTPUT VALIDATION")
    print("=" * 60)
    print(f"Accuracy : {acc:.2%}")
    print(f"F1 Score : {f1:.3f}  (harmonic mean of Precision & Recall)\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (TP / FP / FN / TN per class)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix plot saved -> confusion_matrix.png\n")

    return acc, f1, cm


# ----------------------------------------------------------------------
# MAIN PIPELINE  ("The Full Architecture")
# ----------------------------------------------------------------------
def main():
    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    best_k = find_best_k(X_train_scaled, y_train, X_test_scaled, y_test)
    model, predictions = train_and_predict(X_train_scaled, y_train, X_test_scaled, best_k)
    evaluate_model(y_test, predictions, class_names)

    # Demo: classify one brand-new, unseen flower
    print("=" * 60)
    print("BONUS: PREDICTING A NEW, UNSEEN FLOWER")
    print("=" * 60)
    new_flower = pd.DataFrame(
        [[5.5,2.5,4.0,1.3]],  # sepal_len, sepal_wid, petal_len, petal_wid
        columns=X.columns,
    )
    new_flower_scaled = scaler.transform(new_flower)
    prediction = model.predict(new_flower_scaled)
    print(f"Input measurements: {new_flower.values.tolist()[0]}")
    print(f"Predicted species : {class_names[prediction[0]].upper()}")

    print("\n" + "=" * 60)
    print("PROJECT 2 COMPLETE. Milestone achieved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
