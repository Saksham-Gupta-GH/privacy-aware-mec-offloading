import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import config

def train_privacy_classifier():
    dataset_path = config.BASE_DIR / "data" / "healthcare_tasks.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run dataset_generator.py first.")

    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples from {dataset_path}")

    # Features: Text descriptions
    X = df["description"].values
    # Target: Labeled privacy level (incorporating realistic annotator noise)
    y_labeled = df["labeled_privacy_level"].values
    y_true = df["true_privacy_level"].values

    # Stratified Train/Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_labeled, np.arange(len(df)), test_size=0.20, random_state=42, stratify=y_labeled
    )

    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression classifier with balanced class weights
    classifier = LogisticRegression(
        C=2.5,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    classifier.fit(X_train_vec, y_train)

    # Predictions on test set
    y_pred = classifier.predict(X_test_vec)
    y_pred_proba = classifier.predict_proba(X_test_vec)

    # Evaluation against labeled test set
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    class_report = classification_report(
        y_test, y_pred,
        target_names=["L0: Public", "L1: Restricted", "L2: Critical"],
        output_dict=True
    )

    # Also evaluate against TRUE ground truth for the test subset
    true_test = y_true[idx_test]
    acc_vs_true = accuracy_score(true_test, y_pred)
    f1_vs_true = f1_score(true_test, y_pred, average="weighted")

    print("\n" + "="*50)
    print("CLASSIFIER EMPIRICAL PERFORMANCE REPORT")
    print("="*50)
    print(f"Accuracy (vs Labeled):    {acc*100:.2f}%")
    print(f"Weighted F1 (vs Labeled): {weighted_f1:.4f}")
    print(f"Macro F1 (vs Labeled):    {macro_f1:.4f}")
    print(f"Accuracy (vs True):       {acc_vs_true*100:.2f}% (captures label noise impact)")
    print(f"Weighted F1 (vs True):    {f1_vs_true:.4f}")
    print("="*50)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = ["L0: Public", "L1: Restricted", "L2: Critical"]

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Healthcare Task Privacy Classifier\nConfusion Matrix (Test Set)")
    plt.ylabel("Annotated Privacy Tier")
    plt.xlabel("Predicted Privacy Tier")
    plt.tight_layout()

    # Save plots
    fig_dir = config.BASE_DIR / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = config.BASE_DIR / "experiments" / "results"
    exp_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(fig_dir / "fig1_confusion_matrix.png", dpi=300)
    plt.savefig(exp_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    print(f"Saved confusion matrix figure to {fig_dir / 'fig1_confusion_matrix.png'}")

    # Save models
    models_dir = config.BASE_DIR / "data" / "trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, models_dir / "privacy_classifier.joblib")
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.joblib")
    print(f"Saved trained models to {models_dir}")

    # Save metrics JSON
    metrics = {
        "accuracy_vs_labeled": float(acc),
        "weighted_f1_vs_labeled": float(weighted_f1),
        "macro_f1_vs_labeled": float(macro_f1),
        "accuracy_vs_true": float(acc_vs_true),
        "weighted_f1_vs_true": float(f1_vs_true),
        "classification_report": class_report,
        "confusion_matrix": cm.tolist()
    }
    with open(exp_dir / "classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    train_privacy_classifier()
