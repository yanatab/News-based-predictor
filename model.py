
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



DATA = Path("data")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

MARKET_FEATURES = [
    "return_1", "return_3", "return_12",
    "dist_to_sma", "volatility", "volume_ratio",
    "candle_range", "hour", "weekday"
]

NEWS_FEATURES = ["news_count", "news_sentiment", "news_sber_mentions"]

ALL_FEATURES = MARKET_FEATURES + NEWS_FEATURES

CLASSES = [-1, 0, 1]  #0.1%
CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]

def create_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))


def score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)

def main():

    df = pd.read_csv(DATA / "processed.csv", parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print("\nРаспределение классов:")
    print(df["target"].value_counts().sort_index())
    print("\nДоля классов:")
    print(df["target"].value_counts(normalize=True).sort_index())

    cut1 = int(0.60 * len(df))
    cut2 = int(0.80 * len(df))
    train = df.iloc[:cut1 - 1].copy() #60%
    valid = df.iloc[cut1:cut2 - 1].copy() #20%
    test = df.iloc[cut2:].copy() #20%

    selector = create_model()
    selector.fit(train[ALL_FEATURES], train["target"])
    importance = permutation_importance(selector, valid[ALL_FEATURES], valid["target"], scoring="f1_macro", n_repeats=5, random_state=42)
    ranking = pd.Series(importance.importances_mean, index=ALL_FEATURES).sort_values(ascending=False)
    print(f"Веса: \n{ranking}")

    ranking.to_csv(REPORTS / "feature_importance.csv", header=["importance"])
    selected = (ranking[ranking > 0].head(7).index.tolist())
    if not selected:
        selected = ranking.head(5).index.tolist()

    print("\nВыбранные признаки:")
    for feature in selected:
        print("-", feature)
    plt.figure(figsize=(10, 6))
    ranking.sort_values().plot(kind="barh")
    plt.title("Permutation Feature Importance")
    plt.xlabel("Decrease in Macro F1")
    plt.ylabel("Features")
    plt.axvline(0, color="gray", linewidth=1)
    plt.tight_layout()
    plt.savefig(REPORTS / "13_feature_importance.png", dpi=150)
    plt.close()

    train_full = df.iloc[:cut2 - 1].copy()
    experiments = {"Market-only": MARKET_FEATURES,
        "News-only": NEWS_FEATURES,
        "Market + News": ALL_FEATURES,
        "Selected features": selected}
    results = []

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train_full[ALL_FEATURES], train_full["target"])
    baseline_train_pred = baseline.predict(train_full[ALL_FEATURES])
    baseline_test_pred = baseline.predict(test[ALL_FEATURES])
    baseline_train_accuracy = accuracy_score(train_full["target"], baseline_train_pred)
    baseline_test_accuracy = accuracy_score(test["target"], baseline_test_pred)
    baseline_train_f1 = score(train_full["target"], baseline_train_pred)
    baseline_test_f1 = score(test["target"], baseline_test_pred)

    results.append({
        "model": "Baseline",
        "train_accuracy": baseline_train_accuracy,
        "test_accuracy": baseline_test_accuracy,
        "train_f1": baseline_train_f1,
        "test_f1": baseline_test_f1,
        "f1_difference": baseline_train_f1 - baseline_test_f1
    })
    print("\nBaseline:")
    print(f"Train Accuracy: {baseline_train_accuracy:.4f}")
    print(f"Test Accuracy:  {baseline_test_accuracy:.4f}")
    print(f"Train Macro F1: {baseline_train_f1:.4f}")
    print(f"Test Macro F1:  {baseline_test_f1:.4f}")

    for name, columns in experiments.items():
        model = create_model()
        model.fit(train_full[columns], train_full["target"])

        train_pred = model.predict(train_full[columns])
        test_pred = model.predict(test[columns])
        train_accuracy = accuracy_score(train_full["target"], train_pred)
        train_f1 = score(train_full["target"], train_pred)
        test_accuracy = accuracy_score(test["target"], test_pred)
        test_f1 = score(test["target"], test_pred)
        f1_difference = train_f1 - test_f1

        results.append({
            "model": name,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "f1_difference": f1_difference
        })
        if name == "Market + News":
            ConfusionMatrixDisplay.from_predictions(
                test["target"],
                test_pred,
                labels=CLASSES,
                display_labels=CLASS_NAMES,
                cmap="Blues")
            plt.title("Confusion Matrix - Market + News")
            plt.tight_layout()
            plt.savefig(REPORTS / "15_confusion_matrix.png", dpi=150)
            plt.close()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_f1", ascending=False)
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv(REPORTS / "test_results.csv", index=False)


    x = np.arange(len(results_df))
    width = 0.35
    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, results_df["train_f1"], width, label="Train Macro F1")
    plt.bar(x + width / 2, results_df["test_f1"], width, label="Test Macro F1")
    plt.xticks(x, results_df["model"], rotation=20)
    plt.ylabel("Macro F1")
    plt.title("Train vs Test - Model Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / "14_model_comparison.png", dpi=150)
    plt.close()

    ml_results = results_df[results_df["model"] != "Baseline"]
    best_model = ml_results.iloc[0]
    print("\nЛучшая ML-модель по Test Macro F1:")
    print(best_model["model"])
    print(f"Train Macro F1: {best_model['train_f1']:.4f}")
    print(f"Test Macro F1: {best_model['test_f1']:.4f}")
    print(f"Test Accuracy: {best_model['test_accuracy']:.4f}")
    print(f"Разница Train-Test: {best_model['f1_difference']:.4f}")


if __name__ == "__main__":
    main()
