from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MARKET_FEATURES = [
    "return_1", "return_3", "return_12", "dist_to_sma",
    "volatility", "volume_ratio", "candle_range", "hour", "weekday",
]
CLASSES = [-1, 0, 1]
CANDLE_TIME_IS_OPEN = True
WINDOW_MINUTES = 60
MAX_NEWS_PER_CANDLE = 15
stop_words = [
    "для", "что", "это", "как", "при",
    "его", "она", "они", "был", "была",
    "все", "уже", "также", "всегда",
    "нужно", "может", "будет", "чтобы"
]

def prepare_dataset():
    candles = pd.read_csv(DATA / "processed.csv", parse_dates=["datetime"])
    with sqlite3.connect(DATA / "ria_news_all_2025.db") as conn:
        news = pd.read_sql_query("SELECT published, title, full_text FROM news ORDER BY published", conn)

    candles = candles.sort_values("datetime").reset_index(drop=True)
    candles = candles.dropna(subset=["datetime", "target"] + MARKET_FEATURES)
    news["published"] = pd.to_datetime(news["published"], errors="coerce")
    news = news.dropna(subset=["published"]).sort_values("published")
    news = news.drop_duplicates(subset=["published", "title"])
    news["text"] = (news["title"].fillna("").astype(str).str.lower())
    news["text"] = news["text"].str.replace(r"\b\d+\b", " ", regex=True)
    timestamps = news["published"].to_numpy(dtype="datetime64[ns]")
    texts = news["text"].tolist()
    cutoff = candles["datetime"]
    if CANDLE_TIME_IS_OPEN:
        cutoff = cutoff + pd.Timedelta(minutes=5)
    moments = cutoff.to_numpy(dtype="datetime64[ns]")
    left = np.searchsorted(timestamps, moments - np.timedelta64(WINDOW_MINUTES, "m"), side="right")
    right = np.searchsorted(timestamps, moments, side="left")
    candles["news_count"] = right - left
    candles["news_text"] = [" ".join(texts[max(l, r - MAX_NEWS_PER_CANDLE):r]) for l, r in zip(left, right)]
    return candles


def create_model():
    transformers = []
    transformers.append(("text", TfidfVectorizer(lowercase=True, max_features=2000, min_df=10, max_df=0.85, ngram_range=(1, 2), token_pattern=r"(?u)\b\w\w+\b", stop_words=stop_words), "news_text"))
    transformers.append(("news_count", StandardScaler(), ["news_count"]))
    transformers.append(("market", StandardScaler(), MARKET_FEATURES))
    return Pipeline([("features", ColumnTransformer(transformers)), ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),])

def main():
    df = prepare_dataset()
    split = int(len(df) * 0.80)
    train, test = df.iloc[:split - 1], df.iloc[split:]
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(train), 1)), train["target"])
    base_pred = baseline.predict(np.zeros((len(test), 1)))
    print("Baseline Macro F1:", round(f1_score(test["target"], base_pred, labels=CLASSES, average="macro", zero_division=0), 4))

    cols = ["news_text", "news_count"] + MARKET_FEATURES
    model = create_model()
    model.fit(train[cols], train["target"])
    pred = model.predict(test[cols])
    f1 = f1_score(test["target"], pred, labels=CLASSES, average="macro", zero_division=0)
    print(f"\nMacro F1 = {f1:.4f}")
    print(classification_report(test["target"], pred, labels=CLASSES, target_names=["DOWN", "NEUTRAL", "UP"], zero_division=0))
    feature_names = model.named_steps["features"].get_feature_names_out()
    text_positions = np.flatnonzero(np.char.startswith(np.asarray(feature_names, dtype=str), "text__"))
    clf = model.named_steps["classifier"]
    rows = []
    for class_label, coefficients in zip(clf.classes_, clf.coef_):
        word_positions = text_positions[np.argsort(coefficients[text_positions])[-15:][::-1]]
        print(f"Наиболее характерные слова для класса {class_label}:")
        for p in word_positions:
            word = feature_names[p].removeprefix("text__")
            print(f"  {word:<35} {coefficients[p]:+.3f}")
            rows.append({"class": class_label, "word": word, "coefficient": coefficients[p]})

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(reports / "nlp_words_importance.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()