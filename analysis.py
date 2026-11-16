from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

MARKET_FEATURES = [
    "return_1", "return_3", "return_12",
    "dist_to_sma", "volatility", "volume_ratio",
    "candle_range", "hour", "weekday"
]

NEWS_FEATURES = [
    "news_count", "news_sentiment", "news_sber_mentions"
]

FEATURES = MARKET_FEATURES + NEWS_FEATURES


def save_plot(name):
    plt.tight_layout()
    plt.savefig(REPORTS / name, dpi=150)
    plt.close()


def load_data():
    with sqlite3.connect(DATA / "market_data_clean.db") as conn:
        market = pd.read_sql_query(
            """
            SELECT datetime, open, high, low, close, volume
            FROM sber_5min
            ORDER BY datetime
            """,
            conn
        )
    with sqlite3.connect(DATA / "ria_news_all_2025.db") as conn:
        news = pd.read_sql_query(
            """
            SELECT title, full_text, published
            FROM news
            ORDER BY published
            """,
            conn
        )
    market["datetime"] = pd.to_datetime(market["datetime"], errors="coerce")
    news["published"] = pd.to_datetime(news["published"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    market[numeric] = market[numeric].apply(pd.to_numeric, errors="coerce")

    market = (market.dropna(subset=["datetime"] + numeric).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True))

    market = market[(market["close"] > 0) & (market["volume"] >= 0)].copy()

    news = (news.dropna(subset=["published"]).sort_values("published").reset_index(drop=True))

    return market, news

def make_features(market, news):
    df = market.copy()

    df["return_1"] = df["close"].pct_change(fill_method=None)
    df["return_3"] = df["close"].pct_change(3, fill_method=None)
    df["return_12"] = df["close"].pct_change(12, fill_method=None)

    df["sma_10"] = df["close"].rolling(10).mean()
    df["dist_to_sma"] = (df["close"] / df["sma_10"] - 1)

    df["volatility"] = (df["return_1"].rolling(12).std()) #отклонение, то есть колебание

    df["volume_ratio"] = (df["volume"] / df["volume"].rolling(12).mean().replace(0, np.nan))

    df["candle_range"] = ((df["high"] - df["low"]) / df["close"])

    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.dayofweek

    news = news.copy()
    news["text"] = (news["title"].fillna("").astype(str) + " " + news["full_text"].fillna("").astype(str))

    positive = ["рост", "вырос", "прибыл", "рекорд", "увелич"]
    negative = ["паден", "упал", "убыт", "кризис", "сниж"]

    def sentiment(text):
        text = text.lower()
        pos = sum(word in text for word in positive)
        neg = sum(word in text for word in negative)
        return (pos - neg) / (pos + neg + 1)

    news["sentiment"] = news["text"].apply(sentiment)

    news["sber_mention"] = (news["text"].str.contains("сбер|sber", case=False, regex=True).astype(int))

    times = news["published"].to_numpy(dtype="datetime64[ns]")
    candles = df["datetime"].to_numpy(dtype="datetime64[ns]")

    left = np.searchsorted(times, candles - np.timedelta64(60, "m"), side="right")
    right = np.searchsorted(times, candles, side="right")

    df["news_count"] = right - left

    for column, result in [("sentiment", "news_sentiment"),("sber_mention", "news_sber_mentions")]:
        totals = np.r_[0, np.cumsum(news[column].to_numpy())]
        values = totals[right] - totals[left]

        if column == "sentiment":
            values = values / np.maximum(df["news_count"].to_numpy(), 1)
        df[result] = values


    df["future_return"] = (df["close"].shift(-1) / df["close"] - 1)

    df["target"] = np.select([df["future_return"] < -0.001, df["future_return"] > 0.001], [-1, 1], default=0)

    gap = df["datetime"].shift(-1) - df["datetime"]
    df = df[gap.between(pd.Timedelta(minutes=1), pd.Timedelta(minutes=10))].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)

    return df.reset_index(drop=True), news

def make_charts(df, news):

    plt.figure(figsize=(12, 4))
    plt.plot(df["datetime"], df["close"])
    plt.title("SBER share price")
    plt.xlabel("Date")
    plt.ylabel("Close price")
    save_plot("01_price.png")

    plt.figure(figsize=(8, 4))
    plt.hist(df["return_1"] * 100, bins=60)
    plt.title("Distribution of 5-minute returns")
    plt.xlabel("Return, %")
    plt.ylabel("Candles")
    save_plot("02_returns.png")

    sample = df.sample(n=min(len(df), 4000), random_state=42)
    plt.figure(figsize=(8, 5))
    plt.scatter(sample["volume_ratio"], sample["return_1"] * 100, s=5, alpha=0.3)
    plt.title("Relative trading volume and return")
    plt.xlabel("Relative volume")
    plt.ylabel("Return, %")
    save_plot("03_volume_return.png")

    counts = news.groupby(news["published"].dt.to_period("M")).size()
    plt.figure(figsize=(11, 4))
    counts.plot(kind="bar")
    plt.title("News articles by month")
    plt.ylabel("Count")
    save_plot("04_news_month.png")

    plt.figure(figsize=(8, 4))
    plt.hist(news["sentiment"], bins=25)
    plt.title("Distribution of news sentiment")
    plt.xlabel("Sentiment score")
    plt.ylabel("Articles")
    save_plot("05_sentiment.png")

    plt.figure(figsize=(7, 4))
    df["target"].value_counts().reindex([-1, 0, 1], fill_value=0).plot(kind="bar")
    plt.title("Target class distribution")
    plt.xticks(range(3),["DOWN", "NEUTRAL", "UP"], rotation=0)
    save_plot("06_classes.png")

    corr = df[FEATURES + ["target"]].corr()
    plt.figure(figsize=(11, 9))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Feature correlations")
    save_plot("07_correlations.png")


    plt.figure(figsize=(9, 5))
    corr["target"].drop("target").sort_values().plot(kind="barh")
    plt.title("Correlation with target")
    save_plot("08_target_correlation.png")


    plt.figure(figsize=(9, 4))
    df.groupby("hour")["target"].mean().plot(marker="o")
    plt.axhline(0, color="gray", linewidth=1)
    plt.title("Average direction label by hour")
    plt.xlabel("Hour")
    plt.ylabel("Average target")
    save_plot("09_hour_target.png")

    groups = [df.loc[df["target"] == c, "volatility"].dropna() * 100 for c in [-1, 0, 1]]
    plt.figure(figsize=(8, 4))
    plt.boxplot(groups, tick_labels=["DOWN", "NEUTRAL", "UP"], showfliers=False)
    plt.title("Volatility by target class")
    plt.ylabel("Volatility, %")
    save_plot("10_volatility_classes.png")


    plt.figure(figsize=(8, 4))
    df.groupby("target")["news_count"].mean().reindex([-1, 0, 1]).plot(kind="bar")
    plt.title("Average recent news count by class")
    plt.xticks(range(3),["DOWN", "NEUTRAL", "UP"],rotation=0)
    save_plot("11_news_classes.png")

    plt.figure(figsize=(8, 4))
    df.groupby("target")["news_sentiment"].mean().reindex([-1, 0, 1]).plot(kind="bar")
    plt.title("Average recent news sentiment by class")
    plt.xticks(range(3),["DOWN", "NEUTRAL", "UP"],rotation=0)
    save_plot("12_sentiment_classes.png")

def main():
    market, news = load_data()

    print("Market:", market.shape)
    print("News:", news.shape)

    df, news = make_features(market, news)

    print("\nРаспределение классов:")
    print(df["target"].value_counts(normalize=True).sort_index())

    print("\nКорреляции с target:")
    print(df[FEATURES + ["target"]].corr()["target"].sort_values())

    df.to_csv(DATA / "processed.csv", index=False)

    make_charts(df, news)


if __name__ == "__main__":
    main()
