"""
News Sentiment Tracker / 新闻情绪追踪器

Records news count and average sentiment per fetch,
persists to data/news_sentiment_history.json.
记录每次抓取的新闻数量和平均情绪分数，持久化到 JSON。

Used by the dashboard to display sentiment trends over time.
供 dashboard 展示情绪随时间的变化趋势。
"""

import json
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HISTORY_PATH = os.path.join(DATA_DIR, "news_sentiment_history.json")
os.makedirs(DATA_DIR, exist_ok=True)


def load_sentiment_history() -> list[dict]:
    """Load sentiment history from disk / 从磁盘加载情绪历史"""
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_sentiment_history(history: list[dict]):
    """Save sentiment history to disk / 保存情绪历史到磁盘"""
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def record_sentiment(sentiment_data: dict, polymarket_odds: float | None = None):
    """
    Record a sentiment snapshot from one analysis run.
    记录一次分析的情绪快照。

    Args:
        sentiment_data: output from aggregate_sentiment(), must contain:
            - composite_sentiment: float (-1 to +1)
            - news_count: int
            - hawkish_count, dovish_count, neutral_count: int
            - sentiment_std: float
        polymarket_odds: optional current Yes price of target contract (0-1)
    """
    history = load_sentiment_history()

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "news_count": sentiment_data.get("news_count", 0),
        "composite_sentiment": sentiment_data.get("composite_sentiment", 0.0),
        "sentiment_std": sentiment_data.get("sentiment_std", 0.0),
        "hawkish_count": sentiment_data.get("hawkish_count", 0),
        "dovish_count": sentiment_data.get("dovish_count", 0),
        "neutral_count": sentiment_data.get("neutral_count", 0),
    }

    if polymarket_odds is not None:
        entry["polymarket_odds"] = round(float(polymarket_odds), 4)

    history.append(entry)

    # Keep last 500 entries to prevent unbounded growth / 保留最近 500 条
    if len(history) > 500:
        history = history[-500:]

    save_sentiment_history(history)
    return entry


def get_trend_data() -> list[dict]:
    """
    Get sentiment history formatted for charting.
    获取格式化后的情绪历史数据，用于绘图。

    Returns:
        list of dicts with timestamp, news_count, composite_sentiment, polymarket_odds
    """
    return load_sentiment_history()
