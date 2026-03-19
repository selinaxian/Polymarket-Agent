"""
RAG 存储模块
用 ChromaDB 存储新闻和分析结果，支持语义检索
两个 collection：
  - tariff_news: 每次抓取的新闻 + 当时的分析结果（短期参考）
  - tariff_events: 历史重大关税事件（长期参考，由 tariff_history.py 填充）
"""

import os
import json
import chromadb
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chromadb")
os.makedirs(DB_DIR, exist_ok=True)

_client = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_DIR)
    return _client


# ============================================================
# 短期新闻存储
# ============================================================

def get_news_collection():
    return get_client().get_or_create_collection("tariff_news")


def store_news(news_items: list[dict], analysis: dict = None):
    """
    存储一批新闻及对应的分析结果

    Args:
        news_items: 新闻列表
        analysis: 本次 AI 分析结果（可选，作为 metadata 附加）
    """
    col = get_news_collection()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    documents = []
    metadatas = []
    ids = []

    for i, n in enumerate(news_items):
        doc = f"[{n.get('date', '')}] {n['title']}"
        if n.get("summary"):
            doc += f" | {n['summary'][:200]}"

        meta = {
            "date": n.get("date", ""),
            "source": n.get("source", ""),
            "title": n.get("title", ""),
            "stored_at": ts,
        }
        if analysis:
            meta["policy_signal"] = analysis.get("policy_signal", "")
            meta["signal_strength"] = str(analysis.get("signal_strength", 0))

        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"news_{ts}_{i}")

    if documents:
        col.add(documents=documents, metadatas=metadatas, ids=ids)


def find_similar_news(query: str, n_results: int = 3) -> list[dict]:
    """
    检索与 query 最相似的历史新闻

    Returns:
        list[dict]: 每条包含 document, metadata, distance
    """
    col = get_news_collection()
    if col.count() == 0:
        return []

    results = col.query(query_texts=[query], n_results=min(n_results, col.count()))

    items = []
    for i in range(len(results["documents"][0])):
        items.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })
    return items


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    # 存一些测试新闻
    test_news = [
        {"date": "2025-04-02", "title": "Trump announces sweeping reciprocal tariffs", "source": "Reuters", "summary": "Liberation Day tariffs take effect"},
        {"date": "2025-04-09", "title": "Trump pauses reciprocal tariffs for 90 days", "source": "White House", "summary": "90-day pause announced"},
        {"date": "2025-03-10", "title": "Trump raises China tariffs to 20%", "source": "Reuters", "summary": "Escalation continues"},
    ]
    store_news(test_news, {"policy_signal": "escalation", "signal_strength": 7})

    # 检索
    print("=== 检索相似新闻 ===\n")
    results = find_similar_news("new tariffs on Chinese imports", n_results=3)
    for r in results:
        print(f"  [{r['metadata'].get('date', '')}] {r['document'][:80]}")
        print(f"    distance: {r['distance']:.4f}  signal: {r['metadata'].get('policy_signal', '')}")
        print()
