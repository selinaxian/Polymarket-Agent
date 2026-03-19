"""
历史重大关税事件库
存储 2018-2026 年的重大关税/贸易事件，用 ChromaDB 向量检索历史类比。
与 rag_store.py 共用同一个 ChromaDB 实例，使用 tariff_events collection。
"""

import chromadb
from tools.rag_store import get_client


# ============================================================
# 历史事件数据
# ============================================================

TARIFF_EVENTS = [
    # ── 2018: 贸易战开端 ──
    {
        "id": "evt_2018_01",
        "date": "2018-03-08",
        "description": "Trump signs proclamations imposing 25% tariff on steel and 10% on aluminum imports under Section 232",
        "type": "implementation",
        "countries": ["Global"],
        "market_reaction": {"sp500": -0.4, "dollar_index": -0.2},
        "tariff_landed": True,
        "outcome": "Tariffs implemented; allies given temporary exemptions. Led to broad retaliation from EU, Canada, Mexico.",
    },
    {
        "id": "evt_2018_02",
        "date": "2018-07-06",
        "description": "US imposes 25% tariffs on $34B of Chinese goods under Section 301; China retaliates with matching tariffs",
        "type": "escalation",
        "countries": ["China"],
        "market_reaction": {"sp500": +0.9, "dollar_index": +0.3},
        "tariff_landed": True,
        "outcome": "First round of US-China trade war. Markets initially shrugged it off. Escalation continued through 2018.",
    },
    {
        "id": "evt_2018_03",
        "date": "2018-09-24",
        "description": "US imposes 10% tariffs on $200B of Chinese goods (Section 301 List 3); rate to increase to 25% in Jan 2019",
        "type": "escalation",
        "countries": ["China"],
        "market_reaction": {"sp500": -0.1, "dollar_index": +0.5},
        "tariff_landed": True,
        "outcome": "Massive escalation. Markets dropped sharply in Q4 2018. Led to December G20 meeting and temporary truce.",
    },
    {
        "id": "evt_2018_04",
        "date": "2018-12-01",
        "description": "Trump-Xi dinner at G20 Buenos Aires: agree to 90-day trade truce, delay tariff increase from 10% to 25%",
        "type": "negotiation",
        "countries": ["China"],
        "market_reaction": {"sp500": +1.1, "dollar_index": -0.3},
        "tariff_landed": False,
        "outcome": "Temporary de-escalation. Markets rallied. Tariff hike delayed to March 2019, then further extended.",
    },
    # ── 2019: Escalation and Phase 1 ──
    {
        "id": "evt_2019_01",
        "date": "2019-05-10",
        "description": "US raises tariffs on $200B of Chinese goods from 10% to 25% after trade talks break down",
        "type": "escalation",
        "countries": ["China"],
        "market_reaction": {"sp500": -2.4, "dollar_index": +0.6},
        "tariff_landed": True,
        "outcome": "Sharp market selloff. China retaliated. S&P500 fell ~6% in May. Trade talks suspended.",
    },
    {
        "id": "evt_2019_02",
        "date": "2019-08-01",
        "description": "Trump announces 10% tariff on remaining $300B of Chinese imports; China halts US agricultural purchases",
        "type": "escalation",
        "countries": ["China"],
        "market_reaction": {"sp500": -3.0, "dollar_index": +0.8},
        "tariff_landed": True,
        "outcome": "Major escalation covering virtually all Chinese imports. Markets fell sharply. Led to recession fears.",
    },
    {
        "id": "evt_2019_03",
        "date": "2019-12-13",
        "description": "US and China announce Phase 1 trade deal; US cancels planned December tariff hike, reduces some existing tariffs",
        "type": "deal",
        "countries": ["China"],
        "market_reaction": {"sp500": +0.9, "dollar_index": -0.4},
        "tariff_landed": False,
        "outcome": "De-escalation. Markets rallied. Phase 1 signed Jan 15, 2020. China committed to $200B in purchases (largely unmet).",
    },
    # ── 2020: USMCA ──
    {
        "id": "evt_2020_01",
        "date": "2020-07-01",
        "description": "USMCA (US-Mexico-Canada Agreement) enters into force, replacing NAFTA",
        "type": "deal",
        "countries": ["Canada", "Mexico"],
        "market_reaction": {"sp500": +0.5, "dollar_index": 0.0},
        "tariff_landed": False,
        "outcome": "Largely priced in. Removed steel/aluminum tariff uncertainty for North America. Modest market impact.",
    },
    # ── 2022-2023: Chips and EV ──
    {
        "id": "evt_2022_01",
        "date": "2022-10-07",
        "description": "Biden administration imposes sweeping export controls on semiconductor technology to China",
        "type": "escalation",
        "countries": ["China"],
        "market_reaction": {"sp500": -0.7, "dollar_index": +0.3},
        "tariff_landed": True,
        "outcome": "Not traditional tariffs but technology export ban. Devastated Chinese chip stocks. US semiconductor companies also affected.",
    },
    {
        "id": "evt_2024_01",
        "date": "2024-05-14",
        "description": "Biden raises tariffs on $18B of Chinese goods: 100% on EVs, 50% on solar cells, 25% on steel/aluminum",
        "type": "escalation",
        "countries": ["China"],
        "market_reaction": {"sp500": +0.2, "dollar_index": 0.0},
        "tariff_landed": True,
        "outcome": "Targeted escalation. Markets shrugged it off — most tariffs were on sectors with minimal trade. Symbolic/political move.",
    },
    # ── 2025: Trump 2.0 ──
    {
        "id": "evt_2025_01",
        "date": "2025-02-01",
        "description": "Trump imposes 10% tariff on all Chinese imports, citing fentanyl crisis",
        "type": "implementation",
        "countries": ["China"],
        "market_reaction": {"sp500": -1.5, "dollar_index": +0.8},
        "tariff_landed": True,
        "outcome": "First tariff action of Trump 2.0. Followed by rapid escalation. China retaliated within days.",
    },
    {
        "id": "evt_2025_02",
        "date": "2025-02-04",
        "description": "Trump imposes 25% tariffs on Canada and Mexico over border security; both countries announce retaliation",
        "type": "escalation",
        "countries": ["Canada", "Mexico"],
        "market_reaction": {"sp500": -1.8, "dollar_index": +0.5},
        "tariff_landed": True,
        "outcome": "Tariffs implemented but paused within 48 hours after border concessions. Markets whipsawed.",
    },
    {
        "id": "evt_2025_03",
        "date": "2025-04-02",
        "description": "Trump's 'Liberation Day': sweeping reciprocal tariffs on nearly all trading partners, up to 49% on some countries",
        "type": "escalation",
        "countries": ["Global", "China", "EU", "Japan", "Vietnam"],
        "market_reaction": {"sp500": -10.5, "dollar_index": -2.0},
        "tariff_landed": True,
        "outcome": "Largest tariff action in modern history. S&P500 crashed. 90-day pause announced April 9 for all except China.",
    },
    {
        "id": "evt_2025_04",
        "date": "2025-04-09",
        "description": "Trump announces 90-day pause on reciprocal tariffs for most countries; raises China tariffs to 125%",
        "type": "negotiation",
        "countries": ["Global", "China"],
        "market_reaction": {"sp500": +9.5, "dollar_index": +1.0},
        "tariff_landed": False,
        "outcome": "Massive relief rally. S&P500 had best day since 2008. China exempted from pause — tariffs rose to 145% effective rate.",
    },
    {
        "id": "evt_2025_05",
        "date": "2025-05-12",
        "description": "US and China announce Geneva agreement: US reduces China tariffs from 145% to 30% for 90 days",
        "type": "deal",
        "countries": ["China"],
        "market_reaction": {"sp500": +3.3, "dollar_index": +0.5},
        "tariff_landed": False,
        "outcome": "Temporary de-escalation. Markets surged. Seen as face-saving for both sides. Tariff rates still historically high.",
    },
    # ── 2026 ──
    {
        "id": "evt_2026_01",
        "date": "2026-02-20",
        "description": "Trump signs executive orders: Temporary Import Surcharge + ending certain prior tariff actions",
        "type": "implementation",
        "countries": ["Global"],
        "market_reaction": {"sp500": -0.8, "dollar_index": +0.3},
        "tariff_landed": True,
        "outcome": "Restructured tariff framework — some old tariffs removed, replaced with broader surcharge.",
    },
    {
        "id": "evt_2026_02",
        "date": "2026-03-15",
        "description": "USTR launches 60 Section 301 investigations targeting forced labor in global supply chains",
        "type": "threat",
        "countries": ["China", "Global"],
        "market_reaction": {"sp500": -0.5, "dollar_index": +0.2},
        "tariff_landed": False,
        "outcome": "Investigations launched but no tariffs yet imposed. Signal of potential future escalation.",
    },
]


# ============================================================
# ChromaDB 存储和检索
# ============================================================

def get_events_collection():
    return get_client().get_or_create_collection("tariff_events")


def load_events_to_db():
    """将历史事件加载到 ChromaDB（幂等，跳过已存在的）"""
    col = get_events_collection()
    existing_ids = set(col.get()["ids"]) if col.count() > 0 else set()

    documents = []
    metadatas = []
    ids = []

    for evt in TARIFF_EVENTS:
        if evt["id"] in existing_ids:
            continue

        doc = f"[{evt['date']}] {evt['description']}. Outcome: {evt['outcome']}"

        meta = {
            "date": evt["date"],
            "type": evt["type"],
            "countries": ", ".join(evt["countries"]),
            "sp500_pct": str(evt["market_reaction"]["sp500"]),
            "dollar_index_pct": str(evt["market_reaction"]["dollar_index"]),
            "tariff_landed": str(evt["tariff_landed"]),
            "description": evt["description"],
            "outcome": evt["outcome"],
        }

        documents.append(doc)
        metadatas.append(meta)
        ids.append(evt["id"])

    if documents:
        col.add(documents=documents, metadatas=metadatas, ids=ids)

    return len(documents)


def find_similar_events(description: str, n_results: int = 3) -> list[dict]:
    """
    根据描述查找最相似的历史关税事件

    Args:
        description: 当前新闻/事件描述
        n_results: 返回结果数

    Returns:
        list[dict]: 每条包含 date, description, type, countries, market_reaction, outcome, distance
    """
    # 确保数据已加载
    col = get_events_collection()
    if col.count() == 0:
        load_events_to_db()

    results = col.query(
        query_texts=[description],
        n_results=min(n_results, col.count()),
    )

    items = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        items.append({
            "date": meta.get("date", ""),
            "description": meta.get("description", ""),
            "type": meta.get("type", ""),
            "countries": meta.get("countries", ""),
            "sp500_pct": float(meta.get("sp500_pct", 0)),
            "dollar_index_pct": float(meta.get("dollar_index_pct", 0)),
            "tariff_landed": meta.get("tariff_landed", "") == "True",
            "outcome": meta.get("outcome", ""),
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })

    return items


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    # 加载事件到 DB
    added = load_events_to_db()
    col = get_events_collection()
    print(f"事件库: {col.count()} 条 (新增 {added} 条)\n")

    # 测试几个当前新闻的检索
    test_queries = [
        "Trump threatens new tariffs on Chinese imports over trade imbalance",
        "US and EU negotiate trade agreement to reduce tariffs",
        "Markets crash as sweeping tariffs take effect globally",
        "90-day tariff pause announced to allow negotiations",
        "Section 301 investigation launched against forced labor",
    ]

    for q in test_queries:
        print(f"查询: {q[:60]}...")
        results = find_similar_events(q, n_results=3)
        for r in results:
            landed = "落地" if r["tariff_landed"] else "未落地"
            print(f"  [{r['date']}] {r['description'][:55]}")
            print(f"    类型: {r['type']}  国家: {r['countries']}  S&P500: {r['sp500_pct']:+.1f}%  {landed}")
            print(f"    距离: {r['distance']:.4f}")
        print()
