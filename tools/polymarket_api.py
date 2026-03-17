"""
Polymarket 数据接口
功能：从 Polymarket API 获取真实的市场数据、合约信息和价格
"""

import json
import requests


# API 基础地址
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def _parse_clob_ids(raw) -> tuple[str, str]:
    """Parse clobTokenIds which may be a JSON string or list."""
    if not raw:
        return ("", "")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return ("", "")
    else:
        parsed = raw
    if isinstance(parsed, list) and len(parsed) >= 2:
        return (str(parsed[0]), str(parsed[1]))
    if isinstance(parsed, list) and len(parsed) == 1:
        return (str(parsed[0]), "")
    return ("", "")


def search_markets(query: str, limit: int = 10) -> list:
    """
    搜索 Polymarket 上的市场/合约

    Gamma API 不支持服务端文本搜索，所有搜索参数都会被忽略。
    因此采用分页拉取 + 客户端关键词过滤的方式。

    Args:
        query: 搜索关键词，比如 "Fed rate" 或 "interest rate"
        limit: 返回结果数量

    Returns:
        list: 匹配的事件列表（每个事件包含 markets 子列表）
    """
    keywords = [k.lower() for k in query.split("|")] if "|" in query else [query.lower()]
    matched_events = []
    offset = 0
    max_pages = 10  # 最多扫描 500 个事件

    while offset < max_pages * 50 and len(matched_events) < limit:
        resp = requests.get(f"{GAMMA_BASE}/events", params={
            "limit": 50,
            "offset": offset,
            "active": True,
            "closed": False,
        })
        resp.raise_for_status()
        events = resp.json()
        if not events:
            break

        for event in events:
            title = event.get("title", "").lower()
            desc = (event.get("description") or "").lower()
            # 也检查子市场的 question
            market_text = " ".join(
                m.get("question", "").lower() for m in event.get("markets", [])
            )
            combined = f"{title} {desc} {market_text}"
            if any(k in combined for k in keywords):
                matched_events.append(event)

        offset += 50

    return matched_events[:limit]


def get_price_history(token_id: str, interval: str = "1d", fidelity: int = 60) -> list:
    """
    获取价格历史数据
    
    Args:
        token_id: 合约的 token ID
        interval: 时间范围 ("1d", "1w", "1m", "3m", "all")
        fidelity: 数据精度（分钟）
    
    Returns:
        list: 历史价格数据
    """
    url = f"{CLOB_BASE}/prices-history"
    params = {
        "market": token_id,
        "interval": interval,
        "fidelity": fidelity,
    }
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


# ============================================================
# 便捷函数：搜索关税/贸易相关合约
# ============================================================

def find_tariff_contracts(keyword: str = "tariff|trade war|trade deal|import duty|usmca|section 301|trade policy|trade agreement") -> list:
    """
    搜索所有关税/贸易相关的活跃合约，返回格式化的列表

    Args:
        keyword: 搜索关键词，用 "|" 分隔多个关键词（OR 匹配）

    Returns:
        list[dict]: 格式化的合约信息
    """
    events = search_markets(keyword, limit=50)

    results = []
    for event in events:
        title = event.get("title", "")
        markets = event.get("markets", [])

        for market in markets:
            results.append({
                "event_title": title,
                "question": market.get("question", ""),
                "market_id": market.get("id", ""),
                "condition_id": market.get("conditionId", ""),
                "yes_price": market.get("outcomePrices", ""),
                "volume": market.get("volume", ""),
                "liquidity": market.get("liquidity", ""),
                "end_date": market.get("endDate", ""),
                "description": market.get("description", "")[:200],
                "token_ids": {
                    "yes": _parse_clob_ids(market.get("clobTokenIds", ""))[0],
                    "no": _parse_clob_ids(market.get("clobTokenIds", ""))[1],
                },
            })

    return results


# ============================================================
# 测试：直接运行此文件看效果
# ============================================================

if __name__ == "__main__":
    import json

    contracts = find_tariff_contracts()
    seen = set()
    unique = [c for c in contracts if not (c["market_id"] in seen or seen.add(c["market_id"]))]
    print(f"关税/贸易合约: 共 {len(unique)} 个\n")
    for i, c in enumerate(unique[:10], 1):
        print(f"  {i}. [{c['event_title']}]")
        print(f"     {c['question']}")
        print(f"     Yes: {c['yes_price']}  交易量: {c['volume']}")
        print()
