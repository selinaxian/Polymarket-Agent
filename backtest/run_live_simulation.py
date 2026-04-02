"""
活跃合约动态模拟回测（ReAct 全流程，无 look-ahead bias）
=====================================================
对活跃关税/贸易合约，用历史价格 + ReAct agent 搜索历史新闻做模拟交易。
每轮分析时 search_news tool 自动限制日期，只能看到分析日之前的新闻。

用法：
  python -m backtest.run_live_simulation
  python -m backtest.run_live_simulation --limit 3 --months 1 --interval 7
"""

import json
import time
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.polymarket_api import find_tariff_contracts, get_price_history
from config import invoke_with_retry

EDGE_THRESHOLD = 10.0
BET_SIZE = 100.0
PROFIT_TAKE_MOVE = 0.15
SIM_START = datetime(2026, 3, 4)  # default: 1 month ago
SIM_INTERVAL_DAYS = 7


# ============================================================
# 数据准备
# ============================================================

def load_contracts_with_history(limit: int | None = None) -> list[dict]:
    """获取活跃关税合约 + 历史价格。"""
    print("搜索活跃关税/贸易合约...")
    raw = find_tariff_contracts()
    seen = set()
    unique = [c for c in raw if not (c["market_id"] in seen or seen.add(c["market_id"]))]

    results = []
    for c in unique:
        tid = c.get("token_ids", {}).get("yes", "")
        if not tid or len(tid) < 10:
            continue

        vol = float(c.get("volume", 0) or 0)
        try:
            data = get_price_history(tid, interval="all", fidelity=1440)
            hist = data.get("history", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        except Exception:
            hist = []

        if not hist:
            continue

        start_ts = int(SIM_START.timestamp())
        recent = [h for h in hist if h["t"] >= start_ts]
        if len(recent) < 3:
            continue

        results.append({
            "question": c["question"], "market_id": c["market_id"],
            "event_title": c["event_title"], "token_id": tid,
            "volume": vol, "price_history": recent,
            "current_price": recent[-1]["p"],
        })

    results.sort(key=lambda x: x["volume"], reverse=True)
    if limit:
        results = results[:limit]

    print(f"  {len(results)} 个合约有足够历史数据")
    return results


def get_price_at_date(history: list[dict], target_date: datetime) -> float | None:
    target_ts = int(target_date.timestamp())
    best = None
    for h in history:
        if h["t"] <= target_ts + 86400:
            best = h["p"]
    return best


# ============================================================
# ReAct 全流程模拟回测
# ============================================================

def run_live_simulation(contracts: list[dict] | None = None, limit: int | None = None) -> dict:
    """
    用完整 ReAct agent 流程对活跃合约做动态模拟回测。
    """
    if contracts is None:
        contracts = load_contracts_with_history(limit=limit)

    if not contracts:
        print("无可用合约")
        return {"contracts": [], "trades": [], "summary": {}}

    from agents.tariff_agent import create_backtest_react_agent, create_tariff_decision_chain
    from agents.workflow import _extract_json_from_text

    decision_chain = create_tariff_decision_chain()

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    analysis_dates = []
    d = SIM_START
    while d < today:
        analysis_dates.append(d)
        d += timedelta(days=SIM_INTERVAL_DAYS)

    print(f"\n模拟回测（ReAct 全流程）: {len(contracts)} 个合约 x {len(analysis_dates)} 轮")
    print(f"时间: {SIM_START.strftime('%Y-%m-%d')} ~ {today.strftime('%Y-%m-%d')}, 每 {SIM_INTERVAL_DAYS} 天")
    print(f"规则: edge>{EDGE_THRESHOLD}% 买入 | 反转止损 | 移动>{PROFIT_TAKE_MOVE:.0%} 锁利\n")

    all_trades = []

    for ci, contract in enumerate(contracts):
        question = contract["question"]
        history = contract["price_history"]
        current_price = contract["current_price"]

        print(f"{'='*60}")
        print(f"[{ci+1}/{len(contracts)}] {question[:55]}")
        print(f"  当前: {current_price:.1%}  |  交易量: ${contract['volume']:,.0f}")

        position = None
        contract_trades = []

        for analysis_date in analysis_dates:
            date_str = analysis_date.strftime("%Y-%m-%d")

            price_at_date = get_price_at_date(history, analysis_date)
            if price_at_date is None:
                continue

            # ReAct agent with date-limited search
            try:
                react_agent = create_backtest_react_agent(before_date=date_str)
                react_result = react_agent.invoke({
                    "messages": [{
                        "role": "user",
                        "content": (
                            f"分析日期: {date_str}\n"
                            f"合约问题: {question}\n"
                            f"当前 Yes 赔率: {price_at_date:.1%}\n\n"
                            f"请搜索相关新闻和历史类比，然后输出分析 JSON。"
                        ),
                    }],
                })

                raw_content = react_result["messages"][-1].content
                if isinstance(raw_content, list):
                    final_msg = "\n".join(
                        part.get("text", str(part)) if isinstance(part, dict) else str(part)
                        for part in raw_content
                    )
                else:
                    final_msg = str(raw_content)

                analysis = _extract_json_from_text(final_msg)
                if not analysis:
                    print(f"    [{date_str}] JSON 解析失败")
                    time.sleep(2)
                    continue

                decision = invoke_with_retry(decision_chain, {
                    "tariff_analysis": json.dumps(analysis, ensure_ascii=False, indent=2),
                    "contract_question": question,
                    "yes_price": price_at_date,
                    "no_price": 1 - price_at_date,
                    "volume": str(contract["volume"]),
                })
            except Exception as e:
                print(f"    [{date_str}] 分析失败: {str(e)[:50]}")
                time.sleep(3)
                continue

            rec = decision.get("recommendation", "NO_TRADE")
            edge = decision.get("edge", 0)
            est_prob = decision.get("our_estimated_probability", 0)

            # ── Trading logic ──
            action = None

            if position is None:
                if rec == "BUY_YES" and abs(edge) >= EDGE_THRESHOLD:
                    position = {"direction": "YES", "entry_price": price_at_date, "entry_date": date_str}
                    action = "BUY_YES"
                elif rec == "BUY_NO" and abs(edge) >= EDGE_THRESHOLD:
                    position = {"direction": "NO", "entry_price": price_at_date, "entry_date": date_str}
                    action = "BUY_NO"
            else:
                entry_p = position["entry_price"]

                # Reversal exit
                if position["direction"] == "YES" and rec == "BUY_NO" and abs(edge) >= EDGE_THRESHOLD:
                    pnl = BET_SIZE * (price_at_date / entry_p - 1)
                    contract_trades.append({
                        "date": date_str, "action": "SELL_YES (反转)",
                        "price": price_at_date, "pnl": round(pnl, 2),
                        "reason": f"edge 反转 -> BUY_NO (edge {edge:+.1f}%)",
                    })
                    position = None
                elif position["direction"] == "NO" and rec == "BUY_YES" and abs(edge) >= EDGE_THRESHOLD:
                    no_entry, no_now = 1 - entry_p, 1 - price_at_date
                    pnl = BET_SIZE * (no_now / no_entry - 1)
                    contract_trades.append({
                        "date": date_str, "action": "SELL_NO (反转)",
                        "price": price_at_date, "pnl": round(pnl, 2),
                        "reason": f"edge 反转 -> BUY_YES (edge {edge:+.1f}%)",
                    })
                    position = None

                # Profit taking
                elif position["direction"] == "YES" and price_at_date - entry_p >= PROFIT_TAKE_MOVE:
                    pnl = BET_SIZE * (price_at_date / entry_p - 1)
                    contract_trades.append({
                        "date": date_str, "action": "SELL_YES (锁利)",
                        "price": price_at_date, "pnl": round(pnl, 2),
                        "reason": f"赔率 {entry_p:.1%} -> {price_at_date:.1%}",
                    })
                    position = None
                elif position["direction"] == "NO" and entry_p - price_at_date >= PROFIT_TAKE_MOVE:
                    no_entry, no_now = 1 - entry_p, 1 - price_at_date
                    pnl = BET_SIZE * (no_now / no_entry - 1)
                    contract_trades.append({
                        "date": date_str, "action": "SELL_NO (锁利)",
                        "price": price_at_date, "pnl": round(pnl, 2),
                        "reason": f"赔率 {entry_p:.1%} -> {price_at_date:.1%}",
                    })
                    position = None

            if action in ("BUY_YES", "BUY_NO"):
                contract_trades.append({
                    "date": date_str, "action": action,
                    "price": price_at_date, "pnl": 0,
                    "reason": f"AI: {rec} (prob {est_prob}%, edge {edge:+.1f}%)",
                })

            status = action or ("HOLD" if position else "NO_POS")
            print(f"    [{date_str}] Yes={price_at_date:.1%}  AI={rec} edge={edge:+.1f}%  -> {status}")
            time.sleep(3)

        # Mark-to-market
        if position:
            entry_p = position["entry_price"]
            if position["direction"] == "YES":
                mtm = BET_SIZE * (current_price / entry_p - 1)
            else:
                mtm = BET_SIZE * ((1 - current_price) / (1 - entry_p) - 1)

            contract_trades.append({
                "date": today.strftime("%Y-%m-%d"),
                "action": f"MTM_{position['direction']} (浮动)",
                "price": current_price, "pnl": round(mtm, 2),
                "reason": f"持仓 {position['direction']} from {position['entry_date']}",
            })
            print(f"    [今天] 浮动 PnL: ${mtm:+.0f}")

        for t in contract_trades:
            t["question"] = question[:50]
        all_trades.extend(contract_trades)

        total_pnl = sum(t["pnl"] for t in contract_trades)
        print(f"  合约 PnL: ${total_pnl:+.0f} ({len(contract_trades)} 笔)")

    summary = _compute_summary(all_trades)
    return {
        "contracts": [{"question": c["question"][:50], "volume": c["volume"],
                        "current_price": c["current_price"]} for c in contracts],
        "trades": all_trades, "summary": summary,
    }


def _compute_summary(trades):
    pnl_events = [t for t in trades if t["pnl"] != 0]
    pnl_vals = [t["pnl"] for t in pnl_events]

    if not pnl_vals:
        return {"total_trades": len(trades), "cumulative_pnl": 0, "max_drawdown": 0,
                "sharpe": 0, "win_rate": 0, "buy_entries": 0, "pnl_events": 0}

    cum, peak, max_dd = 0, 0, 0
    for p in pnl_vals:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    mean = sum(pnl_vals) / len(pnl_vals)
    std = (sum((p - mean) ** 2 for p in pnl_vals) / len(pnl_vals)) ** 0.5

    return {
        "total_trades": len(trades),
        "buy_entries": sum(1 for t in trades if t["action"].startswith("BUY")),
        "pnl_events": len(pnl_events),
        "cumulative_pnl": round(cum, 2), "max_drawdown": round(max_dd, 2),
        "sharpe": round(mean / std if std > 0 else 0, 2),
        "win_rate": round(sum(1 for p in pnl_vals if p > 0) / len(pnl_vals) * 100, 1),
    }


def print_simulation_report(result):
    s = result["summary"]
    print(f"\n{'='*60}")
    print(f"  模拟回测报告（ReAct 全流程）")
    print(f"{'='*60}\n")
    print(f"  合约数: {len(result['contracts'])}")
    print(f"  交易笔数: {s.get('total_trades', 0)}")
    print(f"  PnL: ${s.get('cumulative_pnl', 0):+.0f}")
    print(f"  Sharpe: {s.get('sharpe', 0):.2f}")
    print(f"  最大回撤: ${s.get('max_drawdown', 0):.0f}")
    print(f"  胜率: {s.get('win_rate', 0):.0f}%")

    for t in result["trades"]:
        pnl_str = f"${t['pnl']:+.0f}" if t["pnl"] != 0 else ""
        print(f"\n  [{t['date']}] {t['action']}  {t['question']}")
        print(f"    价格: {t['price']:.1%}  {pnl_str}  {t['reason'][:60]}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--months", type=int, default=1)
    parser.add_argument("--interval", type=int, default=7)
    args = parser.parse_args()

    SIM_START = datetime.now() - timedelta(days=args.months * 30)
    SIM_INTERVAL_DAYS = args.interval

    result = run_live_simulation(limit=args.limit)
    print_simulation_report(result)
