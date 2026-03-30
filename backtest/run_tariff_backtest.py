"""
已结算合约回测（ReAct 全流程，无 look-ahead bias）
=================================================
用完整的 ReAct agent 流程分析已结算合约。
search_news tool 自动加日期限制，只搜 T-1 之前的新闻。

用法：
  python -m backtest.run_tariff_backtest
  python -m backtest.run_tariff_backtest --limit 3
  python -m backtest.run_tariff_backtest --days 7
"""

import json
import time
import sys
import os
import re
import requests
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
from html import unescape

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.tariff_history import TARIFF_HISTORY
from config import invoke_with_retry

EDGE_THRESHOLD = 10.0
BET_SIZE = 100.0


# ============================================================
# 历史新闻获取（给 backtest ReAct agent 的 tool 用）
# ============================================================

def fetch_historical_news(query: str, before_date: str, lookback_days: int = 14, limit: int = 10) -> list[dict]:
    """从 Google News RSS 获取 before_date 之前的历史新闻。"""
    dt_before = datetime.strptime(before_date, "%Y-%m-%d")
    dt_after = dt_before - timedelta(days=lookback_days)
    after_str = dt_after.strftime("%Y-%m-%d")

    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={requests.utils.quote(query + f' after:{after_str} before:{before_date}')}"
        f"&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        resp = requests.get(rss_url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }, timeout=15)
        resp.raise_for_status()
    except Exception:
        return []

    root = ET.fromstring(resp.content)
    results = []
    for item in root.findall(".//item"):
        title = item.findtext("title", "")
        pub_date_raw = item.findtext("pubDate", "")
        source_el = item.find("source")
        source_name = source_el.text if source_el is not None else ""
        description = item.findtext("description", "")
        summary = unescape(re.sub(r"<[^>]+>", "", description)).strip()

        parsed_date = ""
        try:
            dt = datetime.strptime(pub_date_raw.strip(), "%a, %d %b %Y %H:%M:%S %Z")
            parsed_date = dt.strftime("%Y-%m-%d")
        except Exception:
            try:
                dt = datetime.strptime(pub_date_raw.strip()[:25], "%a, %d %b %Y %H:%M:%S")
                parsed_date = dt.strftime("%Y-%m-%d")
            except Exception:
                parsed_date = pub_date_raw

        results.append({
            "date": parsed_date, "title": title,
            "source": source_name, "summary": summary[:300],
        })
        if len(results) >= limit:
            break

    return results


# ============================================================
# ReAct 全流程回测
# ============================================================

def run_tariff_backtest(cases: list[dict] | None = None, limit: int | None = None, days_before: int = 1):
    """
    用完整 ReAct agent 流程回测已结算合约。

    Args:
        cases: 回测合约列表（默认 TARIFF_HISTORY）
        limit: 只跑前 N 个
        days_before: 结算前几天分析（默认 1 = T-1）
    """
    if cases is None:
        cases = TARIFF_HISTORY
    if limit:
        cases = cases[:limit]

    from agents.tariff_agent import (
        create_backtest_react_agent,
        create_tariff_decision_chain,
        score_news_sentiment,
        aggregate_sentiment,
    )
    from agents.workflow import _extract_json_from_text

    decision_chain = create_tariff_decision_chain()
    results = []
    total = len(cases)
    label = f"T-{days_before}"

    print(f"回测 {total} 个已结算合约（ReAct 全流程，{label}）\n")

    for i, case in enumerate(cases, 1):
        question = case["question"]
        actual = case["outcome"]
        resolved_date = case["resolved_date"]
        yes_price = case["yes_price_before"]

        dt_resolved = datetime.strptime(resolved_date, "%Y-%m-%d")
        dt_analysis = dt_resolved - timedelta(days=days_before)
        analysis_date = dt_analysis.strftime("%Y-%m-%d")

        print(f"{'='*60}")
        print(f"[{i}/{total}] {question[:55]}")
        print(f"  结算日: {resolved_date}  |  分析日: {analysis_date} ({label})")
        print(f"  实际结果: {actual}  |  {label} Yes: {yes_price:.0%}")

        # Step 1: ReAct agent with date-limited search
        print(f"  ReAct agent 搜索 {analysis_date} 之前的新闻...")
        react_agent = create_backtest_react_agent(before_date=analysis_date)

        try:
            react_result = react_agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": (
                        f"分析日期: {analysis_date}\n"
                        f"合约问题: {question}\n\n"
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

            tool_count = sum(1 for m in react_result["messages"] if hasattr(m, "type") and m.type == "tool")
            print(f"  工具调用: {tool_count} 次")

            analysis = _extract_json_from_text(final_msg)
            if not analysis:
                print(f"  JSON 解析失败，跳过")
                results.append(_error_result(case, analysis_date, "parse_error"))
                time.sleep(2)
                continue

        except Exception as e:
            print(f"  ReAct 失败: {str(e)[:60]}")
            results.append(_error_result(case, analysis_date, "react_error"))
            time.sleep(2)
            continue

        # Step 2: Decision
        try:
            decision = invoke_with_retry(decision_chain, {
                "tariff_analysis": json.dumps(analysis, ensure_ascii=False, indent=2),
                "contract_question": question,
                "yes_price": yes_price,
                "no_price": 1 - yes_price,
                "volume": str(case.get("volume", "N/A")),
            })
        except Exception as e:
            print(f"  决策失败: {e}")
            results.append(_error_result(case, analysis_date, "decision_error"))
            time.sleep(2)
            continue

        # Step 3: Evaluate
        rec = decision.get("recommendation", "NO_TRADE")
        edge = decision.get("edge", 0)
        est_prob = decision.get("our_estimated_probability", 0)
        confidence = decision.get("confidence", 0)

        triggered = abs(edge) >= EDGE_THRESHOLD and rec != "NO_TRADE"
        predicted = "Yes" if rec == "BUY_YES" else ("No" if rec == "BUY_NO" else None)
        correct = predicted == actual if predicted else None
        pnl = _compute_trade_pnl(rec, yes_price, actual) if triggered else 0.0

        result = {
            "question": question, "resolved_date": resolved_date,
            "analysis_date": analysis_date, "actual_outcome": actual,
            "yes_price_before": yes_price, "event_type": case.get("event_type", ""),
            "recommendation": rec, "estimated_prob": est_prob,
            "edge": edge, "confidence": confidence,
            "signal_triggered": triggered, "direction_correct": correct,
            "pnl": round(pnl, 2), "status": "ok",
            "queries_used": analysis.get("search_queries_used", []),
        }
        results.append(result)

        mark = f"{'OK' if correct else 'MISS'} | PnL: ${pnl:+.0f}" if triggered else "SKIP"
        print(f"  AI: {rec} (prob {est_prob}%, edge {edge:+.1f}%) [{mark}]")
        time.sleep(3)

    return results


def _error_result(case, analysis_date, status):
    return {
        "question": case["question"], "resolved_date": case["resolved_date"],
        "analysis_date": analysis_date, "actual_outcome": case["outcome"],
        "yes_price_before": case["yes_price_before"], "event_type": case.get("event_type", ""),
        "recommendation": "ERROR", "estimated_prob": 0, "edge": 0, "confidence": 0,
        "signal_triggered": False, "direction_correct": None, "pnl": 0, "status": status,
    }


def _compute_trade_pnl(rec, yes_price, actual):
    if rec == "BUY_YES":
        return BET_SIZE * (1.0 / yes_price - 1) if actual == "Yes" else -BET_SIZE
    elif rec == "BUY_NO":
        no_price = 1 - yes_price
        return BET_SIZE * (1.0 / no_price - 1) if actual == "No" else -BET_SIZE
    return 0.0


# ============================================================
# 统计
# ============================================================

def compute_pnl(results):
    ok = [r for r in results if r.get("status") == "ok"]
    triggered = [r for r in ok if r["signal_triggered"]]

    trades = [{
        "question": r["question"][:40], "date": r["resolved_date"],
        "rec": r["recommendation"], "actual": r["actual_outcome"],
        "edge": r["edge"], "pnl": r["pnl"],
    } for r in triggered]

    if not trades:
        return {"trades": [], "cumulative_pnl": 0, "cumulative_series": [],
                "max_drawdown": 0, "sharpe": 0, "total_return_pct": 0,
                "total_invested": 0, "num_trades": 0, "win_rate": 0}

    cum, peak, max_dd = 0, 0, 0
    cum_series = []
    for t in trades:
        cum += t["pnl"]
        cum_series.append(round(cum, 2))
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    pnl_vals = [t["pnl"] for t in trades]
    mean = sum(pnl_vals) / len(pnl_vals)
    std = (sum((p - mean) ** 2 for p in pnl_vals) / len(pnl_vals)) ** 0.5
    sharpe = mean / std if std > 0 else 0

    return {
        "trades": trades, "cumulative_pnl": round(cum, 2),
        "cumulative_series": cum_series, "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "total_return_pct": round(cum / (BET_SIZE * len(trades)) * 100, 1),
        "total_invested": BET_SIZE * len(trades),
        "num_trades": len(trades),
        "win_rate": round(sum(1 for p in pnl_vals if p > 0) / len(pnl_vals) * 100, 1),
    }


def threshold_sensitivity(results, thresholds=None):
    if thresholds is None:
        thresholds = [10.0, 15.0, 20.0, 25.0]
    return [_pnl_at_threshold(results, t) for t in thresholds]


def _pnl_at_threshold(results, threshold):
    ok = [r for r in results if r.get("status") == "ok"]
    triggered = []
    for r in ok:
        if r["recommendation"] in ("BUY_YES", "BUY_NO") and abs(r["edge"]) >= threshold:
            pnl = _compute_trade_pnl(r["recommendation"], r["yes_price_before"], r["actual_outcome"])
            predicted = "Yes" if r["recommendation"] == "BUY_YES" else "No"
            triggered.append({"pnl": round(pnl, 2), "correct": predicted == r["actual_outcome"]})

    if not triggered:
        return {"threshold": threshold, "signals": 0, "correct": 0, "accuracy": 0,
                "pnl": 0, "max_drawdown": 0, "sharpe": 0, "win_rate": 0}

    pnl_vals = [t["pnl"] for t in triggered]
    correct = sum(1 for t in triggered if t["correct"])
    cum, peak, max_dd = 0, 0, 0
    for p in pnl_vals:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    mean = sum(pnl_vals) / len(pnl_vals)
    std = (sum((p - mean) ** 2 for p in pnl_vals) / len(pnl_vals)) ** 0.5

    return {
        "threshold": threshold, "signals": len(triggered), "correct": correct,
        "accuracy": round(correct / len(triggered) * 100, 1),
        "pnl": round(cum, 2), "max_drawdown": round(max_dd, 2),
        "sharpe": round(mean / std if std > 0 else 0, 2),
        "win_rate": round(sum(1 for p in pnl_vals if p > 0) / len(pnl_vals) * 100, 1),
    }


def print_backtest_report(results):
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        print("\n没有有效结果。")
        return

    triggered = [r for r in ok if r["signal_triggered"]]
    correct = [r for r in triggered if r["direction_correct"]]

    print(f"\n{'='*60}")
    print(f"  回测报告（ReAct 全流程）")
    print(f"{'='*60}\n")
    print(f"  合约数: {len(ok)}")
    print(f"  信号触发: {len(triggered)} (edge >= {EDGE_THRESHOLD}%)")
    if triggered:
        print(f"  准确率: {len(correct)}/{len(triggered)} ({len(correct)/len(triggered)*100:.0f}%)")

    pnl = compute_pnl(results)
    if pnl["trades"]:
        print(f"\n  PnL: ${pnl['cumulative_pnl']:+.0f}")
        print(f"  Sharpe: {pnl['sharpe']:.2f}")
        print(f"  最大回撤: ${pnl['max_drawdown']:.0f}")
        print(f"  胜率: {pnl['win_rate']:.0f}%")

    print(f"\n  逐条:")
    for r in ok:
        mark = ("OK" if r["direction_correct"] else "MISS") if r["signal_triggered"] else "SKIP"
        pnl_str = f"${r['pnl']:+.0f}" if r["signal_triggered"] else ""
        print(f"  [{r['analysis_date']}] {r['question'][:45]}  {r['recommendation']}  edge={r['edge']:+.1f}%  {mark} {pnl_str}")
        if r.get("queries_used"):
            print(f"    搜索词: {r['queries_used']}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--days", type=int, default=1)
    args = parser.parse_args()

    results = run_tariff_backtest(limit=args.limit, days_before=args.days)
    print_backtest_report(results)
