"""
Polymarket 关税/贸易政策监控系统
=================================
持续监控关税/贸易相关合约，自动分析新闻并生成交易信号。

用法：
  python monitor.py                  # 交互式选择 watchlist 并开始监控
  python monitor.py --once           # 只跑一轮分析（测试用）
  python monitor.py --list           # 列出所有可用合约
"""

import json
import os
import time
import re
import argparse
from datetime import datetime

from tools.polymarket_api import find_tariff_contracts
from tools.news_scraper import get_all_tariff_news

# ============================================================
# 数据目录
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.json")
SIGNALS_FILE = os.path.join(DATA_DIR, "signals.json")
ANALYSIS_LOG_FILE = os.path.join(DATA_DIR, "analysis_log.json")

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# 辅助函数
# ============================================================

def parse_prices(contract: dict) -> tuple[float, float]:
    raw = contract.get("yes_price", "")
    if isinstance(raw, list) and len(raw) >= 2:
        return float(raw[0]), float(raw[1])
    if isinstance(raw, str) and raw.startswith("["):
        try:
            prices = json.loads(raw.replace("'", '"'))
            return float(prices[0]), float(prices[1])
        except Exception:
            pass
    return 0.5, 0.5


def load_json(path: str) -> list:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 合约发现
# ============================================================

MIN_VOLUME = 20_000   # 最低交易量阈值（$）
EDGE_THRESHOLD = 8.0   # 触发信号的最小 edge（百分点）


def discover_contracts() -> list[dict]:
    """搜索关税/贸易相关合约，过滤交易量 > MIN_VOLUME"""
    print(f"  搜索关税/贸易合约 (vol > ${MIN_VOLUME:,})...")
    raw = find_tariff_contracts()

    seen = set()
    unique = [c for c in raw if not (c["market_id"] in seen or seen.add(c["market_id"]))]

    filtered = [c for c in unique if float(c.get("volume", 0) or 0) > MIN_VOLUME]
    print(f"  {len(unique)} 个合约中 {len(filtered)} 个交易量达标")
    return filtered


# ============================================================
# Watchlist 管理
# ============================================================

def load_watchlist() -> list[dict]:
    return load_json(WATCHLIST_FILE)


def save_watchlist(watchlist: list[dict]):
    save_json(WATCHLIST_FILE, watchlist)


def interactive_select_watchlist(contracts: list[dict]) -> list[dict]:
    """交互式选择合约加入 watchlist"""
    print(f"\n找到 {len(contracts)} 个可用合约:\n")

    # 按事件分组显示
    events = {}
    for c in contracts:
        events.setdefault(c["event_title"], []).append(c)

    idx = 1
    index_map = {}
    for event_title, markets in events.items():
        print(f"  [{event_title}]")
        for m in markets:
            yp, np_ = parse_prices(m)
            print(f"    {idx:3d}. {m['question'][:65]}")
            print(f"         Yes: {yp:.1%}  |  交易量: ${float(m.get('volume', 0)):,.0f}")
            index_map[idx] = m
            idx += 1
        print()

    # 获取用户选择
    existing = load_watchlist()
    existing_ids = {c["market_id"] for c in existing}

    print("输入合约编号加入 watchlist（逗号分隔，如 1,3,5）")
    print(f"当前 watchlist: {len(existing)} 个合约")
    if existing:
        for c in existing:
            print(f"  - {c['question'][:60]}")

    print("\n输入 'all' 全选, 'keep' 保留现有, 'clear' 清空:")
    choice = input("> ").strip()

    if choice.lower() == "keep" and existing:
        return existing
    elif choice.lower() == "clear":
        return []
    elif choice.lower() == "all":
        selected = list(index_map.values())
    else:
        nums = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
        selected = [index_map[n] for n in nums if n in index_map]

    # 合并现有 watchlist
    for c in selected:
        if c["market_id"] not in existing_ids:
            existing.append(c)
            existing_ids.add(c["market_id"])

    save_watchlist(existing)
    print(f"\nWatchlist 已更新: {len(existing)} 个合约")
    return existing


# ============================================================
# 分析循环
# ============================================================

def run_analysis_cycle(watchlist: list[dict], cycle_num: int = 0) -> list[dict]:
    """
    执行一轮完整的分析循环：
    1. 刷新合约赔率
    2. 抓取最新新闻
    3. LLM 分析
    4. 计算 edge，触发信号

    Returns:
        list[dict]: 本轮产生的信号
    """
    ts = timestamp()
    print(f"\n{'='*60}")
    print(f"  分析循环 #{cycle_num}  {ts}")
    print(f"{'='*60}")

    if not watchlist:
        print("  Watchlist 为空，跳过")
        return []

    # ── 1. 刷新合约赔率 ──
    print("\n[1/4] 刷新合约赔率...")
    refreshed = _refresh_prices(watchlist)

    # ── 2. 抓取新闻 ──
    print("\n[2/4] 抓取最新新闻...")
    news = _fetch_news()
    print(f"  获取 {len(news)} 条新闻")

    if not news:
        print("  无新闻可分析，跳过本轮")
        return []

    # ── 3. LLM 分析 ──
    print("\n[3/4] AI 分析新闻对合约的影响...")
    analysis = _analyze_news_for_contracts(news, refreshed)

    # ── 4. 计算 edge，生成信号 ──
    print("\n[4/4] 计算信号...")
    signals = _generate_signals(refreshed, analysis, ts)

    # 保存记录
    _save_analysis_log(ts, cycle_num, refreshed, news, analysis, signals)

    return signals


def _refresh_prices(watchlist: list[dict]) -> list[dict]:
    """用最新的 Polymarket 数据刷新 watchlist 合约"""
    fresh_tariff = find_tariff_contracts()
    all_fresh = {c["market_id"]: c for c in fresh_tariff}

    refreshed = []
    for c in watchlist:
        mid = c["market_id"]
        if mid in all_fresh:
            updated = all_fresh[mid]
            yp, np_ = parse_prices(updated)
            print(f"  {updated['question'][:50]}  Yes: {yp:.1%}")
            refreshed.append(updated)
        else:
            # 合约可能已关闭
            refreshed.append(c)

    return refreshed


def _fetch_news() -> list[dict]:
    """抓取最新关税/贸易新闻"""
    try:
        return get_all_tariff_news(limit_per_source=8)
    except Exception as e:
        print(f"  新闻抓取失败: {e}")
        return []


def _analyze_news_for_contracts(news: list[dict], contracts: list[dict]) -> dict:
    """用 LLM 分析新闻对 watchlist 合约的影响"""
    from agents.tariff_agent import create_tariff_analysis_chain

    # 格式化新闻
    news_lines = []
    for i, n in enumerate(news[:12], 1):
        news_lines.append(f"[{i}] [{n.get('date','')}] {n['title']} ({n.get('source','')})")
        if n.get("summary"):
            news_lines.append(f"    {n['summary'][:150]}")

    # 格式化合约
    contract_lines = []
    for c in contracts:
        yp, np_ = parse_prices(c)
        contract_lines.append(f"- {c['question']}  (Yes: {yp:.1%}, No: {np_:.1%})")

    combined_text = (
        "=== 最新新闻 ===\n"
        + "\n".join(news_lines)
        + "\n\n=== 监控中的 Polymarket 合约 ===\n"
        + "\n".join(contract_lines)
    )

    try:
        chain = create_tariff_analysis_chain()
        from config import invoke_with_retry
        result = invoke_with_retry(chain, {"news_text": combined_text[:8000]})
        print("  AI 分析完成")
        return result
    except Exception as e:
        print(f"  AI 分析失败: {e}")
        return {}


def _generate_signals(
    contracts: list[dict],
    analysis: dict,
    ts: str,
    edge_threshold: float = EDGE_THRESHOLD,
) -> list[dict]:
    """根据分析结果对每个合约计算 edge 并生成信号"""
    signals = []

    # 从分析中提取政策信号强度
    signal_strength = analysis.get("signal_strength", 0)
    policy_signal = analysis.get("policy_signal", "neutral")
    tariff_actions = analysis.get("tariff_actions", [])
    summary = analysis.get("summary", "")

    for c in contracts:
        yp, np_ = parse_prices(c)
        question = c["question"].lower()

        # 基于分析结果估算概率
        estimated_prob = _estimate_probability(question, yp, signal_strength, policy_signal, tariff_actions)

        if estimated_prob is None:
            continue

        edge = estimated_prob - yp * 100
        abs_edge = abs(edge)

        signal = {
            "timestamp": ts,
            "question": c["question"],
            "market_id": c["market_id"],
            "market_yes": yp,
            "market_no": np_,
            "estimated_prob": round(estimated_prob, 1),
            "edge": round(edge, 1),
            "policy_signal": policy_signal,
            "signal_strength": signal_strength,
            "summary": summary[:200],
        }

        if abs_edge >= edge_threshold:
            if edge > 0:
                signal["action"] = "BUY_YES"
                signal["alert"] = True
            else:
                signal["action"] = "BUY_NO"
                signal["alert"] = True
            print(f"\n  *** 交易信号 ***")
            print(f"  合约: {c['question'][:55]}")
            print(f"  市场: {yp:.1%}  |  AI估计: {estimated_prob:.1f}%  |  Edge: {edge:+.1f}%")
            print(f"  建议: {signal['action']}")
        else:
            signal["action"] = "NO_TRADE"
            signal["alert"] = False
            print(f"  {c['question'][:50]}  edge: {edge:+.1f}% (不触发)")

        signals.append(signal)

    # 保存信号
    existing_signals = load_json(SIGNALS_FILE)
    existing_signals.extend([s for s in signals if s.get("alert")])
    # 只保留最近 200 条
    save_json(SIGNALS_FILE, existing_signals[-200:])

    alert_count = sum(1 for s in signals if s.get("alert"))
    print(f"\n  本轮信号: {alert_count} 个触发 / {len(signals)} 个合约")

    return signals


def _estimate_probability(
    question: str,
    current_yes: float,
    signal_strength: int,
    policy_signal: str,
    tariff_actions: list,
) -> float | None:
    """
    基于 AI 分析结果估算合约概率。

    简单启发式方法：以市场价为基准，根据政策信号方向和强度调整。
    实际场景中应由 LLM 逐合约单独判断。
    """
    base = current_yes * 100  # 市场当前概率

    # 关税相关信号的方向性调整
    # signal_strength > 0 表示贸易保护/加税倾向
    # signal_strength < 0 表示贸易自由化/减税倾向
    adjustment = 0

    # 贸易协议相关合约
    if "trade deal" in question or "trade agreement" in question:
        if policy_signal in ("negotiation", "de-escalation"):
            adjustment = signal_strength * -1.5  # 谈判利好 -> 协议概率上升
        elif policy_signal in ("escalation", "threat"):
            adjustment = signal_strength * 1.5  # 升级 -> 协议概率下降

    # 关税相关合约
    elif "tariff" in question:
        if policy_signal in ("escalation", "implementation"):
            adjustment = signal_strength * 2  # 加税信号 -> 关税合约概率上升
        elif policy_signal in ("de-escalation",):
            adjustment = signal_strength * 2

    # Fed 利率相关
    elif any(k in question for k in ["fed", "rate cut", "interest rate", "fomc"]):
        # 贸易战升级 -> 经济放缓预期 -> 降息概率上升
        if "cut" in question or "decrease" in question:
            adjustment = signal_strength * 1.0
        elif "no" in question and "cut" in question:
            adjustment = signal_strength * -1.0

    # 其他合约
    else:
        return None

    estimated = max(1, min(99, base + adjustment))
    return estimated


# ============================================================
# 日志
# ============================================================

def _save_analysis_log(ts, cycle_num, contracts, news, analysis, signals):
    """保存分析日志"""
    log = load_json(ANALYSIS_LOG_FILE)
    log.append({
        "timestamp": ts,
        "cycle": cycle_num,
        "contracts_count": len(contracts),
        "news_count": len(news),
        "news_titles": [n["title"] for n in news[:10]],
        "analysis_summary": analysis.get("summary", ""),
        "policy_signal": analysis.get("policy_signal", ""),
        "signal_strength": analysis.get("signal_strength", 0),
        "alerts": [s for s in signals if s.get("alert")],
        "total_signals": len(signals),
    })
    # 只保留最近 100 轮
    save_json(ANALYSIS_LOG_FILE, log[-100:])


# ============================================================
# 主循环
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Polymarket 监控系统")
    parser.add_argument("--once", action="store_true", help="只跑一轮分析")
    parser.add_argument("--list", action="store_true", help="列出所有可用合约")
    parser.add_argument("--interval", type=int, default=30, help="分析间隔（分钟）")
    args = parser.parse_args()

    print("="*60)
    print("  Polymarket 预测市场监控系统")
    print("="*60)

    # 发现合约
    print("\n搜索可用合约...")
    contracts = discover_contracts()
    print(f"共发现 {len(contracts)} 个相关合约")

    if args.list:
        for i, c in enumerate(contracts, 1):
            yp, _ = parse_prices(c)
            print(f"  {i:3d}. [{c['event_title']}] {c['question'][:55]}  Yes: {yp:.1%}")
        return

    # 选择 watchlist
    existing_wl = load_watchlist()
    if existing_wl and not args.once:
        print(f"\n已有 watchlist ({len(existing_wl)} 个合约):")
        for c in existing_wl:
            print(f"  - {c['question'][:60]}")
        use_existing = input("\n使用现有 watchlist? (y/n): ").strip().lower()
        if use_existing == "y":
            watchlist = existing_wl
        else:
            watchlist = interactive_select_watchlist(contracts)
    elif args.once:
        # --once 模式：如果有现有 watchlist 就用，否则取前 5 个
        watchlist = existing_wl if existing_wl else contracts[:5]
        if not existing_wl:
            save_watchlist(watchlist)
            print(f"\n自动选择前 {len(watchlist)} 个合约作为 watchlist")
    else:
        watchlist = interactive_select_watchlist(contracts)

    if not watchlist:
        print("Watchlist 为空，退出")
        return

    # 运行分析
    if args.once:
        signals = run_analysis_cycle(watchlist, cycle_num=1)
        _print_signal_summary(signals)
        return

    # 持续监控
    print(f"\n开始持续监控 (间隔 {args.interval} 分钟, Ctrl+C 停止)\n")
    cycle = 0
    while True:
        cycle += 1
        try:
            signals = run_analysis_cycle(watchlist, cycle_num=cycle)
            _print_signal_summary(signals)
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            break
        except Exception as e:
            print(f"\n  本轮出错: {e}")

        # 等待下一轮
        next_time = datetime.now().strftime("%H:%M")
        print(f"\n  下一轮: {args.interval} 分钟后 (当前 {next_time})")
        try:
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            break


def _print_signal_summary(signals: list[dict]):
    """打印本轮信号摘要"""
    alerts = [s for s in signals if s.get("alert")]
    if not alerts:
        print("\n  本轮无交易信号触发")
        return

    print(f"\n{'='*60}")
    print(f"  交易信号汇总 ({len(alerts)} 个)")
    print(f"{'='*60}")
    for s in alerts:
        print(f"\n  {s['action']}  {s['question'][:55]}")
        print(f"  市场: {s['market_yes']:.1%}  →  AI估计: {s['estimated_prob']:.1f}%  |  Edge: {s['edge']:+.1f}%")


if __name__ == "__main__":
    main()
