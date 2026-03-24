"""
Polymarket 关税/贸易政策预测 AI Agent — Demo
=============================================
使用 LangGraph 多 Agent 工作流：
  DataAgent → AnalysisAgent → DecisionAgent
全部使用真实数据（白宫/USTR/Google News + Polymarket API）
"""

import json
from agents.workflow import workflow, _parse_prices


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    print_section("LangGraph 关税/贸易政策分析工作流")
    print()

    result = workflow.invoke({
        "news": [],
        "contracts": [],
        "target_contract": {},
        "tariff_analysis": {},
        "decision": {},
        "risk_review": {},
        "final_decision": {},
        "error": "",
    })

    if result.get("error"):
        print(f"\n流程中断: {result['error']}")
        return

    # ── 数据采集结果 ──
    print_section("数据采集结果")
    print(f"\n  新闻数量: {len(result['news'])} 条")
    print(f"  合约数量: {len(result['contracts'])} 个")

    target = result["target_contract"]
    yp, np_ = _parse_prices(target)
    print(f"\n  目标合约: {target['question']}")
    print(f"  Yes: {yp:.1%}  |  No: {np_:.1%}  |  交易量: ${float(target.get('volume', 0)):,.0f}")

    print(f"\n  新闻摘要:")
    for n in result["news"][:5]:
        print(f"    [{n.get('date','')}] {n['title'][:55]}  ({n.get('source','')})")

    print(f"\n  全部合约:")
    for i, c in enumerate(result["contracts"][:8], 1):
        cp, _ = _parse_prices(c)
        print(f"    {i}. {c['question'][:55]}  Yes: {cp:.1%}")

    # ── 关税分析结果 ──
    print_section("关税政策分析")
    ta = result["tariff_analysis"]
    print(f"\n  政策信号: {ta.get('policy_signal', 'N/A')}")
    print(f"  保护主义强度: {ta.get('signal_strength', 0):+d}/10")
    print(f"  态势: {ta.get('summary', 'N/A')}")

    if ta.get("affected_countries"):
        print(f"\n  受影响国家:")
        for ac in ta["affected_countries"][:5]:
            print(f"    [{ac.get('impact','?'):>8s}] {ac['country']}: {ac.get('details','')[:80]}")

    if ta.get("tariff_actions"):
        print(f"\n  关税动作:")
        for action in ta["tariff_actions"]:
            print(f"    - {action.get('action','')}  ({action.get('status','')}, 概率: {action.get('probability','?')}%)")

    if ta.get("key_signals"):
        print(f"\n  关键信号:")
        for sig in ta["key_signals"]:
            print(f"    - {sig}")

    # ── 交易决策 ──
    print_section("交易决策")
    dec = result["decision"]
    rec = dec.get("recommendation", "N/A")
    if rec == "BUY_YES":
        action = "买入 YES"
    elif rec == "BUY_NO":
        action = "买入 NO"
    else:
        action = "不交易"

    print(f"\n  合约: {dec.get('contract_question', 'N/A')}")
    print(f"  市场价格: {dec.get('market_yes_price', 'N/A')}")
    print(f"  AI 估计: {dec.get('our_estimated_probability', 'N/A')}%")
    print(f"  建议: {action}")
    print(f"  Edge: {dec.get('edge', 'N/A')} 百分点")
    print(f"  置信度: {dec.get('confidence', 'N/A')}%")
    print(f"\n  推理: {dec.get('reasoning', 'N/A')[:300]}")

    if dec.get("risk_factors"):
        print(f"\n  风险因素:")
        for r in dec["risk_factors"]:
            print(f"    - {r}")

    # ── 风险评估 ──
    print_section("风险评估 (RiskAgent)")
    rr = result.get("risk_review", {})
    fd = result.get("final_decision", {})

    print(f"\n  风险等级: {rr.get('risk_level', 'N/A')} ({rr.get('risk_score', '?')}/10)")

    if rr.get("contradictory_signals"):
        print(f"\n  矛盾信号:")
        for cs in rr["contradictory_signals"]:
            print(f"    - {cs}")

    src = rr.get("source_reliability", {})
    if src:
        print(f"\n  来源可靠性: 权威 {src.get('high_reliability_count', '?')} 条 / "
              f"不可靠 {src.get('low_reliability_count', '?')} 条")
        print(f"  评估: {src.get('assessment', '')[:120]}")

    if rr.get("warnings"):
        print(f"\n  风险警告:")
        for w in rr["warnings"]:
            print(f"    - {w}")

    print(f"\n  综合评估: {rr.get('assessment', 'N/A')[:200]}")

    # ── 最终决策 ──
    print_section("最终决策 (风控后)")
    final_rec = fd.get("recommendation", "N/A")
    if fd.get("original_recommendation"):
        print(f"\n  原始建议: {fd['original_recommendation']}  ->  降级为: {final_rec}")
        print(f"  降级原因: {fd.get('override_reason', '')}")
    else:
        print(f"\n  最终建议: {final_rec} (风控通过，维持原建议)")

    print(f"  Edge: {fd.get('edge', 'N/A')} 百分点  |  置信度: {fd.get('confidence', 'N/A')}%")

    print_section("流程完成")


if __name__ == "__main__":
    main()
