"""
LangGraph Multi-Agent Workflow for Tariff/Trade Policy Trading
LangGraph 多 Agent 协作工作流（关税/贸易政策交易）

Pipeline / 流程:
  DataAgent → AnalysisAgent → DecisionAgent → RiskAgent → END
  (数据采集)    (RAG增强分析)    (交易决策)     (风控审查)

Each agent reads from and writes to a shared AgentState (TypedDict).
每个 Agent 通过共享的 AgentState 读写数据。

The workflow uses conditional edges: if DataAgent encounters an error
(e.g. no news or no contracts found), the pipeline short-circuits to END.
工作流使用条件边：DataAgent 出错时直接跳到 END。
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import json
import config
from tools.polymarket_api import find_tariff_contracts
from agents.tariff_agent import (
    create_tariff_analysis_chain,
    create_tariff_decision_chain,
    create_analysis_react_agent,
    score_news_sentiment,
    aggregate_sentiment,
)


# ============================================================
# Shared State / 共享状态定义
# All agents read from and write to this TypedDict.
# 所有 Agent 通过此 TypedDict 共享数据。
# ============================================================

class AgentState(TypedDict):
    # DataAgent outputs / DataAgent 输出
    news: list[dict]             # Tariff/trade news / 关税贸易新闻
    contracts: list[dict]        # Polymarket contracts / 合约列表
    target_contract: dict        # Selected contract (highest volume) / 目标合约（最大交易量）
    # AnalysisAgent outputs / AnalysisAgent 输出
    tariff_analysis: dict        # Policy signal analysis / 政策信号分析结果
    # DecisionAgent outputs / DecisionAgent 输出
    decision: dict               # Trade recommendation / 交易建议
    # RiskAgent outputs / RiskAgent 输出
    risk_review: dict            # Risk assessment / 风险评估结果
    final_decision: dict         # Post-risk decision (may override) / 风控后最终建议
    # Pipeline state / 流程状态
    error: str                   # Error message, empty if OK / 错误信息


# ============================================================
# DataAgent: Contract Discovery / 合约发现
# Searches Polymarket for tariff/trade contracts.
# 搜索 Polymarket 关税/贸易合约。
# News scraping is handled by AnalysisAgent's tools.
# 新闻搜索由 AnalysisAgent 的 tools 自主完成。
# ============================================================

def data_agent(state: AgentState) -> dict:
    """Search Polymarket for tariff/trade contracts / 搜索关税/贸易合约"""
    print("\n[DataAgent] 开始采集数据...")

    # --- Search Polymarket for tariff-related contracts ---
    # 注意：Gamma API 忽略搜索参数，采用分页拉取 + 客户端关键词过滤
    print("  [DataAgent] 搜索 Polymarket 关税/贸易合约...")
    raw = find_tariff_contracts()

    seen = set()
    contracts = [c for c in raw if not (c["market_id"] in seen or seen.add(c["market_id"]))]

    if not contracts:
        return {"error": "没有找到活跃的关税/贸易相关合约"}

    print(f"  [DataAgent] 找到 {len(contracts)} 个合约")

    # Select target: highest volume contract / 选择交易量最大的合约
    target = max(contracts, key=lambda c: float(c.get("volume", 0) or 0))
    print(f"  [DataAgent] 目标合约: {target['question']}")
    print("[DataAgent] 数据采集完成\n")

    return {
        "news": [],  # News is now fetched by AnalysisAgent's tools
        "contracts": contracts,
        "target_contract": target,
    }


# ============================================================
# AnalysisAgent: RAG-Enhanced Policy Analysis / RAG 增强的政策分析
# Uses two RAG sources before calling LLM:
# 调用 LLM 前先检索两个 RAG 来源：
#   1. tariff_events (ChromaDB) — 17 historical events (2018-2026)
#      历史重大事件：含 S&P500 反应、是否落地
#   2. tariff_news (ChromaDB) — recent similar news from past runs
#      近期相似新闻：含当时的政策信号
# Both are injected into the LLM prompt as context.
# 两者拼接进 LLM prompt 作为分析上下文。
# ============================================================

def analysis_agent(state: AgentState) -> dict:
    """
    ReAct AnalysisAgent: LLM autonomously searches for news and
    historical context using tools, then produces analysis.
    ReAct 分析 Agent：LLM 自主使用工具搜索新闻和历史参考，然后输出分析。

    News is fetched entirely by the ReAct agent's tools — DataAgent
    no longer provides news. The tool-searched news is extracted from
    ReAct messages and passed to RiskAgent via state["news"].
    新闻完全由 ReAct agent 的工具搜索 — DataAgent 不再提供新闻。
    工具搜索到的新闻从 ReAct 消息中提取，通过 state["news"] 传给 RiskAgent。
    """
    print("[AnalysisAgent] 开始分析（ReAct mode）...")

    if state.get("error"):
        return {}

    from tools.rag_store import store_news
    from tools.news_tracker import record_sentiment

    target = state.get("target_contract", {})
    target_q = target.get("question", "unknown contract")

    # ── Step 1: ReAct agent autonomously searches for information ──
    print(f"  [AnalysisAgent] 目标合约: {target_q[:55]}")
    print(f"  [AnalysisAgent] 启动 ReAct agent，LLM 将自主搜索新闻和历史参考...")

    react_agent = create_analysis_react_agent()
    react_input = {
        "messages": [{
            "role": "user",
            "content": (
                f"请分析以下 Polymarket 合约:\n\n"
                f"合约问题: {target_q}\n\n"
                f"请按照工作流程操作：\n"
                f"1. 先用 search_news 搜索与此合约直接相关的新闻（至少搜 2 次，用不同关键词）\n"
                f"2. 用 search_historical_events 搜索历史类比\n"
                f"3. 可选用 search_recent_analyses 查看近期类似分析\n"
                f"4. 最后输出你的 JSON 分析结论"
            ),
        }],
    }

    try:
        react_result = react_agent.invoke(react_input)
        raw_content = react_result["messages"][-1].content
        if isinstance(raw_content, list):
            final_msg = "\n".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        else:
            final_msg = str(raw_content)
        print(f"  [AnalysisAgent] ReAct agent 完成，输出 {len(final_msg)} 字符")

        # Count tool calls
        tool_msgs = [m for m in react_result["messages"] if hasattr(m, "type") and m.type == "tool"]
        print(f"  [AnalysisAgent] 工具调用次数: {len(tool_msgs)}")

        # Extract news from tool results for downstream agents (RiskAgent)
        # 从工具返回的消息中提取新闻，传给 RiskAgent
        searched_news = _extract_news_from_tool_messages(react_result["messages"])
        print(f"  [AnalysisAgent] 从工具结果中提取 {len(searched_news)} 条新闻")

    except Exception as e:
        print(f"  [AnalysisAgent] ReAct agent 失败: {e}")
        print(f"  [AnalysisAgent] 降级：用通用搜索获取新闻...")
        from tools.news_scraper import get_trade_news_rss
        fallback_news = get_trade_news_rss("tariff trade policy", limit=15)
        searched_news = fallback_news
        news_text = "\n".join(
            f"[{n.get('date', '')}] {n['title']} ({n.get('source', '')})"
            for n in fallback_news[:12]
        )
        chain = create_tariff_analysis_chain()
        result = config.invoke_with_retry(chain, {"news_text": news_text[:8000]})
        return {"tariff_analysis": result, "news": fallback_news}

    # ── Step 2: Parse JSON from ReAct agent's final output ──
    result = _extract_json_from_text(final_msg)
    if not result:
        print("  [AnalysisAgent] 无法解析 JSON，降级为传统分析链...")
        news_text = "\n".join(
            f"[{n.get('date', '')}] {n['title']} ({n.get('source', '')})"
            for n in searched_news[:12]
        )
        chain = create_tariff_analysis_chain()
        result = config.invoke_with_retry(chain, {"news_text": news_text[:8000]})
        return {"tariff_analysis": result, "news": searched_news}

    # ── Step 3: Per-news sentiment scoring on searched news ──
    print("  [AnalysisAgent] 逐条新闻情绪打分...")
    scored_news = score_news_sentiment(searched_news[:12])
    sentiment_agg = aggregate_sentiment(scored_news)
    result["sentiment"] = sentiment_agg
    print(f"  [AnalysisAgent] 综合情绪: {sentiment_agg['composite_sentiment']:+.3f} "
          f"(鹰派{sentiment_agg['hawkish_count']} / 鸽派{sentiment_agg['dovish_count']})")

    # ── Step 4: Store to RAG + sentiment history ──
    try:
        store_news(searched_news[:10], result)
    except Exception:
        pass
    try:
        record_sentiment(sentiment_agg)
    except Exception:
        pass

    signal = result.get("policy_signal", "unknown")
    strength = result.get("signal_strength", 0)
    queries_used = result.get("search_queries_used", [])
    print(f"  [AnalysisAgent] 政策信号: {signal}")
    print(f"  [AnalysisAgent] 保护主义强度: {strength:+d}/10" if isinstance(strength, int) else f"  [AnalysisAgent] 保护主义强度: {strength}")
    print(f"  [AnalysisAgent] 综合情绪: {sentiment_agg['composite_sentiment']:+.3f}")
    if queries_used:
        print(f"  [AnalysisAgent] 搜索词: {queries_used}")
    print("[AnalysisAgent] 分析完成\n")

    # Pass searched news to downstream agents via state["news"]
    return {"tariff_analysis": result, "news": searched_news}


def _extract_news_from_tool_messages(messages: list) -> list[dict]:
    """
    Extract structured news items from ReAct agent's tool call results.
    从 ReAct agent 的工具调用结果中提取新闻条目。

    Parses the text output of search_news tool calls back into dicts.
    """
    import re as _re

    news_items = []
    seen_titles = set()

    for msg in messages:
        # Only look at tool response messages from search_news
        if not (hasattr(msg, "type") and msg.type == "tool"):
            continue
        if not hasattr(msg, "name") or msg.name != "search_news":
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Parse lines like: [1] [2026-03-31] Title here (Source)
        for match in _re.finditer(
            r'\[\d+\]\s+\[([^\]]*)\]\s+(.+?)\s+\(([^)]+)\)\s*$',
            content, _re.MULTILINE,
        ):
            date, title, source = match.group(1), match.group(2), match.group(3)
            key = title[:60].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                news_items.append({
                    "date": date,
                    "title": title.strip(),
                    "source": source.strip(),
                    "summary": "",
                })

    return news_items


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON from LLM output that may contain markdown code blocks."""
    import re as _re

    # Try to find ```json ... ``` block
    match = _re.search(r'```json\s*\n?(.*?)\n?\s*```', text, _re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    match = _re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, _re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# DecisionAgent: Trade Decision / 交易决策
# Compares AI estimated probability vs market odds.
# 对比 AI 估计概率与市场赔率，输出交易建议。
# ============================================================

def _parse_prices(contract: dict) -> tuple[float, float]:
    """Parse Yes/No prices from Polymarket format / 解析 Polymarket 价格格式"""
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


def decision_agent(state: AgentState) -> dict:
    """Generate trade recommendation / 综合分析结果和市场数据输出交易建议"""
    print("[DecisionAgent] 开始生成交易建议...")

    if state.get("error") or not state.get("tariff_analysis"):
        return {}

    chain = create_tariff_decision_chain()
    target = state["target_contract"]
    yes_price, no_price = _parse_prices(target)

    result = config.invoke_with_retry(chain, {
        "tariff_analysis": json.dumps(state["tariff_analysis"], ensure_ascii=False, indent=2),
        "contract_question": target["question"],
        "yes_price": yes_price,
        "no_price": no_price,
        "volume": target.get("volume", "N/A"),
    })

    rec = result.get("recommendation", "N/A")
    edge = result.get("edge", "N/A")
    print(f"  [DecisionAgent] 建议: {rec}")
    print(f"  [DecisionAgent] 优势: {edge} 个百分点")
    print("[DecisionAgent] 决策完成\n")

    return {"decision": result}


# ============================================================
# RiskAgent: Risk Review / 风险评估
# Checks for contradictory signals, evaluates source reliability,
# and can override BUY recommendations to NO_TRADE if risk is too high.
# 检查矛盾信号、评估来源可靠性，风险过高时将买入建议降级为 NO_TRADE。
# ============================================================

def risk_agent(state: AgentState) -> dict:
    """Review risk and optionally downgrade recommendation / 风险审查，必要时降级建议"""
    print("[RiskAgent] 开始风险评估...")

    if state.get("error") or not state.get("decision"):
        return {}

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.TEMPERATURE,
    )

    # Risk review prompt — checks 4 dimensions:
    # 风控审查 prompt — 检查 4 个维度：
    #   1. Contradictory signals between news / 新闻间矛盾信号
    #   2. Source reliability (official vs tabloid) / 来源可靠性
    #   3. AI overconfidence / AI 是否过度自信
    #   4. Edge too small to trade / Edge 是否过小
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位风控审查员，负责审查 AI 交易建议的可靠性。

你会收到：
1. 原始新闻列表（含日期和来源）
2. AI 的关税政策分析
3. AI 的交易建议

你需要检查以下风险：

### 矛盾信号判定规则（重要）
新闻带有日期，你必须区分「政策演变」和「真正矛盾」：
- **不算矛盾：** 不同时间段的新闻观点不同。例如 3 月说"加税"、4 月说"谈判"，这是政策立场随时间演变，属于正常的政策转向，不应视为矛盾。
- **算矛盾：** 同一周内（7 天内），来自不同来源的新闻传递相互对立的信号。例如同一周内白宫说"即将加税"而国务院说"正在谈判减税"，才是真正的矛盾信号，说明政策方向不确定。
- 判定时请注明矛盾新闻的日期和来源，以便追溯。

### 其他风险维度
- 消息来源的可靠性（官方来源 vs 小众媒体 vs 社交媒体）
- AI 分析是否过度自信或忽略了关键风险
- 建议的 edge 是否过小不值得交易

请严格按照以下 JSON 格式输出，不要包含任何其他文字：

{{
    "contradictory_signals": [
        "矛盾信号描述（注明日期和来源，例：[2025-03-15 Reuters] vs [2025-03-17 White House] ...）"
    ],
    "policy_evolution": "政策演变总结（如果不同时间段的新闻方向不同，在此描述政策如何随时间变化，这不算矛盾）",
    "source_reliability": {{
        "high_reliability_count": 来自官方/权威来源的新闻数量,
        "low_reliability_count": 来自不可靠来源的新闻数量,
        "assessment": "对来源整体可靠性的评价"
    }},
    "risk_level": "low / medium / high / critical",
    "risk_score": 1到10的整数，10表示风险最高,
    "override_to_no_trade": true或false（是否应该把建议降级为NO_TRADE）,
    "override_reason": "降级原因（如果override_to_no_trade为false则填空字符串）",
    "warnings": [
        "需要注意的风险点"
    ],
    "assessment": "对交易建议整体的风险评估总结"
}}"""),
        ("human", """
## 新闻来源（按日期排列）
{news_summary}

## AI 关税政策分析
政策信号: {policy_signal}
保护主义强度: {signal_strength}
态势总结: {analysis_summary}

## AI 交易建议
合约: {contract_question}
建议: {recommendation}
Edge: {edge} 百分点
置信度: {confidence}%
推理: {reasoning}

请进行风险评估。注意区分「政策演变」和「真正矛盾」。
""")
    ])

    chain = prompt | llm | JsonOutputParser()

    # Format news with dates for risk review / 格式化新闻（含日期）供风控审查
    news_lines = []
    for n in state["news"][:12]:
        date = n.get("date", "日期不详")
        source = n.get("source", "Unknown")
        news_lines.append(f"[{date}] [{source}] {n['title'][:80]}")

    ta = state["tariff_analysis"]
    dec = state["decision"]

    risk_review = config.invoke_with_retry(chain, {
        "news_summary": "\n".join(news_lines),
        "policy_signal": ta.get("policy_signal", ""),
        "signal_strength": ta.get("signal_strength", 0),
        "analysis_summary": ta.get("summary", ""),
        "contract_question": dec.get("contract_question", ""),
        "recommendation": dec.get("recommendation", ""),
        "edge": dec.get("edge", 0),
        "confidence": dec.get("confidence", 0),
        "reasoning": dec.get("reasoning", "")[:500],
    })

    risk_level = risk_review.get("risk_level", "unknown")
    risk_score = risk_review.get("risk_score", 0)
    override = risk_review.get("override_to_no_trade", False)

    print(f"  [RiskAgent] 风险等级: {risk_level} ({risk_score}/10)")

    # Build final decision: copy original, override if needed
    # 构建最终决策：复制原始建议，必要时降级
    final = dict(dec)
    if override and final.get("recommendation") in ("BUY_YES", "BUY_NO"):
        original_rec = final["recommendation"]
        final["recommendation"] = "NO_TRADE"
        final["override_reason"] = risk_review.get("override_reason", "")
        final["original_recommendation"] = original_rec
        print(f"  [RiskAgent] 降级: {original_rec} -> NO_TRADE")
        print(f"  [RiskAgent] 原因: {risk_review.get('override_reason', '')[:80]}")
    else:
        print(f"  [RiskAgent] 维持建议: {final.get('recommendation', 'N/A')}")

    print("[RiskAgent] 风险评估完成\n")

    return {
        "risk_review": risk_review,
        "final_decision": final,
    }


# ============================================================
# Routing / 路由
# If DataAgent sets error, skip all subsequent agents.
# DataAgent 出错时跳过后续所有 Agent。
# ============================================================

def should_continue(state: AgentState) -> str:
    if state.get("error"):
        return END
    return "continue"


# ============================================================
# Build Workflow / 构建工作流
# ============================================================

def build_workflow():
    """
    Build and compile the LangGraph workflow.
    构建并编译 LangGraph 工作流。

    Graph structure / 图结构:
        START → DataAgent →(error?)→ END
                    ↓ (ok)
               AnalysisAgent
                    ↓
               DecisionAgent
                    ↓
                RiskAgent → END
    """
    graph = StateGraph(AgentState)

    # Register nodes / 注册节点
    graph.add_node("data_agent", data_agent)
    graph.add_node("analysis_agent", analysis_agent)
    graph.add_node("decision_agent", decision_agent)
    graph.add_node("risk_agent", risk_agent)

    # Wire edges / 连接边
    graph.add_edge(START, "data_agent")
    graph.add_conditional_edges(
        "data_agent", should_continue,
        {"continue": "analysis_agent", END: END},
    )
    graph.add_edge("analysis_agent", "decision_agent")
    graph.add_edge("decision_agent", "risk_agent")
    graph.add_edge("risk_agent", END)

    return graph.compile()


# Pre-compiled workflow instance, import as: from agents.workflow import workflow
# 预编译的工作流实例，使用方式：from agents.workflow import workflow
workflow = build_workflow()
