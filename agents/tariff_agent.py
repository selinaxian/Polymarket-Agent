"""
Tariff/Trade Policy Analysis & Decision Agent
关税/贸易政策分析与决策 Agent

Two LLM chains / 两条 LLM 链:
  1. Analysis chain: extracts policy signals, affected countries/sectors, landing probability
     分析链：提取政策信号、受影响国家/行业、落地概率
  2. Decision chain: compares AI estimate vs market odds, outputs BUY_YES/BUY_NO/NO_TRADE
     决策链：对比 AI 估计 vs 市场赔率，输出交易建议

Both chains use Gemini with structured JSON output via JsonOutputParser.
两条链都使用 Gemini，通过 JsonOutputParser 输出结构化 JSON。
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import statistics
import config


# ============================================================
# Analysis Chain / 分析链
# Input: news text (plain text, may include RAG context)
# Output: structured JSON with policy_signal, signal_strength,
#         affected_countries, affected_sectors, tariff_actions, etc.
# 输入：新闻文本（纯文本，可能包含 RAG 上下文）
# 输出：结构化 JSON，含政策信号、强度、受影响国家/行业、关税动作等
# ============================================================

def create_tariff_analysis_chain():
    """Create the tariff analysis LLM chain / 创建关税分析 LLM 链"""
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.TEMPERATURE,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位资深的国际贸易政策分析师，专注于美国关税政策和贸易战分析。

你会收到关于关税/贸易政策的新闻和声明，你需要提取政策信号并评估影响。

请严格按照以下 JSON 格式输出，不要包含任何其他文字：

{{
    "policy_signal": "escalation / de-escalation / negotiation / threat / implementation / neutral",
    "signal_strength": 一个从-10到10的整数，-10表示极度贸易自由化，10表示极度贸易保护/加税,
    "affected_countries": [
        {{
            "country": "国家名称",
            "impact": "positive / negative / neutral",
            "details": "具体影响说明"
        }}
    ],
    "affected_sectors": [
        {{
            "sector": "行业名称",
            "impact": "positive / negative / neutral",
            "details": "具体影响说明"
        }}
    ],
    "tariff_actions": [
        {{
            "action": "具体关税动作（如：对中国加征25%关税）",
            "status": "announced / implemented / proposed / threatened / revoked",
            "probability": 落地概率（0-100）
        }}
    ],
    "key_signals": [
        "关键信号1",
        "关键信号2"
    ],
    "market_implications": {{
        "short_term": "短期市场影响分析",
        "medium_term": "中期影响分析",
        "polymarket_relevance": "与 Polymarket 预测市场的相关性分析"
    }},
    "summary": "一句话总结当前关税/贸易政策态势"
}}"""),
        ("human", "请分析以下关税/贸易政策相关的新闻和声明：\n\n{news_text}")
    ])

    return prompt | llm | JsonOutputParser()


def analyze_tariff_news(news_items: list[dict]) -> dict:
    """
    Analyze a batch of tariff/trade news (standalone, without RAG).
    分析一批关税/贸易新闻（独立调用，不含 RAG）。
    For RAG-enhanced analysis, use AnalysisAgent in workflow.py instead.
    如需 RAG 增强分析，请使用 workflow.py 中的 AnalysisAgent。

    Args:
        news_items: list of news dicts with title, date, source, summary

    Returns:
        dict: structured analysis result
    """
    # Format news into plain text / 格式化新闻为纯文本
    lines = []
    for i, item in enumerate(news_items, 1):
        lines.append(f"[{i}] [{item.get('date', '')}] {item['title']}")
        lines.append(f"    来源: {item.get('source', 'Unknown')}")
        if item.get("summary"):
            lines.append(f"    摘要: {item['summary'][:200]}")
        if item.get("category"):
            lines.append(f"    类别: {item['category']}")
        lines.append("")

    news_text = "\n".join(lines)

    chain = create_tariff_analysis_chain()
    return config.invoke_with_retry(chain, {"news_text": news_text[:8000]})


def analyze_tariff_document(text: str, title: str = "") -> dict:
    """
    Analyze a single tariff/trade policy document (executive order, statement, etc.)
    分析单篇关税/贸易政策文件（行政命令、声明等）

    Args:
        text: document full text / 文件全文
        title: document title / 文件标题

    Returns:
        dict: structured analysis result
    """
    doc_text = f"标题: {title}\n\n{text}" if title else text

    chain = create_tariff_analysis_chain()
    return config.invoke_with_retry(chain, {"news_text": doc_text[:8000]})


# ============================================================
# AnalysisAgent Tools (ReAct) / 分析 Agent 工具定义
# These are real LangChain @tool functions that the ReAct agent
# can autonomously decide when and how to call.
# 这些是真正的 LangChain @tool，ReAct agent 自主决定何时调用。
# ============================================================

from langchain_core.tools import tool


@tool
def search_news(query: str) -> str:
    """Search Google News RSS for tariff/trade related articles.
    Use this to find news relevant to a specific contract or topic.
    You can call this multiple times with different queries.

    Args:
        query: English search keywords, e.g. "US Pakistan trade deal" or "Trump EU auto tariff"

    Returns:
        Formatted news articles with date, title, source, and summary.
    """
    from tools.news_scraper import get_trade_news_rss

    results = get_trade_news_rss(query=query, limit=10)
    if not results:
        return f"No news found for query: {query}"

    lines = []
    for i, n in enumerate(results, 1):
        lines.append(f"[{i}] [{n.get('date', '')}] {n['title']} ({n.get('source', '')})")
        if n.get("summary"):
            lines.append(f"    {n['summary'][:200]}")
    return "\n".join(lines)


@tool
def search_historical_events(query: str) -> str:
    """Search the historical tariff events database (2018-2026) for similar past events.
    Returns the most similar historical tariff events with S&P500 reactions and outcomes.
    Use this to find historical analogs for the current situation.

    Args:
        query: Description of the current situation, e.g. "US threatens tariffs on EU automobiles"

    Returns:
        Similar historical events with dates, S&P500 reactions, and whether tariffs actually landed.
    """
    from tools.tariff_history import find_similar_events

    events = find_similar_events(query, n_results=3)
    if not events:
        return "No similar historical events found."

    lines = []
    for evt in events:
        landed = "LANDED" if evt["tariff_landed"] else "DID NOT LAND"
        lines.append(
            f"- [{evt['date']}] {evt['description'][:100]}\n"
            f"  Type: {evt['type']} | S&P500: {evt['sp500_pct']:+.1f}% | Tariff: {landed}\n"
            f"  Outcome: {evt['outcome'][:150]}"
        )
    return "\n".join(lines)


@tool
def search_recent_analyses(query: str) -> str:
    """Search past analysis records for similar recent news and the policy signals assigned at that time.
    Use this to check what signals were detected in recent similar situations.

    Args:
        query: Headlines or topic to search for, e.g. "China tariff escalation"

    Returns:
        Recent similar news with their previously assigned policy signals.
    """
    from tools.rag_store import find_similar_news

    results = find_similar_news(query, n_results=3)
    if not results:
        return "No similar recent analyses found."

    lines = []
    for sn in results:
        signal = sn["metadata"].get("policy_signal", "?")
        strength = sn["metadata"].get("signal_strength", "?")
        lines.append(f"- {sn['document'][:150]} (signal: {signal}, strength: {strength})")
    return "\n".join(lines)


# Collect tools for the ReAct agent / 收集工具列表
ANALYSIS_TOOLS = [search_news, search_historical_events, search_recent_analyses]


def create_analysis_react_agent():
    """
    Create a ReAct agent for AnalysisAgent that can autonomously search
    for news and historical context before producing its analysis.
    创建 ReAct 分析 agent，能自主搜索新闻和历史参考。

    Returns:
        A compiled LangGraph ReAct agent
    """
    from langgraph.prebuilt import create_react_agent

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.TEMPERATURE,
    )

    system_prompt = """你是一位资深的国际贸易政策分析师，专注于美国关税政策和贸易战分析。

你的任务是分析一个 Polymarket 预测市场合约，评估关税/贸易政策的走向。

## 工作流程

1. **搜索相关新闻**：使用 search_news 工具搜索与合约直接相关的新闻。
   - 先搜合约核心话题（如合约问 US-Pakistan trade deal，就搜 "US Pakistan trade deal"）
   - 如果结果不够或角度单一，换一个关键词再搜（如 "Pakistan trade negotiations 2026"）
   - 也搜一下更宏观的背景（如 "Trump trade policy"），但合约专属新闻优先
   - 你可以搜索 2-4 次，直到你觉得信息足够做出判断

2. **搜索历史类比**：使用 search_historical_events 工具查找相似的历史关税事件。
   - 用当前局势的简短描述作为查询词

3. **搜索近期分析记录**（可选）：使用 search_recent_analyses 工具查看最近类似新闻的分析结果。

4. **综合分析**：基于搜集到的所有信息，输出你的分析结论。

## 最终输出格式

在你完成所有搜索后，请输出如下 JSON（用 ```json 包裹）：

```json
{
    "policy_signal": "escalation / de-escalation / negotiation / threat / implementation / neutral",
    "signal_strength": -10到10的整数,
    "affected_countries": [{"country": "国家", "impact": "positive/negative/neutral", "details": "说明"}],
    "affected_sectors": [{"sector": "行业", "impact": "positive/negative/neutral", "details": "说明"}],
    "tariff_actions": [{"action": "具体动作", "status": "announced/implemented/proposed/threatened/revoked", "probability": 0到100}],
    "key_signals": ["信号1", "信号2"],
    "market_implications": {"short_term": "短期", "medium_term": "中期", "polymarket_relevance": "与合约的相关性"},
    "summary": "一句话总结",
    "search_queries_used": ["你实际使用的搜索词1", "搜索词2"]
}
```

重要：你必须先使用工具搜索信息，不要凭空分析。合约专属新闻比通用新闻更重要。"""

    return create_react_agent(llm, ANALYSIS_TOOLS, prompt=system_prompt)


def create_backtest_react_agent(before_date: str):
    """
    Create a ReAct agent for backtesting that can only see news before a given date.
    创建回测用 ReAct agent，search_news 自动加日期限制，防止 look-ahead bias。

    Args:
        before_date: cutoff date (YYYY-MM-DD), news after this date will not be returned

    Returns:
        A compiled LangGraph ReAct agent with date-limited tools
    """
    from langgraph.prebuilt import create_react_agent
    from backtest.run_tariff_backtest import fetch_historical_news

    @tool
    def search_news_historical(query: str) -> str:
        """Search Google News for tariff/trade articles before the analysis date.
        This tool automatically filters out future news to prevent look-ahead bias.
        You can call this multiple times with different queries.

        Args:
            query: English search keywords, e.g. "US Pakistan trade deal"

        Returns:
            News articles published before the analysis date.
        """
        results = fetch_historical_news(query, before_date=before_date, lookback_days=14, limit=10)
        if not results:
            return f"No news found for '{query}' before {before_date}"

        lines = [f"(News before {before_date} only)"]
        for i, n in enumerate(results, 1):
            lines.append(f"[{i}] [{n.get('date', '')}] {n['title']} ({n.get('source', '')})")
            if n.get("summary"):
                lines.append(f"    {n['summary'][:200]}")
        return "\n".join(lines)

    backtest_tools = [search_news_historical, search_historical_events, search_recent_analyses]

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.TEMPERATURE,
    )

    system_prompt = f"""你是一位资深的国际贸易政策分析师，专注于美国关税政策和贸易战分析。

你的任务是分析一个 Polymarket 预测市场合约，评估关税/贸易政策的走向。

**重要：你现在处于回测模式，分析日期是 {before_date}。你只能看到此日期之前的信息。**

## 工作流程

1. **搜索相关新闻**：使用 search_news_historical 工具搜索与合约相关的新闻（自动过滤为 {before_date} 之前的新闻）。
   - 先搜合约核心话题
   - 换关键词再搜 1-2 次
   - 你可以搜索 2-3 次

2. **搜索历史类比**：使用 search_historical_events 工具查找相似的历史关税事件。

3. **综合分析**：基于搜集到的所有信息，输出分析结论。

## 最终输出格式

```json
{{{{
    "policy_signal": "escalation / de-escalation / negotiation / threat / implementation / neutral",
    "signal_strength": -10到10的整数,
    "affected_countries": [{{{{"country": "国家", "impact": "positive/negative/neutral", "details": "说明"}}}}],
    "affected_sectors": [{{{{"sector": "行业", "impact": "positive/negative/neutral", "details": "说明"}}}}],
    "tariff_actions": [{{{{"action": "具体动作", "status": "announced/implemented/proposed/threatened/revoked", "probability": 0到100}}}}],
    "key_signals": ["信号1", "信号2"],
    "market_implications": {{{{"short_term": "短期", "medium_term": "中期", "polymarket_relevance": "与合约的相关性"}}}},
    "summary": "一句话总结",
    "search_queries_used": ["搜索词1", "搜索词2"]
}}}}
```

重要：你只能基于 {before_date} 之前的信息进行分析，不能使用未来数据。"""

    return create_react_agent(llm, backtest_tools, prompt=system_prompt)


# ============================================================
# Per-News Sentiment Scoring / 单条新闻情绪打分
# LLM scores each news item individually from -1 (bearish/dovish)
# to +1 (hawkish/escalation), then aggregate into a composite indicator.
# LLM 对每条新闻单独打情绪分（-1 到 +1），然后聚合为综合情绪指标。
# ============================================================

def create_sentiment_scoring_chain():
    """Create LLM chain for scoring individual news sentiment / 创建单条新闻情绪打分链"""
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.TEMPERATURE,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位国际贸易政策情绪分析专家。你需要对单条关税/贸易新闻打一个情绪分数。

评分规则：
- -1.0: 极度鸽派/缓和信号（如：全面取消关税、达成重大贸易协议）
- -0.5: 温和缓和（如：暂停加税、重启谈判）
- 0.0: 中性/信息性（如：例行声明、无明确方向）
- +0.5: 温和鹰派（如：威胁加税、调查启动）
- +1.0: 极度鹰派/升级信号（如：大幅加征关税、贸易战升级）

请严格按照以下 JSON 格式输出，不要包含任何其他文字：

{{
    "sentiment_score": 情绪分数（-1.0到+1.0，精确到0.1）,
    "reasoning": "一句话解释评分理由"
}}"""),
        ("human", "请对以下新闻打情绪分数：\n\n标题: {title}\n来源: {source}\n日期: {date}\n摘要: {summary}")
    ])

    return prompt | llm | JsonOutputParser()


def score_news_sentiment(news_items: list[dict]) -> list[dict]:
    """
    Score each news item individually with sentiment from -1 to +1.
    对每条新闻单独打情绪分（-1 到 +1）。

    Args:
        news_items: list of news dicts with title, date, source, summary

    Returns:
        list of dicts with original news fields + sentiment_score, reasoning
    """
    chain = create_sentiment_scoring_chain()
    scored = []

    for item in news_items:
        try:
            result = config.invoke_with_retry(chain, {
                "title": item.get("title", ""),
                "source": item.get("source", "Unknown"),
                "date": item.get("date", ""),
                "summary": item.get("summary", "")[:300],
            })
            score = max(-1.0, min(1.0, float(result.get("sentiment_score", 0))))
            scored.append({
                **item,
                "sentiment_score": round(score, 2),
                "sentiment_reasoning": result.get("reasoning", ""),
            })
        except Exception as e:
            # On failure, assign neutral score / 失败时给中性分
            scored.append({
                **item,
                "sentiment_score": 0.0,
                "sentiment_reasoning": f"评分失败: {str(e)[:50]}",
            })

    return scored


def aggregate_sentiment(scored_news: list[dict]) -> dict:
    """
    Aggregate per-news sentiment scores into a composite indicator.
    聚合所有新闻的情绪分数，得到综合情绪指标。

    Returns:
        dict with:
          - composite_sentiment: weighted average (-1 to +1)
          - news_count: number of scored items
          - sentiment_std: standard deviation
          - hawkish_count: number of positive scores
          - dovish_count: number of negative scores
          - neutral_count: number of zero scores
          - scored_news: list of scored items (for display)
    """
    scores = [item["sentiment_score"] for item in scored_news]

    if not scores:
        return {
            "composite_sentiment": 0.0,
            "news_count": 0,
            "sentiment_std": 0.0,
            "hawkish_count": 0,
            "dovish_count": 0,
            "neutral_count": 0,
            "scored_news": [],
        }

    # Weighted average: more recent news (earlier in list) gets higher weight
    # 加权平均：列表前面的（更新的）新闻权重更高
    weights = [1.0 / (1 + 0.1 * i) for i in range(len(scores))]
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    weight_total = sum(weights)
    composite = round(weighted_sum / weight_total, 3)

    return {
        "composite_sentiment": composite,
        "news_count": len(scores),
        "sentiment_std": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
        "hawkish_count": sum(1 for s in scores if s > 0.1),
        "dovish_count": sum(1 for s in scores if s < -0.1),
        "neutral_count": sum(1 for s in scores if -0.1 <= s <= 0.1),
        "scored_news": scored_news,
    }


# ============================================================
# Decision Chain / 决策链
# Input: tariff analysis result + Polymarket contract info
# Output: BUY_YES / BUY_NO / NO_TRADE with edge, confidence, reasoning
# 输入：关税分析结果 + Polymarket 合约信息
# 输出：BUY_YES / BUY_NO / NO_TRADE + edge、置信度、推理
# ============================================================

def create_tariff_decision_chain():
    """Create the tariff trading decision LLM chain / 创建关税交易决策 LLM 链"""
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.TEMPERATURE,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位量化交易策略分析师，专注于政策事件驱动的预测市场交易。

你会收到两部分信息：
1. AI 分析师对关税/贸易政策的分析结果
2. Polymarket 上相关合约的当前市场数据

你的任务是判断市场定价是否合理，并给出交易建议。

请严格按照以下 JSON 格式输出，不要包含任何其他文字：

{{
    "contract_question": "合约问题",
    "market_yes_price": 当前 Yes 价格,
    "our_estimated_probability": 我们估计的概率（0-100）,
    "recommendation": "BUY_YES / BUY_NO / NO_TRADE",
    "edge": 估计的概率优势（百分点）,
    "confidence": 置信度（0-100）,
    "reasoning": "详细推理过程",
    "risk_factors": [
        "风险因素1",
        "风险因素2"
    ]
}}"""),
        ("human", """
## 关税/贸易政策分析结果
{tariff_analysis}

## Polymarket 合约信息
合约问题：{contract_question}
当前 Yes 价格：{yes_price}
当前 No 价格：{no_price}
交易量：{volume}

请给出你的分析和交易建议。
""")
    ])

    return prompt | llm | JsonOutputParser()


def make_tariff_decision(tariff_analysis: dict, contract_info: dict) -> dict:
    """
    Generate trade recommendation by comparing analysis vs market odds.
    结合关税分析结果和市场数据，输出交易建议。

    Args:
        tariff_analysis: analysis result from analyze_tariff_news()
                        来自 analyze_tariff_news() 的分析结果
        contract_info: Polymarket contract with question, yes_price, no_price, volume
                      Polymarket 合约信息

    Returns:
        dict: recommendation (BUY_YES/BUY_NO/NO_TRADE), edge, confidence, reasoning
    """
    chain = create_tariff_decision_chain()
    return config.invoke_with_retry(chain, {
        "tariff_analysis": json.dumps(tariff_analysis, ensure_ascii=False, indent=2),
        "contract_question": contract_info["question"],
        "yes_price": contract_info["yes_price"],
        "no_price": contract_info["no_price"],
        "volume": contract_info.get("volume", "N/A"),
    })


# ============================================================
# Test / 测试
# ============================================================

if __name__ == "__main__":
    from tools.news_scraper import get_all_tariff_news

    print("=== 抓取关税新闻 ===\n")
    news = get_all_tariff_news(limit_per_source=5)
    print(f"共 {len(news)} 条新闻\n")

    for n in news[:5]:
        print(f"  [{n['date']}] {n['title']}")
        print(f"    {n['source']}")
    print()

    print("=== AI 分析关税态势 ===\n")
    analysis = analyze_tariff_news(news[:10])
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
