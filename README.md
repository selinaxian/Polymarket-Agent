# Polymarket Tariff Policy Prediction Agent

An LLM-powered multi-agent system that analyzes tariff/trade policy news to identify mispricing in [Polymarket](https://polymarket.com/) prediction market contracts. The system uses a **ReAct agent** that autonomously searches for contract-specific news via Google News RSS, retrieves historical analogs from a vector database, and generates trading recommendations — all validated through **look-ahead-bias-free backtesting**.

**Core finding:** The ReAct-based approach (where the LLM decides what to search) dramatically outperforms fixed-keyword analysis. On 5 resolved tariff contracts analyzed at T-1, the system achieves **100% accuracy, +$138 PnL, and Sharpe 1.73** — compared to -$83 PnL when using generic news. The key is letting the LLM search for news *specific to each contract* rather than feeding it the same batch of generic tariff headlines.

---

## Research Question

> Can an LLM agent that autonomously searches for contract-specific news identify mispricing in Polymarket tariff policy contracts — and does active position management (stop-loss on signal reversal) improve returns over hold-to-settlement?

---

## Key Results

### Backtest 1: Resolved Contracts at T-1 (5 contracts, ReAct full pipeline)

Analyze each contract 1 day before settlement using only news available at that date.

| Metric | Value |
|--------|-------|
| Signals triggered | 3 / 5 |
| **Accuracy** | **100%** (3/3) |
| **Cumulative PnL** | **+$138** |
| Sharpe ratio | **1.73** |
| Max drawdown | $0 |

The agent searched targeted queries like `"Trump 25% tariffs Canada Mexico"` and `"Trump tariffs Canada Mexico paused delayed"` — finding contract-relevant news that generic `"tariff trade policy"` queries would miss. 2 contracts were correctly skipped (edge < 10%).

### Backtest 2: Live Simulation (5 active contracts, 1 month, weekly rebalancing)

Simulate trading the top-5 active contracts over the past month with weekly analysis and dynamic position management.

| Metric | Value |
|--------|-------|
| Total trades | 14 |
| **Cumulative PnL** | **+$77** |
| Sharpe ratio | **1.03** |
| Max drawdown | $4 |
| Win rate | 71% |
| Losing contracts | **0 / 5** |

All 5 contracts were profitable (+$18, +$26, +$3, +$12, +$19). Reversal-based stop-losses prevented any contract from going negative.

### Key Findings

1. **Contract-specific search is critical.** Letting the LLM choose its own search queries (ReAct) produces +$138 PnL vs. -$83 with fixed keywords — a $221 improvement from the same underlying model.
2. **Active management beats hold-to-settlement.** Dynamic simulation with reversal exits: +$77. Previous hold-to-settlement backtest: -$147.
3. **The model knows when NOT to trade.** 2/5 resolved contracts were correctly skipped (edge too small), avoiding potential losses.

---

## System Architecture

```
DataAgent ──► AnalysisAgent (ReAct) ──► DecisionAgent ──► RiskAgent ──► FINAL DECISION
                   │                         │                │
                   ▼                         │                │
            ┌─────────────┐                  │                │
            │  Tools:      │                  │                │
            │  search_news │ (Google RSS)     │                │
            │  search_hist │ (ChromaDB)       │                │
            │  search_rag  │ (ChromaDB)       │                │
            └─────────────┘                  │                │
                                             │                │
                   LLM decides what to       │                │
                   search & how many times   │                │
```

**DataAgent** — Discovers active tariff/trade contracts on Polymarket via Gamma API. No news scraping (that's the AnalysisAgent's job now).

**AnalysisAgent (ReAct)** — The core innovation. A ReAct agent with 3 tools (`search_news`, `search_historical_events`, `search_recent_analyses`) that autonomously decides what queries to run based on the contract question. It typically searches 2-5 times with different keywords, then outputs a structured policy analysis with signal strength, affected countries, and landing probability.

**DecisionAgent** — Compares the AI's estimated probability against Polymarket odds. Outputs BUY_YES / BUY_NO / NO_TRADE with edge calculation and detailed reasoning.

**RiskAgent** — Reviews the analysis for contradictory signals (within the same week only — cross-week differences are treated as policy evolution, not contradictions), source reliability, and AI overconfidence. Can override BUY recommendations to NO_TRADE.

---

## Tech Stack

LangGraph + LangChain | Google Gemini 2.5 Flash | ChromaDB (RAG) | Streamlit + Plotly | Polymarket Gamma/CLOB API | Python 3.11+

---

## Quick Start

```bash
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key" > .env

python demo_tariff.py                                    # Full 4-agent pipeline
streamlit run dashboard.py                               # Interactive dashboard
python -m backtest.run_tariff_backtest --limit 5          # Resolved contract backtest
python -m backtest.run_live_simulation --limit 5          # Live simulation backtest
```

---

## Dashboard

Three-tab interactive dashboard for contract analysis and backtesting.

### Tab 1: Contract List
Browse all active tariff/trade contracts with probability chart and price history.

![Contract List](images/tab_contracts.png)

### Tab 2: AI Analysis
Select a contract → ReAct agent autonomously searches news → displays sentiment scores, policy signals, and trading recommendation.

![AI Analysis](images/tab_analysis.png)

### Tab 3: Backtest
Run resolved contract backtests (configurable T-N) or live simulations (configurable time range and rebalancing frequency).

![Backtest](images/tab_backtest.png)

---

## Project Structure

```
Polymarket-Agent/
├── agents/
│   ├── tariff_agent.py          # ReAct tools + analysis/decision chains
│   └── workflow.py              # LangGraph 4-agent pipeline
├── tools/
│   ├── news_scraper.py          # Google News RSS + White House + USTR
│   ├── polymarket_api.py        # Polymarket Gamma + CLOB API
│   ├── rag_store.py             # ChromaDB news storage
│   ├── tariff_history.py        # 17 historical tariff events (2018-2026)
│   └── news_tracker.py          # Sentiment history tracking
├── backtest/
│   ├── tariff_history.py        # 12 resolved contract ground truth
│   ├── run_tariff_backtest.py   # Resolved contract backtest (ReAct)
│   └── run_live_simulation.py   # Live simulation backtest (ReAct)
├── dashboard.py                 # Streamlit dashboard (3 tabs)
├── demo_tariff.py               # Full pipeline demo
├── monitor.py                   # CLI monitoring tool
└── config.py                    # API config + retry logic
```

---

## Limitations & Future Work

**Limitations:**
- Small sample (5+5 contracts) — results are directional, not statistically conclusive
- ReAct agent behavior is non-deterministic — same contract may get different search queries on re-run
- Google News RSS date filtering is imperfect — some results may leak across date boundaries
- All trades use flat $100 sizing; no Kelly criterion or volatility scaling

**Future work:**
- Multi-model ensemble (GPT-4 + Claude + Gemini) to reduce single-model overconfidence
- Pre-announcement signal detection (congressional hearings, Federal Register filings)
- Expand to sanctions, tech export controls, and Fed rate contracts
- Calibration training: use historical accuracy to adjust edge estimates
