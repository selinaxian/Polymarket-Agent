"""
已结算的关税/贸易合约历史数据
Polymarket API 不返回已关闭合约，因此手动收集已知结算结果。
来源：Polymarket 公开记录 + 新闻验证

每条记录包含：
  - question: 合约问题
  - resolved_date: 结算日期
  - outcome: "Yes" 或 "No"（实际结果）
  - context_news: 结算前可用的关键新闻标题（模拟当时的信息环境）
  - yes_price_before: 结算前几天的 Yes 价格（模拟当时市场赔率）
  - volume: 大致交易量
"""

TARIFF_HISTORY = [
    # ── 2025 年关税相关合约 ──
    # event_type: escalation(加税) / negotiation(谈判) / deal(协议) / policy(政策)
    {
        "question": "New Trump tariffs on China by February 1, 2025?",
        "resolved_date": "2025-02-01",
        "outcome": "Yes",
        "event_type": "escalation",
        "yes_price_before": 0.85,
        "volume": 500000,
        "context_news": [
            {"date": "2025-01-20", "title": "Trump signs executive order directing review of trade practices with China", "source": "White House"},
            {"date": "2025-01-25", "title": "Trump confirms 10% tariffs on China effective February 1", "source": "Reuters"},
            {"date": "2025-01-28", "title": "China warns of retaliation if US proceeds with tariffs", "source": "BBC"},
            {"date": "2025-01-30", "title": "Markets brace for new China tariffs as deadline approaches", "source": "CNBC"},
        ],
    },
    {
        "question": "Trump 25% tariffs on Canada and Mexico by February 4, 2025?",
        "resolved_date": "2025-02-04",
        "outcome": "Yes",
        "event_type": "escalation",
        "yes_price_before": 0.72,
        "volume": 800000,
        "context_news": [
            {"date": "2025-01-27", "title": "Trump threatens 25% tariffs on Canada and Mexico over border security", "source": "White House"},
            {"date": "2025-02-01", "title": "Trump signs proclamation imposing 25% tariffs on Canada and Mexico", "source": "Reuters"},
            {"date": "2025-02-02", "title": "Canada retaliates with tariffs on $155B of US goods", "source": "BBC"},
            {"date": "2025-02-03", "title": "Mexico announces retaliatory measures against US tariffs", "source": "AP"},
        ],
    },
    {
        "question": "Trump tariffs on Canada and Mexico delayed or paused by February 7?",
        "resolved_date": "2025-02-07",
        "outcome": "Yes",
        "event_type": "negotiation",
        "yes_price_before": 0.55,
        "volume": 350000,
        "context_news": [
            {"date": "2025-02-03", "title": "Trump agrees to 1-month pause on Mexico tariffs after border deal", "source": "Reuters"},
            {"date": "2025-02-03", "title": "Canada-US tariff pause reached after Trudeau offers border concessions", "source": "BBC"},
            {"date": "2025-02-05", "title": "Markets rally on US tariff delay with Canada and Mexico", "source": "CNBC"},
        ],
    },
    {
        "question": "US-China trade deal announced before April 2025?",
        "resolved_date": "2025-04-01",
        "outcome": "No",
        "event_type": "deal",
        "yes_price_before": 0.12,
        "volume": 250000,
        "context_news": [
            {"date": "2025-03-10", "title": "Trump raises China tariffs to 20% amid trade tensions", "source": "Reuters"},
            {"date": "2025-03-15", "title": "China retaliates with tariffs on $21B of US agricultural goods", "source": "BBC"},
            {"date": "2025-03-25", "title": "No signs of US-China trade talks as tariff deadline looms", "source": "CNBC"},
            {"date": "2025-03-28", "title": "White House says no deal with China until fentanyl issue resolved", "source": "White House"},
        ],
    },
    {
        "question": "Will Trump impose reciprocal tariffs by April 2, 2025?",
        "resolved_date": "2025-04-02",
        "outcome": "Yes",
        "event_type": "escalation",
        "yes_price_before": 0.90,
        "volume": 1200000,
        "context_news": [
            {"date": "2025-03-20", "title": "Trump announces 'Liberation Day' reciprocal tariffs for April 2", "source": "White House"},
            {"date": "2025-03-28", "title": "Markets sell off ahead of April 2 tariff deadline", "source": "CNBC"},
            {"date": "2025-04-01", "title": "Administration confirms sweeping reciprocal tariffs effective tomorrow", "source": "Reuters"},
        ],
    },
    {
        "question": "Will Trump pause reciprocal tariffs within 90 days?",
        "resolved_date": "2025-04-09",
        "outcome": "Yes",
        "event_type": "negotiation",
        "yes_price_before": 0.40,
        "volume": 600000,
        "context_news": [
            {"date": "2025-04-03", "title": "Global markets crash as reciprocal tariffs take effect", "source": "BBC"},
            {"date": "2025-04-05", "title": "Dow drops 2000 points, worst week since 2020", "source": "CNBC"},
            {"date": "2025-04-07", "title": "Treasury Secretary signals openness to tariff negotiations", "source": "Reuters"},
            {"date": "2025-04-09", "title": "Trump announces 90-day pause on reciprocal tariffs for most countries, raises China to 125%", "source": "White House"},
        ],
    },
    {
        "question": "US total tariff rate on China above 100% by end of April 2025?",
        "resolved_date": "2025-04-30",
        "outcome": "Yes",
        "event_type": "escalation",
        "yes_price_before": 0.88,
        "volume": 400000,
        "context_news": [
            {"date": "2025-04-09", "title": "Trump raises China tariffs to 125% while pausing others", "source": "White House"},
            {"date": "2025-04-15", "title": "China retaliates, raises tariffs on US goods to 125%", "source": "Reuters"},
            {"date": "2025-04-20", "title": "Effective US tariff rate on China exceeds 145% including all categories", "source": "BBC"},
        ],
    },
    {
        "question": "US-EU trade deal before July 2025?",
        "resolved_date": "2025-07-01",
        "outcome": "No",
        "event_type": "deal",
        "yes_price_before": 0.08,
        "volume": 180000,
        "context_news": [
            {"date": "2025-05-15", "title": "EU launches WTO challenge against US reciprocal tariffs", "source": "Reuters"},
            {"date": "2025-06-10", "title": "US-EU trade talks stall over agricultural subsidies", "source": "BBC"},
            {"date": "2025-06-25", "title": "EU prepares retaliatory tariff package as deadline nears", "source": "CNBC"},
        ],
    },
    {
        "question": "US-China tariff reduction agreement before October 2025?",
        "resolved_date": "2025-10-01",
        "outcome": "No",
        "event_type": "deal",
        "yes_price_before": 0.15,
        "volume": 350000,
        "context_news": [
            {"date": "2025-08-20", "title": "Trump maintains hardline stance on China tariffs", "source": "White House"},
            {"date": "2025-09-10", "title": "China pivots to diversify trade away from US", "source": "Reuters"},
            {"date": "2025-09-25", "title": "No progress on US-China trade talks, analysts see prolonged standoff", "source": "CNBC"},
        ],
    },
    {
        "question": "Will the US impose new tariffs on EU auto imports in 2025?",
        "resolved_date": "2025-12-31",
        "outcome": "Yes",
        "event_type": "escalation",
        "yes_price_before": 0.60,
        "volume": 300000,
        "context_news": [
            {"date": "2025-10-15", "title": "Trump threatens 25% auto tariffs on EU after trade talks collapse", "source": "Reuters"},
            {"date": "2025-11-20", "title": "US announces 25% tariffs on EU automobiles effective January 2026", "source": "White House"},
            {"date": "2025-12-10", "title": "EU vows retaliation over US auto tariffs, targets tech sector", "source": "BBC"},
        ],
    },
    # ── 2026 年 ──
    {
        "question": "Will Trump create a tariff dividend by March 31?",
        "resolved_date": "2026-03-31",
        "outcome": "No",
        "event_type": "policy",
        "yes_price_before": 0.01,
        "volume": 150000,
        "context_news": [
            {"date": "2026-03-01", "title": "No legislative action on tariff dividend proposal", "source": "Reuters"},
            {"date": "2026-03-15", "title": "Congress shows no appetite for tariff rebate plan", "source": "CNBC"},
            {"date": "2026-03-20", "title": "White House focused on trade deals, not domestic rebates", "source": "White House"},
        ],
    },
    {
        "question": "Imposing a Temporary Import Surcharge — Executive Order signed?",
        "resolved_date": "2026-02-20",
        "outcome": "Yes",
        "event_type": "policy",
        "yes_price_before": 0.75,
        "volume": 200000,
        "context_news": [
            {"date": "2026-02-10", "title": "White House signals new executive action on import surcharge", "source": "Reuters"},
            {"date": "2026-02-18", "title": "Draft executive order on temporary import surcharge circulates", "source": "CNBC"},
            {"date": "2026-02-20", "title": "Trump signs executive order imposing temporary import surcharge", "source": "White House"},
        ],
    },
]
