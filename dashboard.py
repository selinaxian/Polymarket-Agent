"""
Polymarket Tariff Policy Prediction — Streamlit Dashboard

Three tabs:
  1. 合约列表 (Contracts) — browse + select contracts
  2. AI 分析 (Analysis) — ReAct agent analyzes selected contract
  3. 回测 (Backtest) — simulate past performance with configurable params
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Tariff Prediction Agent", page_icon="🏛️", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Helper functions ──────────────────────────────────────

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


def contracts_to_df(contracts):
    rows = []
    for c in contracts:
        yp, np_ = parse_prices(c)
        rows.append({
            "合约": c["question"],
            "Yes": yp,
            "No": np_,
            "交易量 ($)": float(c.get("volume", 0) or 0),
            "截止日期": c.get("end_date", "")[:10],
        })
    return pd.DataFrame(rows)


# ── Data loaders (cached) ─────────────────────────────────

@st.cache_data(ttl=300, show_spinner="搜索关税/贸易合约...")
def load_tariff_contracts():
    from tools.polymarket_api import find_tariff_contracts
    raw = find_tariff_contracts()
    seen = set()
    return [c for c in raw if not (c["market_id"] in seen or seen.add(c["market_id"]))]


@st.cache_data(ttl=600, show_spinner="获取价格历史...")
def load_price_history(token_id: str):
    from tools.polymarket_api import get_price_history
    try:
        data = get_price_history(token_id, interval="all", fidelity=1440)
        if isinstance(data, dict) and "history" in data:
            return data["history"]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


# ── Sidebar ───────────────────────────────────────────────

st.sidebar.title("Tariff Prediction Agent")
st.sidebar.markdown("关税/贸易政策 AI 预测系统")


# ── Tabs ──────────────────────────────────────────────────

tab_contracts, tab_analysis, tab_backtest = st.tabs([
    "合约列表", "AI 分析", "回测",
])


# ============================================================
# Tab 1 — Contract List / 合约列表
# ============================================================

with tab_contracts:
    st.header("Polymarket 关税/贸易合约")
    contracts = load_tariff_contracts()

    if contracts:
        df = contracts_to_df(contracts)
        c1, c2 = st.columns(2)
        c1.metric("活跃合约数", len(df))
        c2.metric("总交易量", f"${df['交易量 ($)'].sum():,.0f}")

        st.dataframe(
            df.style.format({"Yes": "{:.1%}", "No": "{:.1%}", "交易量 ($)": "${:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

        # Plotly bar chart
        st.subheader("合约概率一览")
        fig_bar = go.Figure(go.Bar(
            x=df["Yes"],
            y=df["合约"],
            orientation="h",
            marker_color=["#2ecc71" if v > 0.5 else "#3498db" for v in df["Yes"]],
            text=[f"{v:.1%}" for v in df["Yes"]],
            textposition="auto",
        ))
        fig_bar.update_layout(
            xaxis_title="Yes 概率", yaxis_title="",
            xaxis_tickformat=".0%", height=max(300, len(df) * 35),
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Price history chart
        st.divider()
        st.subheader("合约价格走势")
        sel_chart = st.selectbox("选择合约查看走势", [c["question"] for c in contracts], key="chart_sel")
        target_chart = next(c for c in contracts if c["question"] == sel_chart)
        token_id = target_chart.get("token_ids", {}).get("yes", "")

        if token_id and len(token_id) > 10:
            history = load_price_history(token_id)
            if history:
                hist_df = pd.DataFrame(history)
                if "t" in hist_df.columns and "p" in hist_df.columns:
                    hist_df["time"] = pd.to_datetime(hist_df["t"], unit="s")
                    hist_df["price"] = hist_df["p"].astype(float)
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(
                        x=hist_df["time"], y=hist_df["price"],
                        mode="lines", name="Yes Price",
                        line=dict(color="#2ecc71", width=2),
                        fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
                    ))
                    fig_line.update_layout(
                        yaxis_title="Yes 价格", yaxis_tickformat=".0%",
                        height=350, margin=dict(l=10, r=10, t=10, b=30),
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.caption("价格数据格式不支持图表展示")
            else:
                st.caption("暂无历史价格数据")
        else:
            st.caption("该合约无 token ID，无法获取价格历史")
    else:
        st.warning("未找到关税/贸易相关合约")


# ============================================================
# Tab 2 — AI Analysis / AI 分析
# Select contract → ReAct agent searches & analyzes → show results
# ============================================================

with tab_analysis:
    st.header("AI 合约分析")

    if not contracts:
        st.warning("请先在「合约列表」页加载合约")
    else:
        # ── Contract selection ──
        questions = [c["question"] for c in contracts]
        selected_q = st.selectbox("选择要分析的合约", questions, key="analysis_sel")
        target = next(c for c in contracts if c["question"] == selected_q)
        yp, np_ = parse_prices(target)

        col1, col2, col3 = st.columns(3)
        col1.metric("Yes 价格", f"{yp:.1%}")
        col2.metric("No 价格", f"{np_:.1%}")
        col3.metric("交易量", f"${float(target.get('volume', 0)):,.0f}")

        st.divider()

        # ── Run analysis ──
        if st.button("运行 AI 分析", type="primary", key="run_analysis"):
            with st.spinner("ReAct Agent 正在自主搜索新闻和历史参考..."):
                from agents.tariff_agent import (
                    create_analysis_react_agent,
                    create_tariff_analysis_chain,
                    score_news_sentiment,
                    aggregate_sentiment,
                )
                from agents.workflow import _extract_json_from_text, _extract_news_from_tool_messages
                from config import invoke_with_retry

                react_agent = create_analysis_react_agent()
                react_input = {
                    "messages": [{
                        "role": "user",
                        "content": (
                            f"请分析以下 Polymarket 合约:\n\n"
                            f"合约问题: {selected_q}\n\n"
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

                    # Extract news from tool calls
                    searched_news = _extract_news_from_tool_messages(react_result["messages"])
                    tool_count = sum(1 for m in react_result["messages"] if hasattr(m, "type") and m.type == "tool")

                    # Parse analysis JSON
                    analysis = _extract_json_from_text(final_msg)
                    if not analysis:
                        st.error("无法解析 AI 分析结果，请重试")
                    else:
                        # Sentiment scoring on searched news
                        scored_news = score_news_sentiment(searched_news[:12])
                        sentiment_agg = aggregate_sentiment(scored_news)
                        analysis["sentiment"] = sentiment_agg

                        st.session_state["analysis"] = analysis
                        st.session_state["searched_news"] = searched_news
                        st.session_state["scored_news"] = scored_news
                        st.session_state["tool_count"] = tool_count
                        st.session_state["analysis_contract"] = selected_q

                except Exception as e:
                    st.error(f"分析失败: {e}")

        # ── Display results ──
        if "analysis" in st.session_state and st.session_state.get("analysis_contract") == selected_q:
            ta = st.session_state["analysis"]
            searched_news = st.session_state.get("searched_news", [])
            scored_news = st.session_state.get("scored_news", [])
            tool_count = st.session_state.get("tool_count", 0)

            st.success(f"ReAct Agent 完成: {tool_count} 次工具调用, 搜到 {len(searched_news)} 条新闻")

            # ── Policy signal ──
            st.subheader("政策信号分析")
            sc1, sc2 = st.columns(2)
            sc1.metric("政策信号", ta.get("policy_signal", "N/A"))
            strength = ta.get("signal_strength", 0)
            sc2.metric("保护主义强度", f"{strength:+d}/10" if isinstance(strength, int) else str(strength))
            st.info(f"**态势总结:** {ta.get('summary', 'N/A')}")

            # ── Searched news with sentiment ──
            st.subheader("搜索到的新闻 & 情绪评分")
            queries_used = ta.get("search_queries_used", [])
            if queries_used:
                st.caption(f"搜索词: {', '.join(queries_used)}")

            sentiment_data = ta.get("sentiment", {})
            if sentiment_data:
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("综合情绪", f"{sentiment_data.get('composite_sentiment', 0):+.3f}")
                sm2.metric("鹰派 / 鸽派", f"{sentiment_data.get('hawkish_count', 0)} / {sentiment_data.get('dovish_count', 0)}")
                sm3.metric("新闻数量", sentiment_data.get("news_count", 0))

            for sn in scored_news:
                score = sn.get("sentiment_score", 0)
                icon = "🔴" if score > 0.3 else ("🟢" if score < -0.3 else "🟡")
                st.markdown(f"{icon} **{score:+.1f}** — {sn['title'][:80]}")
                if sn.get("sentiment_reasoning"):
                    st.caption(f"  {sn['sentiment_reasoning'][:120]}")

            # ── Affected countries & sectors ──
            col_l, col_r = st.columns(2)
            with col_l:
                if ta.get("affected_countries"):
                    st.markdown("**受影响国家:**")
                    for ac in ta["affected_countries"][:6]:
                        icon = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(ac.get("impact"), "⚪")
                        st.markdown(f"{icon} **{ac['country']}** — {ac.get('details', '')[:120]}")
            with col_r:
                if ta.get("affected_sectors"):
                    st.markdown("**受影响行业:**")
                    for s in ta["affected_sectors"][:6]:
                        icon = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(s.get("impact"), "⚪")
                        st.markdown(f"{icon} **{s['sector']}** — {s.get('details', '')[:120]}")

            # ── Tariff actions ──
            if ta.get("tariff_actions"):
                st.markdown("**具体关税动作:**")
                for action in ta["tariff_actions"]:
                    st.markdown(f"- {action.get('action', '')} — `{action.get('status', '')}` | 落地概率: **{action.get('probability', '?')}%**")

            # ── Key signals ──
            if ta.get("key_signals"):
                st.markdown("**关键信号:**")
                for sig in ta["key_signals"]:
                    st.markdown(f"- {sig}")

            # ── Trading recommendation ──
            st.divider()
            st.subheader("交易建议")

            if st.button("生成交易建议", type="primary", key="run_decision"):
                with st.spinner("Gemini 生成交易建议..."):
                    from agents.tariff_agent import make_tariff_decision
                    dec = make_tariff_decision(
                        ta,
                        {"question": selected_q, "yes_price": yp, "no_price": np_,
                         "volume": str(target.get("volume", "N/A"))},
                    )
                    st.session_state["decision"] = dec

            if "decision" in st.session_state:
                dec = st.session_state["decision"]
                rec = dec.get("recommendation", "N/A")
                confidence = dec.get("confidence", 0)
                edge = dec.get("edge", 0)

                if rec == "BUY_YES":
                    st.success(f"建议: **买入 YES**  |  Edge: {edge} pp  |  置信度: {confidence}%")
                elif rec == "BUY_NO":
                    st.error(f"建议: **买入 NO**  |  Edge: {edge} pp  |  置信度: {confidence}%")
                else:
                    st.info("建议: 不交易（无明显优势）")

                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("市场 Yes", f"{dec.get('market_yes_price', 0):.1%}")
                dc2.metric("AI 估计", f"{dec.get('our_estimated_probability', 0)}%")
                dc3.metric("Edge", f"{edge} pp")

                st.markdown(f"**推理:** {dec.get('reasoning', 'N/A')}")

                if dec.get("risk_factors"):
                    st.markdown("**风险因素:**")
                    for r in dec["risk_factors"]:
                        st.markdown(f"- {r}")


# ============================================================
# Tab 3 — Backtest / 回测
# ============================================================

with tab_backtest:
    st.header("回测")

    bt_tab_resolved, bt_tab_live = st.tabs(["已结算合约回测", "动态模拟回测"])

    # ── Sub-tab 1: Resolved contract backtest ──
    with bt_tab_resolved:
        st.subheader("已结算合约回测（无 look-ahead bias）")
        st.caption("对 12 个已结算关税合约，用 Google News RSS 按日期搜索历史新闻进行分析。")

        from backtest.tariff_history import TARIFF_HISTORY

        # Dataset overview
        with st.expander("回测数据集", expanded=False):
            bt_rows = []
            for h in TARIFF_HISTORY:
                bt_rows.append({
                    "结算日": h["resolved_date"],
                    "合约": h["question"][:50],
                    "结果": h["outcome"],
                    "类型": h.get("event_type", ""),
                    "结算前Yes": h["yes_price_before"],
                })
            st.dataframe(
                pd.DataFrame(bt_rows).style.format({"结算前Yes": "{:.0%}"}),
                use_container_width=True, hide_index=True,
            )

        # Controls
        rc1, rc2, rc3 = st.columns(3)
        bt_limit = rc1.number_input("合约数", min_value=1, max_value=len(TARIFF_HISTORY), value=len(TARIFF_HISTORY), key="bt_limit")
        bt_days = rc2.selectbox("分析时间点", [1, 3, 5, 7, 14], index=3, format_func=lambda d: f"T-{d}", key="bt_days")

        if rc3.button("运行回测", type="primary", key="run_resolved_bt"):
            with st.spinner(f"ReAct 全流程回测中（{bt_limit} 个合约，T-{bt_days}）..."):
                from backtest.run_tariff_backtest import run_tariff_backtest
                results = run_tariff_backtest(
                    cases=TARIFF_HISTORY[:bt_limit],
                    limit=bt_limit,
                    days_before=bt_days,
                )
                bt_key = f"bt_results_T{bt_days}"
                st.session_state[bt_key] = results
                st.session_state["bt_results"] = results

        # Display results
        if "bt_results" in st.session_state:
            results = st.session_state["bt_results"]
            valid = [r for r in results if r.get("status") == "ok"]
            triggered = [r for r in valid if r["signal_triggered"]]
            correct = [r for r in triggered if r["direction_correct"]]

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("回测合约", len(valid))
            m2.metric("信号触发", len(triggered))
            m3.metric("方向正确", len(correct))
            accuracy = len(correct) / len(triggered) * 100 if triggered else 0
            m4.metric("准确率", f"{accuracy:.0f}%")

            # PnL
            from backtest.run_tariff_backtest import compute_pnl, threshold_sensitivity

            pnl = compute_pnl(valid)
            if pnl["trades"]:
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("累计 PnL", f"${pnl['cumulative_pnl']:+.0f}")
                p2.metric("Sharpe", f"{pnl['sharpe']:.2f}")
                p3.metric("最大回撤", f"${pnl['max_drawdown']:.0f}")
                p4.metric("胜率", f"{pnl['win_rate']:.0f}%")

                # Cumulative PnL chart
                if pnl.get("cumulative_series"):
                    cum_df = pd.DataFrame({
                        "Trade #": list(range(1, len(pnl["cumulative_series"]) + 1)),
                        "PnL ($)": pnl["cumulative_series"],
                    })
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Scatter(
                        x=cum_df["Trade #"], y=cum_df["PnL ($)"],
                        mode="lines+markers",
                        line=dict(color="#2ecc71" if pnl["cumulative_pnl"] >= 0 else "#e74c3c", width=3),
                        fill="tozeroy",
                        fillcolor="rgba(46,204,113,0.1)" if pnl["cumulative_pnl"] >= 0 else "rgba(231,76,60,0.1)",
                    ))
                    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_pnl.update_layout(
                        xaxis_title="Trade #", yaxis_title="Cumulative PnL ($)",
                        height=300, margin=dict(l=10, r=10, t=10, b=30),
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)

            # Threshold sensitivity
            st.subheader("Edge 阈值敏感性")
            sens = threshold_sensitivity(valid, thresholds=[5.0, 10.0, 15.0, 20.0, 25.0])
            sens_df = pd.DataFrame(sens)
            sens_df.columns = ["阈值(%)", "信号数", "正确", "准确率(%)", "PnL($)", "回撤($)", "Sharpe", "胜率(%)"]
            st.dataframe(
                sens_df.style.format({
                    "阈值(%)": "{:.0f}%", "准确率(%)": "{:.1f}%", "PnL($)": "${:+.0f}",
                    "回撤($)": "${:.0f}", "Sharpe": "{:.2f}", "胜率(%)": "{:.0f}%",
                }),
                use_container_width=True, hide_index=True,
            )

            # T-N comparison
            comparison_keys = [k for k in st.session_state if k.startswith("bt_results_T")]
            if len(comparison_keys) >= 2:
                st.subheader("多时间点对比")
                comp_rows = []
                for key in sorted(comparison_keys):
                    label = key.replace("bt_results_", "")
                    res = st.session_state[key]
                    res_ok = [r for r in res if r.get("status") == "ok"]
                    res_trig = [r for r in res_ok if r["signal_triggered"]]
                    res_correct = [r for r in res_trig if r["direction_correct"]]
                    pnl_sum = sum(r["pnl"] for r in res_trig)
                    comp_rows.append({
                        "分析时间": label,
                        "信号数": len(res_trig),
                        "正确": len(res_correct),
                        "准确率": len(res_correct) / len(res_trig) * 100 if res_trig else 0,
                        "PnL ($)": round(pnl_sum, 0),
                    })
                st.dataframe(
                    pd.DataFrame(comp_rows).style.format({"准确率": "{:.1f}%", "PnL ($)": "${:+.0f}"}),
                    use_container_width=True, hide_index=True,
                )

            # Detail table
            with st.expander("逐条详情"):
                detail_rows = []
                for r in valid:
                    status = ("OK" if r["direction_correct"] else "MISS") if r["signal_triggered"] else "skip"
                    pnl_str = f"${r['pnl']:+.0f}" if r["signal_triggered"] else "-"
                    detail_rows.append({
                        "结算日": r["resolved_date"], "分析日": r.get("analysis_date", ""),
                        "合约": r["question"][:40], "实际": r["actual_outcome"],
                        "AI建议": r["recommendation"], "Edge": r["edge"],
                        "PnL": pnl_str, "结果": status,
                    })
                st.dataframe(
                    pd.DataFrame(detail_rows).style.format({"Edge": "{:+.1f}%"}),
                    use_container_width=True, hide_index=True,
                )

    # ── Sub-tab 2: Live simulation backtest ──
    with bt_tab_live:
        st.subheader("动态模拟回测")
        st.caption("对活跃合约用历史价格 + 历史新闻做模拟交易（无 look-ahead bias）")

        lc1, lc2, lc3, lc4 = st.columns(4)
        sim_months = lc1.selectbox("回测时间", [1, 2, 3], index=2, format_func=lambda m: f"{m} 个月", key="sim_months")
        sim_interval = lc2.selectbox("调仓频率", [7, 14, 21], index=1, format_func=lambda d: f"每 {d} 天", key="sim_interval")
        sim_limit = lc3.number_input("合约数量", min_value=1, max_value=15, value=5, key="sim_limit")

        if lc4.button("运行模拟", type="primary", key="run_live_bt"):
            with st.spinner(f"模拟回测中（{sim_limit} 个合约，{sim_months} 个月，每 {sim_interval} 天调仓）..."):
                from backtest.run_live_simulation import run_live_simulation, SIM_START
                from datetime import timedelta
                import backtest.run_live_simulation as sim_module

                # Set simulation params
                today = datetime.now()
                sim_module.SIM_START = today - timedelta(days=sim_months * 30)
                sim_module.SIM_INTERVAL_DAYS = sim_interval

                result = run_live_simulation(limit=sim_limit)
                st.session_state["sim_result"] = result

                # Restore defaults
                sim_module.SIM_START = SIM_START

        if "sim_result" in st.session_state:
            result = st.session_state["sim_result"]
            summary = result["summary"]
            trades = result["trades"]
            sim_contracts = result["contracts"]

            # Metrics
            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("合约数", len(sim_contracts))
            sm2.metric("累计 PnL", f"${summary.get('cumulative_pnl', 0):+.0f}")
            sm3.metric("Sharpe", f"{summary.get('sharpe', 0):.2f}")
            sm4.metric("最大回撤", f"${summary.get('max_drawdown', 0):.0f}")
            sm5.metric("胜率", f"{summary.get('win_rate', 0):.0f}%")

            # Per-contract PnL
            st.subheader("逐合约 PnL")
            contract_pnl = {}
            for t in trades:
                q = t["question"]
                contract_pnl[q] = contract_pnl.get(q, 0) + t["pnl"]

            cpnl_df = pd.DataFrame([
                {"合约": q, "PnL ($)": round(p, 0)}
                for q, p in contract_pnl.items()
            ])
            if not cpnl_df.empty:
                fig_cpnl = go.Figure(go.Bar(
                    x=cpnl_df["合约"],
                    y=cpnl_df["PnL ($)"],
                    marker_color=["#2ecc71" if p >= 0 else "#e74c3c" for p in cpnl_df["PnL ($)"]],
                    text=[f"${p:+.0f}" for p in cpnl_df["PnL ($)"]],
                    textposition="auto",
                ))
                fig_cpnl.update_layout(
                    yaxis_title="PnL ($)", height=350,
                    margin=dict(l=10, r=10, t=10, b=30),
                )
                fig_cpnl.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_cpnl, use_container_width=True)

            # Trade log
            with st.expander("完整交易日志"):
                if trades:
                    log_rows = []
                    for t in trades:
                        log_rows.append({
                            "日期": t["date"],
                            "操作": t["action"],
                            "合约": t["question"],
                            "价格": f"{t['price']:.1%}",
                            "PnL": f"${t['pnl']:+.0f}" if t["pnl"] != 0 else "-",
                            "原因": t["reason"][:60],
                        })
                    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("无交易记录")
