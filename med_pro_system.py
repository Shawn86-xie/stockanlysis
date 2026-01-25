import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# 导入自定义模块
from config import load_config, save_config, save_trade, delete_trade, clear_portfolio
from data_fetcher import fetch_stock_data, get_signals, compute_portfolio_stats, monte_carlo_simulation, get_recent_10_days, analyze_portfolio, get_portfolio_summary, calculate_deviation, fetch_latest_prices
from ai_analyzer import deepseek_analyze
from backtester import generate_signal_series, run_vectorized_backtest, plot_backtest_results, plot_backtest_with_divergence, rsi_grid_search, grid_search_with_backtest, detect_rsi_divergence, plot_divergence_chart, calculate_rsi
from math_engine import fit_trend_line, detect_peaks_valleys
from font_utils import setup_chinese_font

# --- 环境与中文字体配置 ---
# Plotly 图表配置 (符合 Streamlit 2026 标准)
PLOTLY_CONFIG = {
    'displayModeBar': True,    # 显示工具栏
    'displaylogo': False,      # 隐藏 Plotly Logo
    'modeBarButtonsToRemove': ['lasso2d'],  # 保留select2d框选工具
    'responsive': True         # 自适应容器
}

# 自动配置中文字体，优先检测Linux系统路径下的中文字体（如Noto Sans CJK）
font_name, font_path = setup_chinese_font()
if font_name:
    print(f"已使用中文字体: {font_name}")
else:
    print("使用默认字体配置")
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="2026 量化决策看板", layout="wide")

# --- 加载配置 ---
config = load_config()

# --- 标的库定义 (Session State) ---
if 'master_pool' not in st.session_state:
    # 从配置加载标的库，如果不存在则使用默认值
    config_master_pool = config.get('master_pool')
    if config_master_pool:
        st.session_state.master_pool = config_master_pool
    else:
        st.session_state.master_pool = {
            "核心组合": {
                "300347": "泰格医药",
                "688506": "百利天恒",
                "603259": "药明康德",
                "600276": "恒瑞医药",
                "301293": "三博脑科",
                "000538": "云南白药",
                "600521": "华海药业"
            },
            "科研/肾科": {
                "300453": "三鑫医疗",
                "002223": "鱼跃医疗",
                "603108": "润达医疗"
            }
        }

# --- 加载用户保存的选择和持仓 ---
if 'user_settings' not in st.session_state:
    st.session_state.user_settings = config.get('user_settings', {})
    
# 自动加载保存的标的勾选状态
if st.session_state.user_settings:
    saved_codes = st.session_state.user_settings.get('selected_codes', [])
    # 恢复选择状态
    for cat, stocks in st.session_state.master_pool.items():
        for c, n in stocks.items():
            st.session_state[f"sel_{c}"] = c in saved_codes

# --- Streamlit 侧边栏 ---
st.sidebar.title("🤖 决策中心 (2026)")
ds_key = st.sidebar.text_input("DeepSeek Key", value=config.get("deepseek_api_key", ""), type="password",
                               help="用于访问DeepSeek API的密钥，获取AI分析的新闻资讯。")
st.sidebar.markdown("---")

# 标的库管理
with st.sidebar.expander("📚 标的库管理", expanded=False):
    st.write("在这里添加或删除标的。")
    
    # 添加新标的
    st.subheader("添加新标的")
    
    # 股票代码搜索
    search_query = st.text_input("🔍 搜索股票代码或名称", placeholder="输入代码或名称，如 '300347' 或 '泰格医药'", key="stock_search")
    
    search_results = []
    if search_query and len(search_query.strip()) > 0:
        from data_fetcher import search_stock_info
        search_results = search_stock_info(search_query.strip())
    
    # 搜索结果展示
    selected_stock = None
    if search_results:
        st.info(f"找到 {len(search_results)} 个匹配的股票")
        # 创建选择列表
        stock_options = [f"{item['code']} - {item['name']}" for item in search_results]
        selected_option = st.selectbox("选择股票", options=stock_options, key="stock_select")
        
        if selected_option:
            # 解析选择的股票
            code, name = selected_option.split(" - ", 1)
            selected_stock = {'code': code, 'name': name}
            st.success(f"已选择: {name} ({code})")
    
    col_code, col_name, col_cat = st.columns(3)
    with col_code:
        if selected_stock:
            new_code = st.text_input("股票代码", value=selected_stock['code'], placeholder="例如: 000001", key="new_code")
        else:
            new_code = st.text_input("股票代码", value="", placeholder="例如: 000001", key="new_code")
    with col_name:
        if selected_stock:
            new_name = st.text_input("股票名称", value=selected_stock['name'], placeholder="例如: 平安银行", key="new_name")
        else:
            new_name = st.text_input("股票名称", value="", placeholder="例如: 平安银行", key="new_name")
    with col_cat:
        new_category = st.selectbox("分类", ["核心组合", "科研/肾科", "其他"], key="new_category")
    
    if st.button("➕ 添加标的", type="secondary", width='stretch'):
        if new_code and new_name:
            # 确保分类存在
            if new_category not in st.session_state.master_pool:
                st.session_state.master_pool[new_category] = {}
            
            # 检查是否已存在
            if new_code in st.session_state.master_pool[new_category]:
                st.warning(f"股票代码 {new_code} 已在 {new_category} 中存在！")
            else:
                # 添加标的
                st.session_state.master_pool[new_category][new_code] = new_name
                st.success(f"已添加 {new_name}({new_code}) 到 {new_category}")
                # 重新加载页面
                st.rerun()
        else:
            st.warning("请填写股票代码和名称")
    
    st.subheader("删除标的")
    # 显示现有标的供删除
    for cat, stocks in list(st.session_state.master_pool.items()):
        st.write(f"**{cat}**")
        for c, n in list(stocks.items()):
            if st.button(f"🗑️ 删除 {n} ({c})", key=f"del_{cat}_{c}"):
                del st.session_state.master_pool[cat][c]
                # 如果分类为空，删除分类
                if not st.session_state.master_pool[cat]:
                    del st.session_state.master_pool[cat]
                st.success(f"已删除 {n}({c})")
                st.rerun()
    
    # 保存标的库按钮
    st.subheader("保存标的库")
    if st.button("💾 保存标的库到配置文件", type="primary", width='stretch'):
        config['master_pool'] = st.session_state.master_pool
        save_config(config)
        st.success("标的库已保存到配置文件！")
        st.info("应用将重新加载以使用新的标的库...")
        st.rerun()

st.sidebar.markdown("---")

# --- 交易录入界面 ---
with st.sidebar.expander("📝 录入新交易", expanded=False):
    st.subheader("录入新交易")
    
    # 从现有标的库中选择股票
    all_stocks = []
    for cat, stocks in st.session_state.master_pool.items():
        for code, name in stocks.items():
            all_stocks.append({"code": code, "name": name, "category": cat})
    
    if all_stocks:
        # 创建选择列表
        stock_options = [f"{item['code']} - {item['name']} ({item['category']})" for item in all_stocks]
        selected_option = st.selectbox("选择标的", options=stock_options, key="trade_stock_select")
        
        if selected_option:
            # 解析选择的股票
            parts = selected_option.split(" - ")
            code = parts[0]
            name = parts[1].split(" (")[0]
            
            # 显示已选择的信息
            st.info(f"已选择: {name} ({code})")
            
            col_price, col_qty = st.columns(2)
            with col_price:
                buy_price = st.number_input("买入均价 (元)", min_value=0.01, value=100.0, step=0.01, key="buy_price")
            with col_qty:
                quantity = st.number_input("买入数量 (股)", min_value=1, value=100, step=100, key="quantity")
            
            col_date, col_time = st.columns(2)
            with col_date:
                buy_date = st.date_input("买入日期", value=datetime.now(), key="buy_date")
            with col_time:
                # 时间戳使用当前时间
                timestamp = int(datetime.now().timestamp())
                st.text(f"时间戳: {timestamp}")
            
            if st.button("💾 保存交易记录", type="primary", width='stretch'):
                # 数据完整性检查
                if buy_price <= 0:
                    st.error("买入均价必须大于0")
                elif quantity <= 0:
                    st.error("买入数量必须大于0")
                else:
                    trade_data = {
                        "code": code,
                        "name": name,
                        "buy_price": float(buy_price),
                        "quantity": int(quantity),
                        "buy_date": buy_date.strftime("%Y-%m-%d"),
                        "timestamp": timestamp
                    }
                    
                    # 保存交易记录
                    config = load_config()  # 重新加载最新配置
                    config = save_trade(config, trade_data)
                    st.success("交易记录已保存！")
                    # 更新session_state中的portfolio
                    if 'portfolio' not in st.session_state:
                        st.session_state.portfolio = []
                    st.session_state.portfolio = config.get('portfolio', [])
    
    # 显示当前持仓记录
    st.subheader("当前持仓记录")
    config = load_config()
    portfolio = config.get('portfolio', [])
    
    if portfolio:
        for idx, trade in enumerate(portfolio):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.text(f"{trade['name']} ({trade['code']})")
                st.caption(f"{trade['quantity']}股 @ {trade['buy_price']}元")
            with col2:
                st.caption(trade['buy_date'])
            with col3:
                if st.button("🗑️", key=f"del_trade_{idx}", help="删除此记录"):
                    config = delete_trade(config, idx)
                    st.success("记录已删除")
                    st.rerun()
        
        # 一键清空按钮
        if st.button("⚠️ 一键清空持仓", type="secondary", width='stretch'):
            config = clear_portfolio(config)
            st.success("持仓记录已清空")
            st.rerun()
    else:
        st.info("暂无持仓记录")

st.sidebar.markdown("---")

# 风险参数设置（先定义，以便在保存按钮中使用）
stop_loss_val = st.sidebar.slider("风险提醒阈值 (止损)", -0.10, -0.01, config.get("stop_loss_threshold", -0.05),
                                   help="当日跌幅超过此阈值时触发风险提醒。例如-5%表示当日下跌5%以上时提示止损。")
sim_num = st.sidebar.number_input("模拟次数", value=config.get("default_sim_count", 3000),
                                   help="蒙特卡洛模拟中随机生成的投资组合数量。模拟次数越多，结果越精确，但计算时间越长。")

# 新闻数量设置
news_count = st.sidebar.number_input("新闻获取数量", min_value=1, max_value=20, 
                                     value=config.get("news_count", 5),
                                     help="每个标的获取的最新新闻数量，默认为5条。",
                                     key="news_count_input")

st.sidebar.markdown("---")

# 设置管理按钮
col_save, col_load = st.sidebar.columns(2)
with col_save:
    if st.button("💾 保存当前设置", width='stretch'):
        # 收集当前选择
        selected_codes = []
        for cat, stocks in st.session_state.master_pool.items():
            for c, n in stocks.items():
                if st.session_state.get(f"sel_{c}", True):
                    selected_codes.append(c)
        
        # 收集持仓（从session_state获取）
        holdings_save = {}
        selected_names = [st.session_state.master_pool[cat][c] for cat in st.session_state.master_pool for c in selected_codes if c in st.session_state.master_pool[cat]]
        for name in selected_names:
            holdings_save[name] = st.session_state.get(f"hold_{name}", 0)
        
        # 保存到配置
        st.session_state.user_settings = {
            'selected_codes': selected_codes,
            'holdings': holdings_save,
            'stop_loss': stop_loss_val,
            'sim_num': sim_num,
            'news_count': news_count
        }
        
    # 保存DeepSeek密钥到配置
    if ds_key:
        config['deepseek_api_key'] = ds_key
    
    # 更新全局配置并保存到文件
    config['user_settings'] = st.session_state.user_settings
    config['news_count'] = news_count
    save_config(config)
    st.sidebar.success("设置已保存！")

with col_load:
    if st.button("📂 加载保存的设置", width='stretch'):
        if st.session_state.user_settings:
            # 恢复选择状态
            saved_codes = st.session_state.user_settings.get('selected_codes', [])
            for cat, stocks in st.session_state.master_pool.items():
                for c, n in stocks.items():
                    st.session_state[f"sel_{c}"] = c in saved_codes
            
            # 恢复持仓
            saved_holdings = st.session_state.user_settings.get('holdings', {})
            for name, qty in saved_holdings.items():
                st.session_state[f"hold_{name}"] = qty
            
            # 恢复新闻数量设置
            saved_news_count = st.session_state.user_settings.get('news_count', 5)
            st.session_state["news_count_input"] = saved_news_count
            
            st.sidebar.success("设置已加载！")
        else:
            st.sidebar.warning("没有找到保存的设置")

st.sidebar.markdown("---")

# 标的选择
final_sel = {}
for cat, stocks in st.session_state.master_pool.items():
    st.sidebar.write(f"**{cat}**")
    for c, n in stocks.items():
        # 从session_state获取选择状态，默认为True
        checkbox_state = st.session_state.get(f"sel_{c}", True)
        if st.sidebar.checkbox(f"{n} ({c})", value=checkbox_state, key=f"sel_{c}_ui"):
            final_sel[c] = n
            # 同步状态到session_state
            st.session_state[f"sel_{c}"] = True
        else:
            st.session_state[f"sel_{c}"] = False

st.sidebar.markdown("---")
# 开始统计按钮
if st.sidebar.button("🚀 开始统计", type="primary", width='stretch'):
    # 保存当前勾选状态到配置文件
    selected_codes = []
    for cat, stocks in st.session_state.master_pool.items():
        for c, n in stocks.items():
            if st.session_state.get(f"sel_{c}", True):
                selected_codes.append(c)
    
    # 收集其他设置
    holdings_save = {}
    selected_names = [st.session_state.master_pool[cat][c] for cat in st.session_state.master_pool for c in selected_codes if c in st.session_state.master_pool[cat]]
    for name in selected_names:
        holdings_save[name] = st.session_state.get(f"hold_{name}", 0)
    
    # 更新user_settings
    st.session_state.user_settings = {
        'selected_codes': selected_codes,
        'holdings': holdings_save,
        'stop_loss': stop_loss_val,
        'sim_num': sim_num,
        'news_count': news_count
    }
    
    # 保存DeepSeek密钥到配置
    if ds_key:
        config['deepseek_api_key'] = ds_key
    
    # 更新全局配置并保存到文件
    config['user_settings'] = st.session_state.user_settings
    config['news_count'] = news_count
    save_config(config)
    
    # 设置分析状态
    st.session_state.run_analysis = True
    st.session_state.selected_stocks = final_sel.copy()
    st.sidebar.success("开始分析选中的标的，且勾选状态已自动保存！")
    # 注意：这里不能直接rerun，因为需要先保存session_state
    # 我们将在主界面中根据run_analysis状态来触发分析

# 如果之前有分析运行状态，但用户取消了所有选择，则清除分析状态
if not final_sel and st.session_state.get('run_analysis', False):
    st.session_state.run_analysis = False
    st.session_state.selected_stocks = {}

# --- 主界面 ---
st.title("🏥 量化研报系统 V1.0 20260108")
st.caption(f"科研工作者专属调仓决策工具 | 当前配置生效日期: {datetime.now().strftime('%Y-%m-%d')}")

# 检查是否应该运行分析
should_run_analysis = st.session_state.get('run_analysis', False) and st.session_state.get('selected_stocks', {})
analysis_stocks = st.session_state.get('selected_stocks', {})

if should_run_analysis and analysis_stocks:
    # 获取数据
    data = fetch_stock_data(list(analysis_stocks.keys()), list(analysis_stocks.values()))
    if not data.empty:
        # 检查实际获取到的股票数据，更新analysis_stocks
        actual_stocks = {}
        for code, name in analysis_stocks.items():
            if name in data.columns:
                actual_stocks[code] = name
            else:
                st.warning(f"股票 {name}({code}) 数据获取失败，已从分析中排除。")
        # 如果没有成功获取到任何股票，则报错并停止
        if len(actual_stocks) == 0:
            st.error("未能获取到任何股票数据，请检查网络连接或股票代码。")
            st.stop()
        # 更新analysis_stocks为实际获取到的股票
        analysis_stocks = actual_stocks
        # 确保data只包含actual_stocks中的股票（实际上已经如此，但为了安全）
        data = data[list(actual_stocks.values())]
        # 计算收益率
        returns = data.pct_change().dropna()
        # 投资组合统计
        mean_ret, cov_mat = compute_portfolio_stats(returns, config.get("risk_free_rate", 0.0188))
        # 蒙特卡洛模拟
        sim_res = monte_carlo_simulation(mean_ret, cov_mat, sim_num, config.get("risk_free_rate", 0.0188))
        best_p = sim_res.iloc[sim_res['Sharpe'].idxmax()]

        # 功能标签页
        t1, t2, t3, t4, t5, t6, t7 = st.tabs(["💡 AI 资讯深度研判", "🚦 买卖与风险预警", "🕸️ 板块相关性分析", "📊 权重优化实验", "📈 模拟交易", "📈 实盘持仓监测", "📈 独立 K 线分析"])

        with t1:
            st.subheader("DeepSeek专业资讯评分")
            st.info("AI资讯分析需要调用外部API，耗时较长，请手动点击按钮获取。")
            
            # 初始化session_state存储新闻和分析结果
            if 'news_results' not in st.session_state:
                st.session_state.news_results = {}
            
            # 按钮触发获取新闻
            if st.button("🔍 获取最新新闻并分析", type="primary"):
                with st.spinner("正在获取新闻并分析..."):
                    for c, n in analysis_stocks.items():
                        try:
                            import akshare as ak
                            news = ak.stock_news_em(symbol=c).head(news_count)
                            news_list = []
                            for _, row in news.iterrows():
                                # 获取发布日期（假设列名为'新闻发布时间'，如果没有则使用None）
                                publish_date = row.get('新闻发布时间')
                                # 调用更新后的deepseek_analyze函数，返回字典
                                analysis_result = deepseek_analyze(ds_key, row['新闻标题'], n, publish_date)
                                news_list.append({
                                    'title': row['新闻标题'],
                                    'stars': analysis_result['stars'],
                                    'nature': analysis_result['nature'],
                                    'color': analysis_result['color'],
                                    'reason': analysis_result['reason'],
                                    'summary': analysis_result['summary'],
                                    'date': analysis_result['date'],
                                    'url': row.get('文章链接', '')
                                })
                            st.session_state.news_results[c] = {
                                'name': n,
                                'news': news_list
                            }
                        except Exception as e:
                            st.error(f"获取 {n} 的新闻失败：{e}")
            
            # 显示已存储的结果
            for c, news_entry in st.session_state.news_results.items():
                if news_entry['name'] in analysis_stocks.values():  # 只显示当前选中的标的
                    with st.expander(f"📌 {news_entry['name']} ({c}) - 资讯洞察"):
                        for item in news_entry['news']:
                            # 使用两列布局：左侧标题和摘要，右侧评分和日期
                            col_left, col_right = st.columns([3, 1])
                            with col_left:
                                st.markdown(f"#### {item['title']}")
                                st.caption(f"**发布日期**: {item['date']}")
                                st.write(f"**摘要**: {item['summary']}")
                                st.write(f"**分析**: {item['reason']}")
                                if item['url']:
                                    st.caption(f"[原文链接]({item['url']})")
                            with col_right:
                                st.markdown(f":{item['color']}[{item['stars']}]")
                                st.caption(f"**性质**: {item['nature']}")

        with t2:
            st.subheader("量化决策与风险管理")
            st.caption("""
            **列名解释**：
            - **标的**：股票名称
            - **价格**：当前收盘价（元）
            - **当日涨跌**：今日相对于昨日的涨跌幅度
            - **RSI**：相对强弱指标，>70表示超买，<30表示超卖
            - **建议**：基于RSI和均线的买卖建议
            - **风险**：风险提醒，‼️表示触及止损阈值
            - **最优配比**：AI推荐的最优投资组合权重
            """)
            sig_df = get_signals(data, best_p[list(analysis_stocks.values())].to_dict(), stop_loss_val)
            # 检查信号数据是否为空
            if sig_df.empty:
                st.warning("未能生成交易信号，可能是数据获取失败或计算错误。")
            else:
                # 对触及止损的行进行高亮，使用高对比度颜色
                def highlight_risk(val):
                    if '‼️' in str(val):
                        # 警示文本使用红色，加粗提高可见性
                        return 'color: #ff4444; font-weight: bold'
                    else:
                        # 普通文本使用白色，确保在深色背景下可见
                        return 'color: #ffffff'
                
                # 检查'风险'列是否存在
                if '风险' in sig_df.columns:
                    st.table(sig_df.style.map(highlight_risk, subset=['风险']))
                else:
                    st.table(sig_df)
            
            # 初始化回测参数
            if 'backtest_days' not in st.session_state:
                st.session_state.backtest_days = 250
            if 'backtest_initial_capital' not in st.session_state:
                st.session_state.backtest_initial_capital = 100000
            if 'backtest_rsi_buy' not in st.session_state:
                st.session_state.backtest_rsi_buy = 35.0
            if 'backtest_rsi_sell' not in st.session_state:
                st.session_state.backtest_rsi_sell = 75.0
            if 'backtest_results' not in st.session_state:
                st.session_state.backtest_results = {}
            
            # 准备交互式图表的数据
            cum_returns = (1 + returns).cumprod()
            stock_names = list(analysis_stocks.values())
            
            def render_interactive_plot(cum_returns, stock_names):
                highlighted = st.selectbox(
                    "选择要突出显示的股票（或点击图例隐藏/显示）",
                    options=stock_names,
                    key='chart_highlight_select'
                )
                
                fig = go.Figure()
                for name in cum_returns.columns:
                    line_width = 4 if name == highlighted else 1.5
                    line_opacity = 1.0 if name == highlighted else 0.4
                    fig.add_trace(go.Scatter(
                        x=cum_returns.index,
                        y=cum_returns[name],
                        name=name,
                        line=dict(width=line_width),
                        opacity=line_opacity,
                    ))
                
                fig.update_layout(
                    title=f"累积收益率曲线 (当前高亮: {highlighted})",
                    xaxis_title="日期",
                    yaxis_title="累积收益率",
                    hovermode="x unified",
                    showlegend=True,
                )
                st.plotly_chart(fig, config=PLOTLY_CONFIG)
            
            render_interactive_plot(cum_returns, stock_names)
            
            # 显示最近10个交易日价格数据
            st.subheader("📅 最近10个交易日价格")
            recent_data = get_recent_10_days(data)
            # 仅对数值列应用格式化，避免字符串列格式化错误
            numeric_cols = recent_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(recent_data.style.format("{:.2f}", subset=numeric_cols))
            else:
                st.dataframe(recent_data)
            
            # 回测功能
            st.subheader("📊 历史回测引擎（验证策略有效性）")
            st.caption("选择一只股票，基于RSI和移动平均线策略进行历史回测，验证策略在历史数据上的表现。")
            
            # 选择要回测的股票
            if len(stock_names) > 0:
                backtest_stock = st.selectbox("选择要回测的股票", options=stock_names, key="backtest_stock_select")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    backtest_days = st.number_input("回测历史天数", min_value=60, max_value=1000, 
                                                   value=st.session_state.backtest_days, step=10, key="backtest_days_input")
                with col2:
                    initial_capital = st.number_input("初始资金（元）", min_value=10000, max_value=1000000,
                                                     value=st.session_state.backtest_initial_capital, step=10000, key="initial_capital_input")
                with col3:
                    rsi_buy = st.slider("RSI买入阈值", min_value=10.0, max_value=50.0, value=st.session_state.backtest_rsi_buy, step=1.0)
                with col4:
                    rsi_sell = st.slider("RSI卖出阈值", min_value=50.0, max_value=90.0, value=st.session_state.backtest_rsi_sell, step=1.0)
                
                if st.button("🚀 运行回测", type="primary", key="run_backtest_button"):
                    with st.spinner("正在运行回测，请稍候..."):
                        # 获取选定股票的价格序列
                        price_series = data[backtest_stock]
                        
                        # 如果数据长度超过回测天数，则截取
                        if len(price_series) > backtest_days:
                            price_series = price_series.iloc[-backtest_days:]
                        
                        # 生成信号序列
                        signal_series = generate_signal_series(price_series, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
                        
                        # 运行回测
                        result_df, metrics = run_vectorized_backtest(price_series, signal_series, 
                                                                    initial_capital=initial_capital)
                        
                        # 保存结果到session_state
                        st.session_state.backtest_results[backtest_stock] = (result_df, metrics)
                        st.session_state.backtest_days = backtest_days
                        st.session_state.backtest_initial_capital = initial_capital
                        st.session_state.backtest_rsi_buy = rsi_buy
                        st.session_state.backtest_rsi_sell = rsi_sell
                        
                        st.success("回测完成！")
                
                # 显示回测结果
                if backtest_stock in st.session_state.backtest_results:
                    result_df, metrics = st.session_state.backtest_results[backtest_stock]
                    
                    # 显示关键指标
                    st.subheader("回测关键指标")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总收益率", f"{metrics['total_return']:.2%}")
                    with col2:
                        st.metric("年化收益率", f"{metrics['annualized_return']:.2%}")
                    with col3:
                        st.metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
                    with col4:
                        st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
                    
                    col5, col6, col7, col8 = st.columns(4)
                    with col5:
                        st.metric("胜率", f"{metrics['win_rate']:.2%}")
                    with col6:
                        pl_ratio = metrics['profit_loss_ratio']
                        if pl_ratio == np.inf:
                            pl_ratio_display = "∞"
                        else:
                            pl_ratio_display = f"{pl_ratio:.2f}"
                        st.metric("盈亏比", pl_ratio_display)
                    with col7:
                        st.metric("交易次数", f"{metrics['num_trades']}")
                    with col8:
                        st.metric("最终净值", f"{metrics['final_equity']:,.2f}元")
                    
                # 绘制回测图表
                    # 重新获取价格序列（确保变量在作用域内）
                    if 'backtest_stock' in locals() or 'backtest_stock' in globals():
                        current_price_series = data[backtest_stock]
                        if len(current_price_series) > backtest_days:
                            current_price_series = current_price_series.iloc[-backtest_days:]
                    else:
                        current_price_series = price_series  # 回退到原始变量
                    
                    # 背离检测功能
                    st.subheader("🔍 背离检测（寻找最高质量预警信号）")
                    st.caption("背离是价格与RSI动能之间的失配，被视为趋势反转的最强预警信号之一。")
                    
                    col_div1, col_div2, col_div3 = st.columns(3)
                    with col_div1:
                        enable_divergence = st.checkbox("启用背离检测", value=True, key="enable_divergence_checkbox")
                    with col_div2:
                        divergence_order = st.slider("极值检测窗口", min_value=3, max_value=15, value=5, 
                                                    help="窗口越大越稳健，窗口越小越灵敏")
                    with col_div3:
                        max_days_between = st.slider("最大时间间隔(天)", min_value=30, max_value=120, value=60,
                                                     help="两个极值点之间的最大天数限制")
                    
                    if enable_divergence and not current_price_series.empty:
                        with st.spinner("正在检测背离信号..."):
                            # 准备数据进行背离检测
                            price_data = current_price_series if 'current_price_series' in locals() else price_series
                            df_for_divergence = pd.DataFrame({'close': price_data})
                            
                            # 检测背离
                            divergence_df = detect_rsi_divergence(df_for_divergence, 
                                                                order=divergence_order,
                                                                max_days_between=max_days_between)
                            
                            # 保存结果到session_state
                            st.session_state.divergence_results = divergence_df
                            
                            # 绘制带有背离信号的图表
                            fig = plot_backtest_with_divergence(current_price_series, result_df, metrics, divergence_df)
                            
                            # 显示背离统计
                            if divergence_df is not None:
                                bearish_count = divergence_df['bearish_divergence'].sum()
                                bullish_count = divergence_df['bullish_divergence'].sum()
                                
                                if bearish_count > 0 or bullish_count > 0:
                                    st.success(f"发现 {bearish_count} 个顶背离信号和 {bullish_count} 个底背离信号")
                    else:
                        # 绘制普通回测图表
                        fig = plot_backtest_results(current_price_series, result_df, metrics)
                    
                    st.plotly_chart(fig, config=PLOTLY_CONFIG)
                    
                    # 显示背离信号详情
                    if 'divergence_results' in st.session_state and enable_divergence:
                        divergence_df = st.session_state.divergence_results
                        
                        bearish_signals = divergence_df[divergence_df['bearish_divergence']]
                        bullish_signals = divergence_df[divergence_df['bullish_divergence']]
                        
                        if not bearish_signals.empty or not bullish_signals.empty:
                            st.subheader("📊 背离信号详情")
                            
                            if not bearish_signals.empty:
                                st.markdown("**🔴 顶背离信号（卖出预警）**")
                                bearish_display = bearish_signals[['close', 'divergence_strength']].copy()
                                bearish_display['日期'] = bearish_display.index
                                bearish_display['类型'] = '顶背离'
                                bearish_display = bearish_display[['日期', 'close', 'divergence_strength', '类型']]
                                bearish_display.columns = ['日期', '价格', '强度', '类型']
                                st.dataframe(bearish_display.style.format({
                                    '价格': '{:.2f}',
                                    '强度': '{:.1f}'
                                }))
                            
                            if not bullish_signals.empty:
                                st.markdown("**🟢 底背离信号（买入预警）**")
                                bullish_display = bullish_signals[['close', 'divergence_strength']].copy()
                                bullish_display['日期'] = bullish_display.index
                                bullish_display['类型'] = '底背离'
                                bullish_display = bullish_display[['日期', 'close', 'divergence_strength', '类型']]
                                bullish_display.columns = ['日期', '价格', '强度', '类型']
                                st.dataframe(bullish_display.style.format({
                                    '价格': '{:.2f}',
                                    '强度': '{:.1f}'
                                }))
                            
                            # 背离信号解读
                            st.subheader("🧠 背离信号解读")
                            if not bearish_signals.empty:
                                st.warning("""
                                **顶背离（红色向下箭头）**：
                                - **表现**：价格创出新高，但RSI动能指标未能同步创出新高
                                - **意义**：上涨动能枯竭，多头力量衰减
                                - **建议**：考虑减仓或设置严格止损，防范趋势反转风险
                                """)
                            
                            if not bullish_signals.empty:
                                st.success("""
                                **底背离（绿色向上箭头）**：
                                - **临床表现**：价格创出新低，但RSI动能指标表现出更高的低点
                                - **病理意义**：下跌动能衰减，空头力量耗尽
                                - **临床建议**：考虑分批建仓或增加持仓，捕捉潜在反弹机会
                                """)
                    
                    # 网格搜索功能
                    st.subheader("🔍 网格搜索寻优（寻找最佳参数组合）")
                    st.caption("通过穷举RSI参数组合，寻找在历史数据上表现最优的策略参数。")
                    
                    col_gs1, col_gs2, col_gs3 = st.columns(3)
                    with col_gs1:
                        windows_range = st.slider("RSI周期范围", min_value=6, max_value=30, value=(6, 20), step=2)
                    with col_gs2:
                        buy_range = st.slider("买入阈值范围", min_value=10, max_value=45, value=(25, 40), step=5)
                    with col_gs3:
                        sell_range = st.slider("卖出阈值范围", min_value=55, max_value=90, value=(65, 80), step=5)
                    
                    if st.button("🚀 执行网格搜索", type="secondary", key="run_grid_search_button"):
                        with st.spinner("正在遍历参数组合，请稍候（这可能需要几分钟）..."):
                            # 创建参数范围
                            windows = range(windows_range[0], windows_range[1] + 1, 2)
                            buy_thresholds = range(buy_range[0], buy_range[1] + 1, 5)
                            sell_thresholds = range(sell_range[0], sell_range[1] + 1, 5)
                            
                            # 准备数据
                            price_data = current_price_series if 'current_price_series' in locals() else price_series
                            df_for_grid = pd.DataFrame({'close': price_data})
                            
                            # 运行简单网格搜索（快速）
                            simple_results = rsi_grid_search(
                                df_for_grid,
                                windows=windows,
                                buy_thresholds=buy_thresholds,
                                sell_thresholds=sell_thresholds
                            )
                            
                            # 保存结果到session_state
                            st.session_state.grid_search_results = simple_results
                            
                            st.success(f"网格搜索完成！共测试了 {len(simple_results)} 个参数组合。")
                    
                    # 显示网格搜索结果
                    if 'grid_search_results' in st.session_state and not st.session_state.grid_search_results.empty:
                        gs_results = st.session_state.grid_search_results
                        
                        st.subheader("🏆 最优参数组合 Top 10")
                        display_gs = gs_results.head(10).copy()
                        display_gs['排名'] = range(1, len(display_gs) + 1)
                        display_gs = display_gs[['排名', 'window', 'buy_threshold', 'sell_threshold', 
                                                'total_return', 'sharpe', 'num_trades']]
                        
                        # 格式化显示
                        st.dataframe(
                            display_gs.style.format({
                                'total_return': '{:.2%}',
                                'sharpe': '{:.2f}'
                            })
                        )
                        
                        # 显示最优参数
                        best_row = gs_results.iloc[0]
                        st.success(
                            f"**最优参数建议**: RSI周期={best_row['window']}天, "
                            f"买入阈值={best_row['buy_threshold']}, "
                            f"卖出阈值={best_row['sell_threshold']} "
                            f"(总收益: {best_row['total_return']:.2%}, 夏普比率: {best_row['sharpe']:.2f})"
                        )
                        
                        # 自动应用最优参数按钮
                        if st.button("💾 应用最优参数到回测", type="primary", key="apply_best_params"):
                            st.session_state.backtest_rsi_buy = best_row['buy_threshold']
                            st.session_state.backtest_rsi_sell = best_row['sell_threshold']
                            st.success(f"已更新RSI参数：买入={best_row['buy_threshold']}, 卖出={best_row['sell_threshold']}")
                            st.rerun()
            
            # 背离信号实时监测 (独立图表)
            st.subheader("🚦 背离信号实时监测")
            st.caption("专为科研工作者设计的动能背离深度分析，识别价格与RSI动能的失配，捕捉趋势反转预警信号。")
            
            # 选择分析标的
            target_stock = st.selectbox("选择分析标的", options=list(analysis_stocks.values()), key="divergence_target_select")
            
            if target_stock and target_stock in data.columns:
                # 获取该个股的数据
                single_df = pd.DataFrame({'close': data[target_stock]})
                
                # 计算RSI
                single_df['rsi'] = calculate_rsi(single_df['close'])
                
                # 检测背离
                with st.spinner(f"正在分析 {target_stock} 的背离信号..."):
                    divergence_df = detect_rsi_divergence(single_df, order=5, max_days_between=60)
                    
                    # 绘图
                    div_fig = plot_divergence_chart(divergence_df, target_stock)
                    st.plotly_chart(div_fig, config=PLOTLY_CONFIG)
                    
                    # 给出具体的科研判定建议
                    # 检查最近5天是否有信号
                    recent_days = 5
                    if len(divergence_df) >= recent_days:
                        latest_bull = divergence_df['bullish_divergence'].iloc[-recent_days:].any()
                        latest_bear = divergence_df['bearish_divergence'].iloc[-recent_days:].any()
                        
                        if latest_bull:
                            st.success(f"🔍 监测到 **{target_stock}** 近期出现【底背离】，提示下跌动能衰减，可考虑逐步建立头寸。")
                        elif latest_bear:
                            st.error(f"⚠️ 监测到 **{target_stock}** 近期出现【顶背离】，提示上涨动能枯竭，请务必注意风险，考虑减仓。")
                        else:
                            st.info(f"📊 **{target_stock}** 近期未检测到明显背离信号，当前趋势动能与价格方向一致。")
                    
            # 显示最近的背离信号详情
                    bearish_signals = divergence_df[divergence_df['bearish_divergence']]
                    bullish_signals = divergence_df[divergence_df['bullish_divergence']]
                    
                    if not bearish_signals.empty or not bullish_signals.empty:
                        st.subheader("📋 历史背离信号记录")
                        
                        if not bearish_signals.empty:
                            st.markdown("**🔴 顶背离信号记录**")
                            bearish_display = bearish_signals[['close', 'divergence_strength']].copy()
                            bearish_display['日期'] = bearish_display.index
                            bearish_display['类型'] = '顶背离'
                            bearish_display = bearish_display[['日期', 'close', 'divergence_strength', '类型']]
                            bearish_display.columns = ['日期', '价格', '强度', '类型']
                            st.dataframe(bearish_display.tail(5).style.format({
                                '价格': '{:.2f}',
                                '强度': '{:.1f}'
                            }))
                        
                        if not bullish_signals.empty:
                            st.markdown("**🟢 底背离信号记录**")
                            bullish_display = bullish_signals[['close', 'divergence_strength']].copy()
                            bullish_display['日期'] = bullish_display.index
                            bullish_display['类型'] = '底背离'
                            bullish_display = bullish_display[['日期', 'close', 'divergence_strength', '类型']]
                            bullish_display.columns = ['日期', '价格', '强度', '类型']
                            st.dataframe(bullish_display.tail(5).style.format({
                                '价格': '{:.2f}',
                                '强度': '{:.1f}'
                            }))
            
            # 如果target_stock不在data中，显示警告
            elif not target_stock or target_stock not in data.columns:
                st.warning("请选择分析标的并确保数据可用。")
            else:
                st.warning("没有可用的股票数据进行回测。")

        with t3:
            st.subheader("资产相关性矩阵 (防范共振风险)")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(returns.corr(), annot=True, cmap='RdYlGn', center=0, ax=ax_corr)
            st.pyplot(fig_corr)

        with t4:
            col_l, col_r = st.columns(2)
            with col_l:
                st.metric("预期年化收益", f"{best_p['Ret']:.2%}")
                st.caption("📈 **预期年化收益**：基于历史数据计算的该投资组合在未来一年的预期收益率，已考虑复利效应。")
                fig_pie, ax_pie = plt.subplots()
                best_p[list(analysis_stocks.values())].plot.pie(autopct='%1.1f%%', ax=ax_pie, cmap='Pastel1', title="AI 推荐配比")
                st.pyplot(fig_pie)
                st.caption("🥧 **AI推荐配比**：通过蒙特卡洛模拟找到的夏普比率最高的投资组合权重分配。")
            with col_r:
                st.metric("夏普比率 (性价比)", f"{best_p['Sharpe']:.2f}")
                st.caption("⚖️ **夏普比率**：衡量每承担一单位风险所获得的超额收益，数值越高表示风险调整后收益越好。")
                
                # 显示前10个优选组合
                with st.expander("📊 查看前10个优选组合", expanded=False):
                    st.caption("""
                    **指标解释**：
                    - **排名**：按夏普比率从高到低排序
                    - **Ret**：预期年化收益率
                    - **Vol**：预期年化波动率（风险）
                    - **Sharpe**：夏普比率（风险调整后收益）
                    - **各股票列**：在该组合中的权重分配
                    """)
                    # 按夏普比率降序排序，取前10
                    top10 = sim_res.sort_values('Sharpe', ascending=False).head(10).copy()
                    # 重置索引并添加排名
                    top10 = top10.reset_index(drop=True)
                    top10.insert(0, '排名', range(1, len(top10) + 1))
                    
                    # 格式化显示
                    display_df = top10[['排名', 'Ret', 'Vol', 'Sharpe'] + list(analysis_stocks.values())].copy()
                    # 将收益和波动率转换为百分比字符串
                    display_df['Ret'] = display_df['Ret'].apply(lambda x: f"{x:.2%}")
                    display_df['Vol'] = display_df['Vol'].apply(lambda x: f"{x:.2%}")
                    display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.2f}")
                    # 将各股票权重转换为百分比字符串
                    for name in analysis_stocks.values():
                        display_df[name] = display_df[name].apply(lambda x: f"{x:.1%}")
                    
                    st.dataframe(display_df, width='stretch')
                    st.caption(f"共模拟 {sim_num} 次，展示了夏普比率最高的10个组合")
                
                fig_ef, ax_ef = plt.subplots()
                ax_ef.scatter(sim_res.Vol, sim_res.Ret, c=sim_res.Sharpe, cmap='viridis', alpha=0.3)
                ax_ef.scatter(best_p['Vol'], best_p['Ret'], color='red', marker='*', s=200)
                st.pyplot(fig_ef)
                st.caption("📊 **有效前沿图**：每个点代表一个随机投资组合，红色星号标记夏普比率最高的组合。")

        with t5:
            st.subheader("📈 模拟交易 - 持仓分析与调仓建议")
            st.write("请在下方输入您当前的持仓数量（股），系统将根据最优权重给出调仓建议。")
            
            # 初始化持仓输入
            holdings = {}
            col1, col2, col3 = st.columns(3)
            selected_names = list(analysis_stocks.values())
            for idx, name in enumerate(selected_names):
                # 每列放置几个输入框
                with [col1, col2, col3][idx % 3]:
                    holdings[name] = st.number_input(
                        f"{name} 持仓数量",
                        min_value=0,
                        value=0,
                        step=100,
                        key=f"hold_{name}"
                    )
            
            if st.button("分析持仓并生成调仓建议", type="primary"):
                if sum(holdings.values()) == 0:
                    st.warning("您还没有输入任何持仓，请输入持仓数量。")
                else:
                    # 调用分析函数
                    best_weights_dict = best_p[selected_names].to_dict()
                    analysis_df, total_value = analyze_portfolio(holdings, data, best_weights_dict, stop_loss_val)
                    
                    st.metric("持仓总市值", f"{total_value:,.2f} 元")
                    
                    st.subheader("持仓详情与调仓建议")
                    st.dataframe(analysis_df)
                    
                    # 可视化当前权重与建议权重对比
                    st.subheader("权重对比")
                    compare_df = pd.DataFrame({
                        '标的': analysis_df['标的'],
                        '当前权重': analysis_df['当前权重'].str.rstrip('%').astype('float') / 100,
                        '建议权重': analysis_df['建议权重'].str.rstrip('%').astype('float') / 100
                    })
                    compare_df = compare_df.set_index('标的')
                    st.bar_chart(compare_df * 100)  # 转换为百分比显示
                    
                    # 总结调仓动作
                    st.subheader("📋 调仓操作总结")
                    for _, row in analysis_df.iterrows():
                        if row['调仓建议'] != '保持':
                            st.info(f"{row['标的']}: {row['调仓建议']}")

        with t6:
            st.subheader("📈 实盘持仓监测")
            st.caption("基于真实交易记录计算盈亏情况")
            
            # 1. 刷新控制区 (放在 fragment 外，确保控制组件本身不被局部刷新频率干扰)
            col_refresh1, col_refresh2 = st.columns([1, 3])
            with col_refresh1:
                auto_refresh = st.checkbox("⏱️ 启用自动刷新", value=False, key="auto_refresh_t6")
                # 如果开启自动刷新，设置频率（秒），否则为 None
                refresh_freq = 600 if auto_refresh else None
            
            # 2. 定义局部刷新片段
            # run_every 会在不重绘整个页面的情况下，只触发该函数内部逻辑
            @st.fragment(run_every=refresh_freq)
            def render_portfolio_fragment_v2():
                # 注意：manual_btn 放在内部，点击它只会触发本片段刷新
                btn_col1, btn_col2 = st.columns([1, 4])
                with btn_col1:
                    refresh_now = st.button("🔄 立即刷新", type="primary", key="inner_refresh_t6")
                
                # 处理缓存清理：修复之前的 TypeError
                if refresh_now:
                    # 正确做法：直接调用缓存函数的 .clear() 方法
                    fetch_latest_prices.clear()
                    st.toast("已清除缓存，正在获取最新行情...", icon="🔄")

                # 加载持仓 (每次片段执行都会重新 load 确保准确)
                current_config = load_config()
                current_portfolio = current_config.get('portfolio', [])
                
                if not current_portfolio:
                    st.info("暂无真实持仓记录。请在侧边栏录入交易。")
                    return

                # 准备数据
                real_stocks = [trade['name'] for trade in current_portfolio]
                real_codes = [trade['code'] for trade in current_portfolio]

                # 获取价格 (使用 spinner 提示局部加载状态)
                with st.spinner("同步实时行情..."):
                    latest_price_df = fetch_latest_prices(real_codes, real_stocks)
                
                # 备用逻辑
                if latest_price_df.empty or latest_price_df.isna().all().all():
                    latest_price_df = data[real_stocks].iloc[[-1]]

                # 计算盈亏
                summary_df, total_market_value, total_profit_ratio = get_portfolio_summary(current_portfolio, latest_price_df)
                
                if not summary_df.empty:
                    # 显示指标 (Metric)
                    total_profit_amount = summary_df['盈亏额'].sum()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("持仓总成本", f"{summary_df['成本'].sum():,.2f}")
                    m2.metric("持仓总市值", f"{total_market_value:,.2f}")
                    # A股习惯：正数红，负数绿
                    m3.metric("总盈亏比例", f"{total_profit_ratio:.2%}", delta=f"{total_profit_ratio:.2%}")
                    m4.metric("盈亏总额", f"{total_profit_amount:+,.2f}", delta=f"{total_profit_amount:+,.2f}")

                    # --- 表格展示 ---
                    # 定义条件格式化函数
                    def color_profit(val):
                        if isinstance(val, (int, float)):
                            return 'color: #ff4444' if val > 0 else 'color: #00cc00'
                        if isinstance(val, str) and '+' in val: return 'color: #ff4444'
                        if isinstance(val, str) and '-' in val: return 'color: #00cc00'
                        return ''

                    st.subheader("📊 持仓盈亏明细")
                    display_df = summary_df.copy()
                    
                    # 格式化
                    for col in ['成本价', '现价', '成本', '市值']:
                        display_df[col] = display_df[col].map('{:,.2f}'.format)
                    display_df['盈亏比'] = display_df['盈亏比'].map('{:+.2%}'.format)
                    display_df['盈亏额'] = display_df['盈亏额'].map('{:+,.2f}'.format)

                    # 【核心优化】使用固定高度 height=400 彻底锁死布局
                    st.dataframe(
                        display_df.style.map(color_profit, subset=['盈亏额', '盈亏比']),
                        width='stretch',
                        height=400 
                    )

                    # --- 偏离度分析 ---
                    st.subheader("⚖️ 配置偏离度分析")
                    current_weights = {row['资产名称']: row['市值']/total_market_value for _, row in summary_df.iterrows()}
                    optimal_weights = best_p[list(analysis_stocks.values())].to_dict()
                    total_dev, deviations = calculate_deviation(current_weights, optimal_weights)
                    
                    st.write(f"当前配置与 AI 模型最优前沿的**总偏离度**: `{total_dev:.2%}`")
                    
                    dev_df = pd.DataFrame([{
                        "资产名称": n,
                        "当前权重": current_weights.get(n, 0),
                        "最优权重": optimal_weights.get(n, 0),
                        "建议调仓": deviations.get(n, 0)
                    } for n in set(list(current_weights.keys()) + list(optimal_weights.keys()))])

                    st.dataframe(
                        dev_df.style.format({"当前权重": "{:.1%}", "最优权重": "{:.1%}", "建议调仓": "{:+.1%}"}),
                        width='stretch',
                        height=250 # 固定高度
                    )
                    
                    st.caption(f"🕒 片段最后更新时间: {datetime.now().strftime('%H:%M:%S')}")
                else:
                    st.warning("行情计算返回空数据，请检查网络或标的代码。")

            # 3. 执行局部刷新片段
            render_portfolio_fragment_v2()

        with t7:
            st.subheader("📈 独立 K 线分析")
            st.caption("独立技术分析模块，手动选择标的并启动K线分析，不影响主程序状态。")
            
            def render_independent_kline(stock_pool):
                st.subheader("🔍 个股独立技术分析 (手动模式)")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_name = st.selectbox("选择分析标的", options=list(stock_pool.values()), key="ind_kline_select")
                    code = [c for c, n in stock_pool.items() if n == selected_name][0]
                with col2:
                    run_kline = st.checkbox("🚩 启动独立分析引擎", key="kline_active_flag")
                
                if run_kline:
                    with st.spinner(f"正在调取 {selected_name} 全量历史数据..."):
                        from data_fetcher import fetch_candle_data
                        df = fetch_candle_data(code)
                        
                        if df.empty:
                            st.error(f"无法获取 {selected_name} 的K线数据，请稍后重试。")
                        else:
                            import plotly.graph_objects as go
                            from plotly.subplots import make_subplots
                            
                            # 创建基础K线图
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                               vertical_spacing=0.03, row_width=[0.2, 0.8])
                            
                            fig.add_trace(go.Candlestick(
                                x=df['日期'], open=df['开盘'], high=df['最高'], low=df['最低'], close=df['收盘'],
                                name='Price', increasing_line_color='#ff4444', decreasing_line_color='#00cc00'
                            ), row=1, col=1)
                            
                            ma60 = df['收盘'].rolling(window=60).mean()
                            fig.add_trace(go.Scatter(x=df['日期'], y=ma60, line=dict(color='orange', width=1.5), name='MA60'), row=1, col=1)
                            
                            fig.add_trace(go.Bar(x=df['日期'], y=df['成交量'], marker_color='gray', opacity=0.5, name='Volume'), row=2, col=1)
                            
                            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode='x unified')
                            
                            # 添加交互式框选提示和拟合模型选择
                            st.info("📊 交互式趋势拟合分析：使用图表工具栏的框选工具（Box Select）选择一段区间，系统将自动绘制拟合线。")
                            
                            # 拟合模型选择器
                            fit_degree = st.radio("拟合模型选择", [1, 2], 
                                                 format_func=lambda x: "线性趋势 (速度)" if x==1 else "二次曲线 (动能加速度)", 
                                                 horizontal=True, key="fit_degree_selector")
                            
                            # 渲染图表并捕获选择事件
                            event = st.plotly_chart(fig, on_select="rerun", selection_mode=["points", "box", "lasso"], key="kline_chart", config=PLOTLY_CONFIG)

                            # 如果用户进行了框选（使用属性访问方式兼容新版 Streamlit）
                            if event and hasattr(event, 'selection') and event.selection and len(event.selection.points) > 0:
                                # 获取选区索引范围
                                selected_points = event.selection.points
                                idx_start = selected_points[0]["point_index"]
                                idx_end = selected_points[-1]["point_index"]
                                df_slice = df.iloc[idx_start:idx_end+1]
                                
                                # 计算峰谷拟合 (传入 degree 参数)
                                peak_line, p_coeff, p_count = fit_trend_line(df_slice, 'peak', degree=fit_degree)
                                valley_line, v_coeff, v_count = fit_trend_line(df_slice, 'valley', degree=fit_degree)
                                
                                # 定义图例标签前缀和系数名称
                                prefix = "曲率a" if fit_degree == 2 else "斜率k"
                                coeff_name = "曲率" if fit_degree == 2 else "斜率"
                                
                                # 将拟合线叠加到原图
                                if peak_line is not None:
                                    fig.add_trace(go.Scatter(
                                        x=df_slice['日期'], y=peak_line, 
                                        name=f"阻力曲线 ({prefix}:{p_coeff:.4f}, 点数:{p_count})",
                                        line=dict(color='red', dash='dash', width=2)
                                    ), row=1, col=1)
                                if valley_line is not None:
                                    fig.add_trace(go.Scatter(
                                        x=df_slice['日期'], y=valley_line, 
                                        name=f"支撑曲线 ({prefix}:{v_coeff:.4f}, 点数:{v_count})",
                                        line=dict(color='green', dash='dash', width=2)
                                    ), row=1, col=1)
                                
                                # 重新渲染图表
                                st.plotly_chart(fig, config=PLOTLY_CONFIG)
                                
                                # 显示拟合结果分析
                                st.subheader("📈 拟合结果分析")
                                if peak_line is not None:
                                    if fit_degree == 2:
                                        # 二次曲线模式：曲率分析
                                        if p_coeff > 0:
                                            st.success(f"阻力线曲率为正 ({p_coeff:.4f})，凹面向上(∪形)，表明向下的动能在衰减，或者向上的动能在增强。")
                                        elif p_coeff < 0:
                                            st.warning(f"阻力线曲率为负 ({p_coeff:.4f})，凹面向下(∩形)，表明向上的动能在衰减(见顶信号)，或者向下的动能在增强。")
                                        else:
                                            st.info(f"阻力线曲率接近零 ({p_coeff:.4f})，接近直线趋势，动能稳定。")
                                    else:
                                        # 线性模式：斜率分析
                                        if p_coeff > 0:
                                            st.success(f"阻力线斜率为正 ({p_coeff:.2f})，表明在选区内压力位呈上升趋势。")
                                        else:
                                            st.warning(f"阻力线斜率为负 ({p_coeff:.2f})，表明在选区内压力位呈下降趋势。")
                                
                                if valley_line is not None:
                                    if fit_degree == 2:
                                        # 二次曲线模式：曲率分析
                                        if v_coeff > 0:
                                            st.success(f"支撑线曲率为正 ({v_coeff:.4f})，凹面向上(∪形)，表明向下的动能在衰减，或者向上的动能在增强。")
                                        elif v_coeff < 0:
                                            st.warning(f"支撑线曲率为负 ({v_coeff:.4f})，凹面向下(∩形)，表明向上的动能在衰减(见顶信号)，或者向下的动能在增强。")
                                        else:
                                            st.info(f"支撑线曲率接近零 ({v_coeff:.4f})，接近直线趋势，动能稳定。")
                                    else:
                                        # 线性模式：斜率分析
                                        if v_coeff > 0:
                                            st.success(f"支撑线斜率为正 ({v_coeff:.2f})，表明在选区内支撑位呈上升趋势。")
                                        else:
                                            st.warning(f"支撑线斜率为负 ({v_coeff:.2f})，表明在选区内支撑位呈下降趋势。")
                                
                                # 平行通道判定
                                coeff_diff = 0.0
                                if peak_line is not None and valley_line is not None:
                                    coeff_diff = abs(p_coeff - v_coeff)
                                    if fit_degree == 2:
                                        # 二次曲线：曲率相近判定
                                        if coeff_diff < 0.01:  # 曲率相近
                                            st.info("阻力线与支撑线曲率相近，形成平行弯曲通道，适合进行高抛低吸策略。")
                                        else:
                                            st.info("阻力线与支撑线曲率差异明显，表明弯曲程度不同，动能变化不对称。")
                                    else:
                                        # 线性：斜率相近判定
                                        if coeff_diff < 0.1:  # 斜率相近
                                            st.info("阻力线与支撑线斜率相近，形成平行通道，适合进行高抛低吸策略。")
                                        else:
                                            st.info("阻力线与支撑线斜率差异明显，表明趋势通道正在扩大或收缩。")
                                
                                # 突破判定
                                latest_close = None
                                latest_peak = None
                                if not df_slice.empty and peak_line is not None:
                                    latest_close = df_slice['收盘'].iloc[-1]
                                    latest_peak = peak_line[-1]
                                    if latest_close > latest_peak:
                                        st.success(f"最新收盘价 {latest_close:.2f} 已突破阻力线 {latest_peak:.2f}，可能形成突破信号。")
                                    
                                # 二次曲线模式下的额外分析：趋势加速/减速
                                if fit_degree == 2 and peak_line is not None:
                                    if p_coeff > 0:
                                        st.info("阻力线曲率为正，凹面向上，突破后上涨动能可能加速。")
                                    elif p_coeff < 0:
                                        st.warning("阻力线曲率为负，凹面向下，突破后上涨动能可能减速。")
                                
                                # AI深度分析
                                st.subheader("🤖 AI深度分析")
                                if ds_key:
                                    if st.button("🔍 启动AI深度分析", type="secondary", key="ai_fit_analysis"):
                                        with st.spinner("AI正在分析拟合结果，请稍候..."):
                                            try:
                                                from openai import OpenAI
                                                client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
                                                
                                                # 构建分析提示词 - 修复格式化字符串错误
                                                model_type = "二次曲线（加速度）" if fit_degree == 2 else "线性（速度）"
                                                coeff_unit = "曲率a" if fit_degree == 2 else "斜率k"
                                                
                                                # 格式化系数显示
                                                p_coeff_formatted = f"{p_coeff:.4f}" if fit_degree == 2 else f"{p_coeff:.2f}"
                                                v_coeff_formatted = f"{v_coeff:.4f}" if fit_degree == 2 else f"{v_coeff:.2f}"
                                                coeff_diff_formatted = f"{coeff_diff:.4f}" if fit_degree == 2 else f"{coeff_diff:.2f}"
                                                
                                                # 突破信号
                                                if not df_slice.empty and peak_line is not None:
                                                    breakthrough_status = "已突破阻力线" if latest_close > latest_peak else "未突破阻力线"
                                                    breakthrough_text = f"最新收盘价{breakthrough_status}"
                                                else:
                                                    breakthrough_text = "未检测到突破信号"
                                                
                                                analysis_prompt = f"""作为资深金融分析师，请对以下股票技术分析拟合结果进行深度解读：
                                                
标的股票：{selected_name}
拟合模型：{model_type}
阻力线核心系数：{p_coeff_formatted} ({coeff_unit})
支撑线核心系数：{v_coeff_formatted} ({coeff_unit})
系数差异：{coeff_diff_formatted}
{breakthrough_text}
选区数据范围：{len(df_slice)}个交易日

请基于以上技术指标，提供：
1. 技术形态解读（50字内）
2. 多空动能评估（30字内）
3. 具体操作建议（20字内）
4. 风险提示（20字内）"""
                                                
                                                response = client.chat.completions.create(
                                                    model="deepseek-chat",
                                                    messages=[
                                                        {"role": "system", "content": "你是资深金融分析师，擅长技术分析，用简洁专业的中文回答。"},
                                                        {"role": "user", "content": analysis_prompt}
                                                    ],
                                                    temperature=0.3,
                                                    max_tokens=500
                                                )
                                                
                                                ai_analysis = response.choices[0].message.content
                                                
                                                # 显示分析结果
                                                st.success("AI分析完成！")
                                                st.markdown(f"**🤖 AI分析结果：**")
                                                st.info(ai_analysis)
                                                
                                            except Exception as e:
                                                st.error(f"AI分析失败：{e}")
                                else:
                                    st.warning("⚠️ 如需AI深度分析，请在侧边栏配置DeepSeek API密钥。")

                            st.caption(f"提示：当前正在对 {selected_name} 进行技术面独立审计。")
            
            if analysis_stocks:
                stock_pool = analysis_stocks
            else:
                first_cat = list(st.session_state.master_pool.keys())[0]
                stock_pool = st.session_state.master_pool[first_cat]
            
            render_independent_kline(stock_pool)

    else:
        st.error("网络连接异常，无法获取行情。")
else:
    st.warning("👈 请在左侧勾选标的。")
