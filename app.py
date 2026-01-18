"""
FILE: app.py
ROLE: Streamlit 应用入口，提供 Web 界面，展示数据分析结果、图表和交互控件。
LOGIC: 
    1. 加载配置和用户设置，初始化标的库和持仓数据。
    2. 提供侧边栏交互控件，包括标的库管理、交易录入、风险参数设置。
    3. 实现数据缓存机制，使用 session_state 避免重复 I/O，提升性能。
    4. 集成多页面导航，提供量化决策、风险管理、技术分析等功能。
DEPENDENCIES: streamlit, pandas, numpy, matplotlib, seaborn, plotly, config, data_fetcher, services.market_data_service, ai_analyzer, backtester, math_engine, font_utils
"""

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
from data_fetcher import get_signals, compute_portfolio_stats, monte_carlo_simulation, analyze_portfolio, calculate_deviation, fetch_stock_data
from services.market_data_service import safe_fetch_latest_prices, safe_search_stock_info, safe_fetch_candle_data, get_portfolio_summary, get_recent_10_days
from ai_analyzer import deepseek_analyze
from backtester import generate_signal_series, run_vectorized_backtest, plot_backtest_results, plot_backtest_with_divergence, rsi_grid_search, grid_search_with_backtest, detect_rsi_divergence, plot_divergence_chart, calculate_rsi
from math_engine import fit_trend_line, detect_peaks_valleys
from font_utils import setup_chinese_font

# --- 环境与中文字体配置 ---
# 自动配置中文字体，优先检测Linux系统路径下的中文字体（如Noto Sans CJK）
font_name, font_path = setup_chinese_font()
if font_name:
    print(f"已使用中文字体: {font_name}")
else:
    print("使用默认字体配置")
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="主控台", page_icon="🏠", layout="wide")

# --- 加载配置 ---
config = load_config()

# --- 初始化session_state缓存 ---
# 使用session_state缓存历史数据，避免重复加载
# 为什么要使用session_state：
# 1. Streamlit每次交互都会rerun整个脚本
# 2. 如果不缓存，每次rerun都会重新读取数据（即使数据没有变化）
# 3. session_state可以跨rerun保持数据，避免重复I/O
# 4. 结合Parquet本地存储，实现"内存缓存+磁盘缓存"双层加速
if 'hist_data_cache' not in st.session_state:
    st.session_state.hist_data_cache = {}
    
if 'current_stocks_key' not in st.session_state:
    st.session_state.current_stocks_key = None

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
ds_key = st.sidebar.text_input("DeepSeek API 密钥", value=config.get("deepseek_api_key", ""), type="password",
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
        search_results = safe_search_stock_info(search_query.strip())
    
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

# 增量更新按钮
if st.sidebar.button("🔄 刷新历史数据（增量更新）", type="secondary", width='stretch'):
    # 获取当前选中的股票
    selected_codes = []
    selected_names = []
    for cat, stocks in st.session_state.master_pool.items():
        for c, n in stocks.items():
            if st.session_state.get(f"sel_{c}", True):
                selected_codes.append(c)
                selected_names.append(n)

    if not selected_codes:
        st.sidebar.warning("请先勾选要更新的股票！")
    else:
        # ✅ 立即执行增量更新（使用并行优化）
        from market_store import get_market_store
        store = get_market_store()

        with st.spinner(f"正在增量更新 {len(selected_codes)} 只股票的历史数据..."):
            # 使用并行批量更新
            results = store.batch_update_recent_data(
                codes=selected_codes,
                names=selected_names,
                recent_days=30,
                max_workers=5
            )

        # 统计结果
        success_count = sum(1 for v in results.values() if v)
        failed_count = len(results) - success_count

        # ✅ 全面清除所有数据缓存，确保所有页面使用更新后的数据
        keys_to_delete = []
        for key in list(st.session_state.keys()):
            if key.startswith('_latest_prices_cache'):
                keys_to_delete.append(key)
            elif key.startswith('analysis_data_'):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del st.session_state[key]

        # 清除主控台缓存
        st.session_state.hist_data_cache = {}
        st.session_state.current_stocks_key = None

        # 清除页面共享的数据（确保其他页面重新加载）
        for key in ['analysis_data', 'returns', 'mean_ret', 'cov_mat', 'sim_res',
                    'best_p', 'latest_prices', 'signals', 'stock_codes', 'stock_names']:
            if key in st.session_state:
                del st.session_state[key]

        # 显示结果
        if failed_count == 0:
            st.sidebar.success(f"✅ 增量更新完成！成功更新 {success_count} 只股票，缓存已清除。")
        else:
            st.sidebar.warning(f"⚠️ 增量更新完成：成功 {success_count} 只，失败 {failed_count} 只。缓存已清除。")

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
    # 生成缓存键：基于股票代码和名称的组合
    codes = list(analysis_stocks.keys())
    names = list(analysis_stocks.values())
    cache_key = f"{sorted(codes)}:{sorted(names)}"
    
    # 检查缓存是否有效：如果缓存键变化，则重新获取数据
    # 注意：增量更新已在"刷新历史数据"按钮处理时立即执行，这里只需从本地读取
    if (st.session_state.current_stocks_key != cache_key or
        cache_key not in st.session_state.hist_data_cache):

        # 使用fetch_stock_data函数获取数据（从本地Parquet读取，不触发增量更新）
        with st.spinner("正在加载历史行情数据..."):
            data = fetch_stock_data(codes, names, incremental_update=False)
        
        # 缓存数据
        if not data.empty:
            st.session_state.hist_data_cache[cache_key] = data
            st.session_state.current_stocks_key = cache_key
        else:
            # 如果数据为空，尝试使用兜底函数（这里我们直接使用缓存中的旧数据，如果有的话）
            if cache_key in st.session_state.hist_data_cache:
                st.warning("获取最新数据失败，使用缓存的历史数据。")
                data = st.session_state.hist_data_cache[cache_key]
            else:
                st.error("无法获取股票数据，请检查网络连接或股票代码。")
                st.stop()
    else:
        # 使用缓存的数据
        data = st.session_state.hist_data_cache[cache_key]
        st.success(f"使用缓存的历史数据（共 {len(data)} 个交易日）")
    
    if not data.empty:
        # 检查实际获取到的股票数据，更新analysis_stocks
        actual_stocks = {}
        for code, name in analysis_stocks.items():
            if name in data.columns:
                actual_stocks[code] = name
            else:
                st.warning(f"股票 {name}({code}) 数据获取失败，已从分析中排除。可能是网络问题或该股票代码暂时无法获取数据，请检查网络连接或稍后重试。")
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

        # 获取最新价格和交易信号（供子页面使用）
        stock_codes = list(analysis_stocks.keys())
        stock_names = list(analysis_stocks.values())
        latest_prices = safe_fetch_latest_prices(stock_codes, stock_names)
        signals = get_signals(data, best_p.to_dict(), stop_loss_val, latest_prices)

        # ✅ 保存完整分析结果到session_state，供其他页面使用（数据接力模式）
        st.session_state.analysis_data = data
        st.session_state.returns = returns
        st.session_state.mean_ret = mean_ret
        st.session_state.cov_mat = cov_mat
        st.session_state.sim_res = sim_res
        st.session_state.best_p = best_p
        st.session_state.latest_prices = latest_prices
        st.session_state.signals = signals
        st.session_state.stock_codes = stock_codes
        st.session_state.stock_names = stock_names

        # 功能标签页已迁移到独立页面
        # 请在侧边栏选择页面进行访问
        st.subheader("📊 分析结果概要")
        st.success(f"已成功分析 {len(analysis_stocks)} 个标的")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("预期年化收益", f"{best_p['Ret']:.2%}")
        with col2:
            st.metric("夏普比率", f"{best_p['Sharpe']:.2f}")
        with col3:
            st.metric("最优组合波动率", f"{best_p['Vol']:.2%}")
        
        st.subheader("🌐 功能页面导航")
        st.info("""
        本应用已升级为多页面架构，原有标签页现在作为独立页面提供更佳体验。
        请在左侧侧边栏选择对应的页面访问各个功能模块：

        1. **AI资讯深度研判** - DeepSeek专业资讯评分
        2. **买卖与风险预警** - 量化决策与风险管理
        3. **板块相关性分析** - 资产相关性矩阵
        4. **权重优化实验** - 投资组合优化
        5. **模拟交易** - 持仓分析与调仓建议
        6. **实盘持仓监测** - 真实交易盈亏计算
        7. **独立K线分析** - 技术面独立审计
        8. **优先级雷达** - 多因子筛选与综合评分
        9. **宏观审计** - EPU指数与政策分析
        10. **数据完整性检查** - 历史数据缺失检测与修复

        或者直接点击下面的快速链接：
        """)
        
        # 显示快速链接 - 使用st.switch_page实现真正的页面跳转
        st.subheader("🚀 快速导航")
        st.caption("点击下方按钮直接跳转到对应页面，无需通过侧边栏")
        
        # 页面映射
        pages_map = [
            {"name": "AI资讯深度研判", "icon": "💡", "path": "pages/01_ai_news.py"},
            {"name": "买卖与风险预警", "icon": "🚦", "path": "pages/02_signals_risk.py"},
            {"name": "板块相关性分析", "icon": "🕸️", "path": "pages/03_corr.py"},
            {"name": "权重优化实验", "icon": "📊", "path": "pages/04_portfolio_opt.py"},
            {"name": "模拟交易", "icon": "📈", "path": "pages/05_sim_trade.py"},
            {"name": "实盘持仓监测", "icon": "💼", "path": "pages/06_live_portfolio.py"},
            {"name": "独立K线分析", "icon": "📉", "path": "pages/07_kline_lab.py"},
            {"name": "优先级雷达", "icon": "🎯", "path": "pages/08_priority_radar.py"},
            {"name": "宏观审计", "icon": "🌍", "path": "pages/09_macro_audit.py"},
            {"name": "数据完整性检查", "icon": "🔍", "path": "pages/10_data_integrity.py"}
        ]
        
        # 创建两行，每行5个按钮
        cols1 = st.columns(5)
        cols2 = st.columns(5)

        for i, page in enumerate(pages_map):
            if i < 5:
                col = cols1[i]
            else:
                col = cols2[i-5]
            
            with col:
                if st.button(f"{page['icon']} {page['name']}",
                           key=f"nav_{i}",
                           use_container_width=True,
                           help=f"跳转到{page['name']}页面"):
                    # 数据已在上方保存到session_state，直接切换页面
                    st.switch_page(page['path'])
        
        st.caption("提示：点击按钮可直接跳转到对应页面，数据已从主控台传递。")

    else:
        st.error("网络连接异常，无法获取行情。")
else:
    st.warning("👈 请在左侧勾选标的。")
