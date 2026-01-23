import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# 导入自定义模块
from config import load_config, update_trade, find_trade_index, save_config
from services.data_service import (
    initialize_page_data,
    DataServiceError
)
from data_fetcher import calculate_deviation
from services.market_data_service import get_portfolio_summary, safe_fetch_latest_prices
from ai_analyzer import deepseek_analyze
from backtester import generate_signal_series, run_vectorized_backtest, plot_backtest_results, plot_backtest_with_divergence, rsi_grid_search, grid_search_with_backtest, detect_rsi_divergence, plot_divergence_chart, calculate_rsi
from math_engine import fit_trend_line, detect_peaks_valleys
from font_utils import setup_chinese_font

# Plotly 图表配置 (符合 Streamlit 2026 标准)
PLOTLY_CONFIG = {
    'displayModeBar': True,    # 显示工具栏
    'displaylogo': False,      # 隐藏 Plotly Logo
    'modeBarButtonsToRemove': ['lasso2d'],  # 保留select2d框选工具
    'responsive': True         # 自适应容器
}

# 设置中文字体
font_name, font_path = setup_chinese_font()
if font_name:
    print(f"已使用中文字体: {font_name}")
else:
    print("使用默认字体配置")
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(page_title="实盘持仓监测", layout="wide")

# 检查是否有选中的标的
if 'selected_stocks' not in st.session_state or not st.session_state.selected_stocks:
    st.warning("请先返回主页面选择标的并开始分析。")
    st.stop()

# 获取必要的变量
analysis_stocks = st.session_state.selected_stocks
config = load_config()
ds_key = config.get("deepseek_api_key", "")
news_count = config.get("news_count", 5)
stop_loss_val = st.session_state.user_settings.get('stop_loss', -0.05) if 'user_settings' in st.session_state else -0.05
sim_num = st.session_state.user_settings.get('sim_num', 3000) if 'user_settings' in st.session_state else 3000

# 数据接力函数：检查session_state中是否已有现成数据，避免重复初始化
def get_ready_data():
    """
    数据接力：如果主控台已经加载了数据，直接使用；否则调用服务层初始化。
    返回包含所有必要数据的字典。

    数据新鲜度检查：
    1. 检查股票列表是否一致
    2. 检查数据是否过期（15分钟）
    3. 如果不一致或过期，重新加载数据
    """
    # 数据完整性检查
    # 注意：增量更新会清除这些缓存，所以不存在时会自动重新加载
    has_all_data = ('analysis_data' in st.session_state and
                    'returns' in st.session_state and
                    'best_p' in st.session_state and
                    'latest_prices' in st.session_state)

    # 股票列表一致性检查
    stocks_consistent = True
    if has_all_data and 'analysis_stocks' in st.session_state:
        cached_stocks = set(st.session_state.analysis_stocks.values())
        current_stocks = set(analysis_stocks.values())
        stocks_consistent = (cached_stocks == current_stocks)
        if not stocks_consistent:
            st.info("🔄 检测到股票列表变化，正在重新加载数据...")

    # 数据新鲜度检查（15分钟）
    data_fresh = True
    if has_all_data and 'data_timestamp' in st.session_state:
        cache_age = datetime.now() - st.session_state.data_timestamp
        if cache_age > timedelta(minutes=15):
            data_fresh = False
            st.info("🔄 数据已过期（>15分钟），正在刷新...")

    # 如果存在分析数据、股票列表一致、数据新鲜，则直接使用现有数据
    if (has_all_data and stocks_consistent and data_fresh):
        st.success("✅ 使用主控台已加载的数据（数据接力模式）")
        return {
            'data': st.session_state.analysis_data,
            'returns': st.session_state.returns,
            'mean_ret': st.session_state.get('mean_ret', None),
            'cov_mat': st.session_state.get('cov_mat', None),
            'sim_res': st.session_state.get('sim_res', None),
            'best_p': st.session_state.best_p,
            'latest_prices': st.session_state.latest_prices
        }

    # 如果没有现成数据、股票列表变化、数据过期或请求了增量更新，调用服务层初始化
    with st.spinner("从中心库调阅病历中..."):
        try:
            page_data = initialize_page_data(analysis_stocks, config, st.session_state.user_settings)

            # 保存到session_state，供后续使用
            st.session_state.analysis_data = page_data['data']
            st.session_state.returns = page_data['returns']
            st.session_state.mean_ret = page_data.get('mean_ret')
            st.session_state.cov_mat = page_data.get('cov_mat')
            st.session_state.sim_res = page_data.get('sim_res')
            st.session_state.best_p = page_data['best_p']
            st.session_state.latest_prices = page_data['latest_prices']
            st.session_state.analysis_stocks = analysis_stocks  # 保存股票列表用于一致性检查
            st.session_state.data_timestamp = datetime.now()  # 保存时间戳用于新鲜度检查

            return page_data

        except DataServiceError as e:
            st.error(f"数据初始化失败: {e}")
            st.stop()
        except Exception as e:
            st.error(f"未知错误: {e}")
            st.stop()

# 获取数据（使用接力模式）
page_data = get_ready_data()

# 提取数据
data = page_data['data']
returns = page_data['returns']
mean_ret = page_data.get('mean_ret')
cov_mat = page_data.get('cov_mat')
sim_res = page_data.get('sim_res')
best_p = page_data['best_p']
latest_prices = page_data['latest_prices']

# 页面内容开始
st.subheader("📈 实盘持仓监测")
st.caption("基于真实交易记录计算盈亏情况")

# 辅助函数：判断是否为A股交易时间
def is_trading_time():
    """
    判断当前是否为A股交易时间
    交易时间：周一至周五 9:30-11:30, 13:00-15:00（排除法定节假日，这里简单判断）
    """
    now = datetime.now()
    # 检查是否为周末
    if now.weekday() >= 5:  # 5=周六,6=周日
        return False
    
    # 检查时间
    current_time = now.time()
    morning_start = datetime.strptime("09:30", "%H:%M").time()
    morning_end = datetime.strptime("11:30", "%H:%M").time()
    afternoon_start = datetime.strptime("13:00", "%H:%M").time()
    afternoon_end = datetime.strptime("15:00", "%H:%M").time()
    
    if (morning_start <= current_time <= morning_end) or (afternoon_start <= current_time <= afternoon_end):
        return True
    return False

# 1. 刷新控制区 (放在 fragment 外，确保控制组件本身不被局部刷新频率干扰)
col_refresh1, col_refresh2 = st.columns([1, 3])
with col_refresh1:
    # 检查是否为交易时间
    is_trading = is_trading_time()

    # 如果不是交易时间，禁用自动刷新并给出提示
    if is_trading:
        auto_refresh = st.checkbox("⏱️ 启用自动刷新", value=False, key="auto_refresh_t6")
        # 交易时间：使用较短的刷新频率（2分钟），确保实时性
        refresh_freq = 120 if auto_refresh else None
        if auto_refresh:
            st.caption("🔄 交易时间内每2分钟自动刷新，强制获取最新行情")
    else:
        st.info("⏸️ 当前为非交易时间，自动刷新已禁用")
        auto_refresh = False
        refresh_freq = None

# 保存交易时间状态到session_state，供fragment使用
st.session_state.is_trading_time = is_trading

# 2. 定义局部刷新片段
# run_every 会在不重绘整个页面的情况下，只触发该函数内部逻辑
@st.fragment(run_every=refresh_freq)
def render_portfolio_fragment_v2():
    # 加载持仓 (每次片段执行都会重新 load 确保准确)
    current_config = load_config()
    current_portfolio = current_config.get('portfolio', [])
    
    if not current_portfolio:
        st.info("暂无真实持仓记录。请在侧边栏录入交易。")
        return
    
    # 注意：manual_btn 放在内部，点击它只会触发本片段刷新
    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        refresh_now = st.button("🔄 立即刷新", type="primary", key="inner_refresh_t6")
    
    # 处理缓存清理
    if refresh_now:
        st.toast("正在获取最新行情...", icon="🔄")
        # 设置强制刷新标志
        st.session_state.force_refresh_live_portfolio = True
        # 清除 safe_fetch_latest_prices 的缓存，确保获取最新价格
        # 注意：缓存键使用排序后的代码（与market_data_service.py保持一致）
        temp_codes = [trade['code'] for trade in current_portfolio]
        cache_key = f'_latest_prices_cache_{",".join(sorted(temp_codes))}'  # 缓存键使用排序
        cache_time_key = f'_latest_prices_cache_time_{",".join(sorted(temp_codes))}'  # 缓存键使用排序
        if cache_key in st.session_state:
            del st.session_state[cache_key]
        if cache_time_key in st.session_state:
            del st.session_state[cache_time_key]
        st.toast("已清除价格缓存，正在获取最新行情...", icon="🔄")

    # 准备数据并过滤持仓
    # 为什么要这样做：优化后的数据获取可能因为网络问题或股票代码问题导致某些股票没有数据
    # 如果直接使用这些股票，会导致KeyError，因为data中没有这些列

    # 先过滤出在历史数据中存在的持仓
    filtered_portfolio = [trade for trade in current_portfolio if trade['name'] in data.columns]

    if len(filtered_portfolio) != len(current_portfolio):
        missing_trades = [trade for trade in current_portfolio if trade['name'] not in data.columns]
        missing = {trade['name'] for trade in missing_trades}
        st.warning(f"以下持仓股票在历史数据中不存在，已从本次分析中排除: {missing}")

    # 统一更新所有相关变量，确保一致性
    current_portfolio = filtered_portfolio
    real_stocks = [trade['name'] for trade in filtered_portfolio]
    real_codes = [trade['code'] for trade in filtered_portfolio]  # 不排序，保持与real_stocks的对应关系
    
    # 如果过滤后没有持仓，直接返回
    if not current_portfolio:
        st.info("当前持仓中没有在历史数据中找到对应股票的记录，无法计算盈亏。")
        return

    # 检查缓存是否有效
    cache_valid = False
    cache_key = 'live_portfolio_cache'
    force_refresh = st.session_state.get('force_refresh_live_portfolio', False)

    # 判断是否需要强制网络请求
    # 优化：只在必要时才从网络获取实时快照（7分钟），平时使用本地数据（<1秒）
    # 触发网络请求的条件：
    # 1. 用户点击"立即刷新"按钮 (force_refresh=True)
    # 2. 交易时间内 (is_trading=True)
    is_trading = st.session_state.get('is_trading_time', False)
    should_force_network = force_refresh or is_trading  # 智能判断：仅在刷新或交易时间才联网

    # 如果存在缓存，检查持仓是否变化、缓存是否过期
    if cache_key in st.session_state and not force_refresh:
        cache = st.session_state[cache_key]
        # 检查持仓是否相同（简单通过比较持仓字符串）
        cached_portfolio_str = str(cache.get('portfolio', []))
        current_portfolio_str = str(current_portfolio)
        # 检查时间戳（交易时间2分钟，非交易时间10分钟）
        cache_age = datetime.now() - cache.get('timestamp', datetime.min)
        cache_ttl = 120 if is_trading else 600  # 交易时间2分钟，非交易时间10分钟
        # 检查缓存数据是否完整且summary_df非空
        if (cached_portfolio_str == current_portfolio_str and
            cache_age.total_seconds() < cache_ttl and
            not cache.get('summary_df', pd.DataFrame()).empty):  # 确保缓存数据有效
            cache_valid = True

    if cache_valid and not should_force_network:
        # 使用缓存数据
        summary_df = st.session_state[cache_key]['summary_df']
        total_market_value = st.session_state[cache_key]['total_market_value']
        total_profit_ratio = st.session_state[cache_key]['total_profit_ratio']
        latest_price_df = st.session_state[cache_key]['latest_price_df']
        data_source = st.session_state[cache_key].get('data_source', '缓存')
        st.toast(f"使用缓存数据（{data_source}）", icon="💾")
    else:
        # 获取价格 (使用 spinner 提示局部加载状态)
        with st.spinner("同步实时行情..."):
            # 优化：只获取持仓股票的价格，而不是所有分析股票
            # 交易时间或手动刷新时，强制穿透缓存从网络获取实时快照
            latest_price_df = safe_fetch_latest_prices(real_codes, real_stocks, force_network=should_force_network)
        
        # 备用逻辑：如果实时行情获取失败，使用历史数据中的最近一行
        if latest_price_df.empty or latest_price_df.isna().all().all():
            # 确保real_stocks中的股票都在data中（经过前面过滤，这里应该都在）
            if real_stocks and all(stock in data.columns for stock in real_stocks):
                latest_price_df = data[real_stocks].iloc[[-1]]

                # 检查历史数据是否全是NaN
                if latest_price_df.isna().all().all():
                    st.error("❌ 历史数据中没有有效的价格数据，无法计算盈亏。请检查股票代码是否正确。")
                    return

                # 检查是否有部分股票的价格为NaN
                nan_stocks = latest_price_df.columns[latest_price_df.isna().all()].tolist()
                if nan_stocks:
                    st.warning(f"⚠️ 以下股票在历史数据中价格缺失: {', '.join(nan_stocks)}")
            else:
                st.error("❌ 无法获取实时行情，且历史数据中也没有对应股票的数据。")
                return

        # 计算盈亏
        summary_df, total_market_value, total_profit_ratio = get_portfolio_summary(current_portfolio, latest_price_df)

        # 判断数据来源
        # 修改：由于现在始终获取最新价格，数据来源取决于交易状态
        if is_trading:
            data_source = "实时行情"
        else:
            data_source = "最新快照"  # 非交易时间使用最新快照（可能是今日收盘价或昨日收盘价）

        # 更新缓存
        st.session_state[cache_key] = {
            'portfolio': current_portfolio,
            'latest_price_df': latest_price_df,
            'summary_df': summary_df,
            'total_market_value': total_market_value,
            'total_profit_ratio': total_profit_ratio,
            'timestamp': datetime.now(),
            'data_source': data_source
        }
        # 重置强制刷新标志
        st.session_state.force_refresh_live_portfolio = False
    
    if not summary_df.empty:
        # 显示数据来源和更新时间
        cache_timestamp = st.session_state[cache_key].get('timestamp', datetime.now())
        data_source = st.session_state[cache_key].get('data_source', '未知')
        col_info1, col_info2 = st.columns([2, 1])
        with col_info1:
            st.caption(f"📊 数据来源: **{data_source}** | 🕒 更新时间: {cache_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        with col_info2:
            # 显示交易状态
            if is_trading:
                st.caption("🟢 **交易时间** - 自动获取实时行情")
            else:
                st.caption("🔴 **非交易时间** - 使用历史收盘价")

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

        # 初始化编辑状态
        if '_editing_stock' not in st.session_state:
            st.session_state._editing_stock = None

        # 显示表格 + 快捷编辑按钮
        for idx, row in summary_df.iterrows():
            stock_code = row['代码']
            stock_name = row['资产名称'].replace(' 🚫停牌', '')  # 去掉停牌标记
            cost_price = row['成本价']
            current_price = row['现价']
            quantity = row['持仓量']
            cost = row['成本']
            market_val = row['市值']
            profit = row['盈亏额']
            profit_ratio = row['盈亏比']

            # 盈亏颜色
            profit_color = "#ff4444" if profit > 0 else "#00cc00"

            # 判断是否正在编辑该股票
            is_editing = st.session_state._editing_stock == stock_code

            if is_editing:
                # 编辑模式
                st.markdown(f"**✏️ 编辑 {stock_name} ({stock_code})**")
                col_e1, col_e2, col_e3, col_e4 = st.columns([2, 2, 1, 1])
                with col_e1:
                    new_cost_price = st.number_input(
                        "成本价",
                        value=float(cost_price),
                        min_value=0.01,
                        step=0.01,
                        format="%.2f",
                        key=f"edit_cost_{stock_code}"
                    )
                with col_e2:
                    new_quantity = st.number_input(
                        "持仓量",
                        value=int(quantity),
                        min_value=1,
                        step=100,
                        key=f"edit_qty_{stock_code}"
                    )
                with col_e3:
                    if st.button("💾 保存", key=f"save_{stock_code}", type="primary"):
                        # 保存更新
                        cfg = load_config()
                        trade_idx = find_trade_index(cfg, stock_code)
                        if trade_idx >= 0:
                            update_trade(cfg, trade_idx, {
                                'buy_price': new_cost_price,
                                'quantity': new_quantity
                            })
                            st.session_state._editing_stock = None
                            # 只清除缓存，不触发网络刷新（行情数据不变，只需重新计算盈亏）
                            if 'live_portfolio_cache' in st.session_state:
                                del st.session_state['live_portfolio_cache']
                            st.success(f"已更新 {stock_name}")
                            st.rerun()
                        else:
                            st.error("未找到该持仓记录")
                with col_e4:
                    if st.button("❌ 取消", key=f"cancel_{stock_code}"):
                        st.session_state._editing_stock = None
                        st.rerun()
                st.markdown("---")
            else:
                # 显示模式 - 深色背景 + 亮色文字，高对比度
                if profit > 0:
                    bg_color = "rgba(255, 68, 68, 0.15)"  # 淡红色背景（盈利）
                    border_color = "#ff5252"
                    profit_text_color = "#ff5252"  # 亮红色盈亏文字
                    status_icon = "📈"
                elif profit < 0:
                    bg_color = "rgba(0, 230, 118, 0.15)"  # 淡绿色背景（亏损）
                    border_color = "#00e676"
                    profit_text_color = "#00e676"  # 亮绿色盈亏文字
                    status_icon = "📉"
                else:
                    bg_color = "rgba(158, 158, 158, 0.12)"  # 灰色背景（持平）
                    border_color = "#9e9e9e"
                    profit_text_color = "#bdbdbd"
                    status_icon = "➖"

                # 使用容器包装，添加卡片样式
                with st.container():
                    st.markdown(f"""
                    <div style="
                        background: {bg_color};
                        border-left: 6px solid {border_color};
                        border-radius: 10px;
                        padding: 18px 24px;
                        margin-bottom: 12px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                            <div style="flex: 2; min-width: 140px;">
                                <div style="font-size: 20px; font-weight: 700; color: #ffffff;">{stock_name}</div>
                                <div style="font-size: 14px; color: #cccccc; margin-top: 4px;">{stock_code}</div>
                            </div>
                            <div style="flex: 1; text-align: center; min-width: 85px;">
                                <div style="font-size: 12px; color: #aaaaaa; margin-bottom: 6px;">成本价</div>
                                <div style="font-size: 18px; font-weight: 600; color: #ffffff;">{cost_price:,.2f}</div>
                            </div>
                            <div style="flex: 1; text-align: center; min-width: 85px;">
                                <div style="font-size: 12px; color: #aaaaaa; margin-bottom: 6px;">现价</div>
                                <div style="font-size: 18px; font-weight: 600; color: #ffffff;">{current_price:,.2f}</div>
                            </div>
                            <div style="flex: 1; text-align: center; min-width: 85px;">
                                <div style="font-size: 12px; color: #aaaaaa; margin-bottom: 6px;">持仓量</div>
                                <div style="font-size: 18px; font-weight: 600; color: #ffffff;">{int(quantity):,}</div>
                            </div>
                            <div style="flex: 1.2; text-align: center; min-width: 100px;">
                                <div style="font-size: 12px; color: #aaaaaa; margin-bottom: 6px;">市值</div>
                                <div style="font-size: 18px; font-weight: 600; color: #ffffff;">{market_val:,.2f}</div>
                            </div>
                            <div style="flex: 1.8; text-align: right; min-width: 130px; padding-left: 12px; border-left: 1px solid rgba(255,255,255,0.3);">
                                <div style="font-size: 12px; color: #aaaaaa; margin-bottom: 6px;">盈亏 {status_icon}</div>
                                <div style="font-size: 26px; font-weight: 700; color: {profit_text_color};">{profit:+,.2f}</div>
                                <div style="font-size: 18px; font-weight: 600; color: {profit_text_color}; margin-top: 2px;">{profit_ratio:+.2%}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 编辑按钮放在卡片下方右侧
                    col_spacer, col_edit = st.columns([14, 1])
                    with col_edit:
                        if st.button("✏️", key=f"edit_btn_{stock_code}", help="快捷修改成本价和持仓量"):
                            st.session_state._editing_stock = stock_code
                            st.rerun()

        # --- 偏离度分析 ---
        st.subheader("⚖️ 配置偏离度分析")
        current_weights = {row['资产名称']: row['市值']/total_market_value for _, row in summary_df.iterrows()}
        # 过滤可用的股票：只保留在best_p中存在的股票（即在data中成功获取的股票）
        available_stocks = [name for name in analysis_stocks.values() if name in best_p.index]
        if not available_stocks:
            st.warning("无法进行偏离度分析：没有可用的最优权重数据。")
            optimal_weights = {}
        else:
            optimal_weights = best_p[available_stocks].to_dict()
        total_dev, deviations = calculate_deviation(current_weights, optimal_weights)
        
        st.write(f"当前配置与 AI 模型最优前沿的**总偏离度**: `{total_dev:.2%}`")
        
        # 创建饼图比较
        col1, col2 = st.columns(2)
        
        with col1:
            # 当前配置饼图
            current_weights_df = pd.DataFrame({
                '资产': list(current_weights.keys()),
                '权重': list(current_weights.values())
            })
            current_weights_df = current_weights_df[current_weights_df['权重'] > 0]  # 只显示正权重
            if not current_weights_df.empty:
                fig_current = px.pie(current_weights_df, values='权重', names='资产', 
                                    title='📊 当前持仓配置',
                                    color_discrete_sequence=px.colors.qualitative.Set3)
                fig_current.update_traces(textposition='inside', textinfo='percent+label')
                fig_current.update_layout(showlegend=True, height=400)
                st.plotly_chart(fig_current, config=PLOTLY_CONFIG)
            else:
                st.info("暂无持仓配置数据")
        
        with col2:
            # 最优配置饼图
            optimal_weights_df = pd.DataFrame({
                '资产': list(optimal_weights.keys()),
                '权重': list(optimal_weights.values())
            })
            optimal_weights_df = optimal_weights_df[optimal_weights_df['权重'] > 0]  # 只显示正权重
            if not optimal_weights_df.empty:
                fig_optimal = px.pie(optimal_weights_df, values='权重', names='资产', 
                                    title='🎯 AI推荐最优配置',
                                    color_discrete_sequence=px.colors.qualitative.Set2)
                fig_optimal.update_traces(textposition='inside', textinfo='percent+label')
                fig_optimal.update_layout(showlegend=True, height=400)
                st.plotly_chart(fig_optimal, config=PLOTLY_CONFIG)
            else:
                st.info("暂无最优配置数据")
        
        # 偏离度数据表格
        st.subheader("📈 配置偏离度明细")
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
    else:
        st.warning("行情计算返回空数据，请检查网络或标的代码。")

# 加载持仓数据（用于外部判断）
current_config = load_config()
current_portfolio = current_config.get('portfolio', [])

# 3. 执行局部刷新片段
if current_portfolio:  # 只在有持仓时执行片段刷新，避免不必要的数据获取
    render_portfolio_fragment_v2()
else:
    st.info("暂无真实持仓记录。请在侧边栏录入交易。")
