import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 导入自定义模块
from config import load_config
from services.data_service import (
    initialize_page_data,
    get_latest_prices,
    get_trading_signals,
    get_candle_data,
    get_recent_10_days,
    DataServiceError
)
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
st.set_page_config(page_title="买卖与风险预警", layout="wide")

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
    """
    # 检查内存中是否已有完整数据（接力模式）
    # 主控台现在会计算完整数据，包括 latest_prices 和 signals
    # 注意：增量更新会清除这些缓存，所以不存在时会自动重新加载
    has_complete_data = ('analysis_data' in st.session_state and
                         'returns' in st.session_state and
                         'best_p' in st.session_state and
                         'latest_prices' in st.session_state and
                         'signals' in st.session_state)

    if has_complete_data:
        st.success("✅ 使用主控台已加载的数据（数据接力模式）")
        return {
            'data': st.session_state.analysis_data,
            'returns': st.session_state.returns,
            'mean_ret': st.session_state.get('mean_ret', None),
            'cov_mat': st.session_state.get('cov_mat', None),
            'sim_res': st.session_state.get('sim_res', None),
            'best_p': st.session_state.best_p,
            'latest_prices': st.session_state.latest_prices,
            'signals': st.session_state.signals,
            'stock_codes': st.session_state.get('stock_codes', []),
            'stock_names': st.session_state.get('stock_names', [])
        }
    
    # 如果没有现成数据或请求了增量更新，调用服务层初始化
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
            st.session_state.signals = page_data.get('signals', {})
            st.session_state.stock_codes = page_data.get('stock_codes', [])
            st.session_state.stock_names = page_data.get('stock_names', [])

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
signals = page_data.get('signals', {})
stock_codes = page_data.get('stock_codes', [])
stock_names = page_data.get('stock_names', [])

# 页面内容开始
st.subheader("量化决策与风险管理")

# 数据审计状态栏
with st.expander("🔍 数据审计状态", expanded=False):
    from storage_manager import show_data_audit_ui
    show_data_audit_ui()

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
# 过滤掉 data 中不存在的股票
available_stocks = [name for name in analysis_stocks.values() if name in data.columns]
if not available_stocks:
    st.warning("没有可用的股票数据生成信号。")
    sig_df = pd.DataFrame()
else:
    best_weights = best_p[available_stocks].to_dict()
    # 使用统一数据服务获取交易信号
    sig_df = get_trading_signals(data, best_weights, stop_loss_val, latest_prices=latest_prices)
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

# 创建涨跌颜色样式函数
def color_price_changes(df):
    """
    根据价格涨跌为单元格着色
    红色表示上涨，绿色表示下跌，白色表示持平或无可比数据
    """
    # 创建空样式DataFrame
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    # 对数值列进行处理
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # 获取当前列的价格数据
        prices = df[col].values

        # 从第二行开始比较（第一行没有前一天数据）
        for i in range(1, len(prices)):
            if pd.notna(prices[i]) and pd.notna(prices[i-1]):
                if prices[i] > prices[i-1]:
                    # 上涨：红色
                    styles.iloc[i, df.columns.get_loc(col)] = 'color: #ff4444; font-weight: bold'
                elif prices[i] < prices[i-1]:
                    # 下跌：绿色
                    styles.iloc[i, df.columns.get_loc(col)] = 'color: #00cc66; font-weight: bold'
                else:
                    # 持平：白色
                    styles.iloc[i, df.columns.get_loc(col)] = 'color: #ffffff'

        # 第一行使用白色（没有前一天可比数据）
        if len(prices) > 0 and pd.notna(prices[0]):
            styles.iloc[0, df.columns.get_loc(col)] = 'color: #ffffff'

    return styles

# 应用样式并显示
numeric_cols = recent_data.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    st.dataframe(
        recent_data.style
        .apply(color_price_changes, axis=None)
        .format("{:.2f}", subset=numeric_cols)
    )
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
