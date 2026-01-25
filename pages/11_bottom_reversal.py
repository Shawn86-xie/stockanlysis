"""
触底反弹形态挖掘页面

功能：
1. 获取整个市场标的数据
2. 构建触底反弹目标模型
3. 计算各标的与目标模型的匹配度
4. 列出前10个匹配度最高的标的
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# 认证保护
from auth import protect_page
protect_page()
from config import load_config, save_config
from services.market_stock_list import (
    load_market_stock_list,
    get_market_stock_list_info,
    refresh_market_stock_list
)
from services.market_data_service import safe_fetch_candle_data
from services.kline_database import (
    get_kline_db_stats,
    get_cached_stock_codes,
    batch_download_kline,
    incremental_update_kline,
    scan_with_local_db,
    read_local_kline
)

# 页面配置
st.set_page_config(page_title="触底反弹挖掘", layout="wide")


class BottomReboundPattern:
    """
    触底反弹形态识别模型
    核心：捕捉从下跌到触底再到反弹的转折点
    """

    def __init__(self, params=None):
        """
        初始化形态参数

        Args:
            params: 可选的参数字典，用于覆盖默认参数
        """
        # ===== 第一阶段：前期下跌 =====
        self.prior_decline_period = 60      # 前期观察周期
        self.prior_decline_min = -20        # 前期最小跌幅-20%

        # ===== 第二阶段：触底特征 =====
        self.bottom_period = 20             # 触底观察周期
        self.bottom_range_max = 0.10        # 底部震荡幅度<10%
        self.near_bottom_threshold = 1.05   # 当前价格接近底部(5%以内)

        # ===== 第三阶段：反弹启动 =====
        self.rebound_period = 5             # 反弹观察周期
        self.rebound_min = 2                # 最近5日涨幅>2%
        self.rebound_days = 3               # 最近5日中至少3日上涨

        # ===== 辅助确认指标 =====
        self.volume_shrink = 0.7            # 底部缩量：成交量<前期70%
        self.volatility_decline = 0.6       # 波动率下降

        # 如果提供了自定义参数，覆盖默认值
        if params:
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def calculate_indicators(self, df):
        """
        计算所需指标

        Args:
            df: 包含 '收盘', '成交量' 列的 DataFrame

        Returns:
            添加了指标列的 DataFrame
        """
        df = df.copy()

        # 确保数据类型正确
        df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
        df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce')

        # 价格变化
        df['return_60d'] = df['收盘'].pct_change(60) * 100
        df['return_20d'] = df['收盘'].pct_change(20) * 100
        df['return_5d'] = df['收盘'].pct_change(5) * 100

        # 底部位置识别
        df['low_60d'] = df['收盘'].rolling(60).min()
        df['low_20d'] = df['收盘'].rolling(20).min()
        df['price_to_low'] = df['收盘'] / df['low_20d']

        # 底部震荡幅度
        df['high_20d'] = df['收盘'].rolling(20).max()
        df['bottom_range'] = (df['high_20d'] - df['low_20d']) / df['low_20d']

        # 反弹确认
        df['close_change'] = df['收盘'].pct_change()
        df['up_days_5'] = (df['close_change'] > 0).rolling(5).sum()

        # 成交量变化
        df['volume_ma_20'] = df['成交量'].rolling(20).mean()
        df['volume_ma_60'] = df['成交量'].rolling(60).mean()
        df['volume_ratio'] = df['volume_ma_20'] / df['volume_ma_60']

        # 波动率
        df['volatility_20d'] = df['收盘'].pct_change().rolling(20).std()
        df['volatility_60d'] = df['收盘'].pct_change().rolling(60).std()
        df['vol_ratio'] = df['volatility_20d'] / df['volatility_60d']

        # 是否在反弹中
        df['is_rebounding'] = (df['收盘'] > df['收盘'].shift(1)) & \
                              (df['收盘'].shift(1) > df['收盘'].shift(2))

        return df

    def check_pattern(self, df):
        """
        检查是否符合触底反弹形态

        Args:
            df: 已计算指标的 DataFrame

        Returns:
            tuple: (是否匹配, 详情字典)
        """
        if len(df) < 70:
            return False, {'score': 0, 'reason': '数据不足70天'}

        latest = df.iloc[-1]

        # 检查数据有效性
        required_cols = ['return_60d', 'bottom_range', 'price_to_low', 'return_20d',
                        'return_5d', 'up_days_5', 'volume_ratio', 'vol_ratio']
        for col in required_cols:
            if pd.isna(latest[col]):
                return False, {'score': 0, 'reason': f'{col}数据无效'}

        # 阶段1：前期有明显下跌
        phase1_decline = latest['return_60d'] < self.prior_decline_min

        # 阶段2：近期在底部横盘（窄幅震荡）
        phase2_bottom = (
            latest['bottom_range'] < self.bottom_range_max and
            latest['price_to_low'] < self.near_bottom_threshold and
            abs(latest['return_20d']) < 10  # 近20日横盘，涨跌幅不大
        )

        # 阶段3：开始反弹
        phase3_rebound = (
            latest['return_5d'] > self.rebound_min and
            latest['up_days_5'] >= self.rebound_days
        )

        # 辅助确认：成交量萎缩、波动率下降
        confirm_volume = latest['volume_ratio'] < self.volume_shrink
        confirm_volatility = latest['vol_ratio'] < self.volatility_decline

        # 综合判断
        is_match = phase1_decline and phase2_bottom and phase3_rebound

        # 强度评分（越高越符合）
        score = 0
        if phase1_decline: score += 30
        if phase2_bottom: score += 30
        if phase3_rebound: score += 30
        if confirm_volume: score += 5
        if confirm_volatility: score += 5

        details = {
            'score': score,
            'return_60d': round(latest['return_60d'], 2) if not pd.isna(latest['return_60d']) else 0,
            'return_20d': round(latest['return_20d'], 2) if not pd.isna(latest['return_20d']) else 0,
            'return_5d': round(latest['return_5d'], 2) if not pd.isna(latest['return_5d']) else 0,
            'bottom_range': round(latest['bottom_range'] * 100, 2) if not pd.isna(latest['bottom_range']) else 0,
            'price_to_low': round(latest['price_to_low'], 3) if not pd.isna(latest['price_to_low']) else 0,
            'up_days_5': int(latest['up_days_5']) if not pd.isna(latest['up_days_5']) else 0,
            'volume_ratio': round(latest['volume_ratio'], 3) if not pd.isna(latest['volume_ratio']) else 0,
            'vol_ratio': round(latest['vol_ratio'], 3) if not pd.isna(latest['vol_ratio']) else 0,
            'phase1': phase1_decline,
            'phase2': phase2_bottom,
            'phase3': phase3_rebound,
            'confirm_volume': confirm_volume,
            'confirm_volatility': confirm_volatility,
            'current_price': round(latest['收盘'], 2) if not pd.isna(latest['收盘']) else 0
        }

        return is_match, details


def filter_valid_stocks(stock_list):
    """
    过滤掉ST、退市、科创板等股票

    Args:
        stock_list: 股票列表 [{'code': '...', 'name': '...'}, ...]

    Returns:
        过滤后的股票列表
    """
    filtered = []
    for stock in stock_list:
        code = stock['code']
        name = stock['name']

        # 过滤ST股票
        if 'ST' in name.upper() or '*ST' in name.upper():
            continue

        # 过滤退市股票
        if '退' in name:
            continue

        # 过滤北交所股票 (8开头)
        if code.startswith('8'):
            continue

        # 可选：过滤科创板 (688开头) - 注释掉如果需要包含科创板
        # if code.startswith('688'):
        #     continue

        filtered.append(stock)

    return filtered


def plot_rebound_chart(code, name, df, pattern, lookback_days=250):
    """
    绘制触底反弹分析图表

    Args:
        code: 股票代码
        name: 股票名称
        df: K线数据
        pattern: BottomReboundPattern 实例（用于计算指标）
        lookback_days: 显示最近多少天的数据

    Returns:
        plotly.graph_objects.Figure
    """
    # 计算指标
    df = pattern.calculate_indicators(df)

    # 截取最近数据
    df_plot = df.tail(lookback_days).copy()

    # 创建K线图
    fig = go.Figure()

    # 添加K线（蜡烛图）
    fig.add_trace(go.Candlestick(
        x=df_plot['日期'],
        open=df_plot['开盘'],
        high=df_plot['最高'],
        low=df_plot['最低'],
        close=df_plot['收盘'],
        name='K线',
        increasing_line_color='red',
        decreasing_line_color='green'
    ))

    # 添加 20日均线
    df_plot['ma20'] = df_plot['收盘'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=df_plot['日期'],
        y=df_plot['ma20'],
        mode='lines',
        name='MA20',
        line=dict(color='blue', width=1.5)
    ))

    # 添加 60日均线
    df_plot['ma60'] = df_plot['收盘'].rolling(window=60).mean()
    fig.add_trace(go.Scatter(
        x=df_plot['日期'],
        y=df_plot['ma60'],
        mode='lines',
        name='MA60',
        line=dict(color='gold', width=2)
    ))

    # 标记20日最低点
    if 'low_20d' in df_plot.columns:
        # 找到最近的20日最低价位置
        recent_20 = df_plot.tail(20)
        min_idx = recent_20['收盘'].idxmin()
        if min_idx in df_plot.index:
            min_date = df_plot.loc[min_idx, '日期']
            min_price = df_plot.loc[min_idx, '收盘']
            fig.add_trace(go.Scatter(
                x=[min_date],
                y=[min_price],
                mode='markers+text',
                marker=dict(color='green', size=12, symbol='triangle-up'),
                text=['底部'],
                textposition='bottom center',
                name='20日底部',
                showlegend=True
            ))

    # 高亮最近5日反弹区域
    if len(df_plot) >= 5:
        last_5_dates = df_plot['日期'].tail(5)
        last_5_high = df_plot['最高'].tail(5).max()
        last_5_low = df_plot['最低'].tail(5).min()

        fig.add_trace(go.Scatter(
            x=[last_5_dates.iloc[0], last_5_dates.iloc[0], last_5_dates.iloc[-1], last_5_dates.iloc[-1], last_5_dates.iloc[0]],
            y=[last_5_low, last_5_high, last_5_high, last_5_low, last_5_low],
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(color='rgba(255, 165, 0, 0.5)', width=1),
            name='5日反弹区',
            hoverinfo='skip'
        ))

    # 添加当前价格标注
    last_price = df_plot['收盘'].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[df_plot['日期'].iloc[-1]],
        y=[last_price],
        mode='markers+text',
        marker=dict(color='blue', size=10),
        text=[f'{last_price:.2f}'],
        textposition='top right',
        name='现价',
        showlegend=False
    ))

    # 布局设置
    fig.update_layout(
        title=f'{name} ({code}) - 触底反弹分析',
        xaxis_title='日期',
        yaxis_title='价格',
        template='plotly_white',
        height=500,
        xaxis_rangeslider_visible=False
    )

    return fig


def scan_market_for_rebound(stock_list, pattern, progress_bar=None, status_text=None, stats_container=None, max_stocks=None):
    """
    扫描市场寻找触底反弹的标的

    Args:
        stock_list: 股票列表
        pattern: BottomReboundPattern 实例
        progress_bar: Streamlit 进度条对象
        status_text: Streamlit 文本占位符
        stats_container: Streamlit 统计容器
        max_stocks: 最大扫描数量（用于测试）

    Returns:
        匹配结果 DataFrame
    """
    import time

    results = []
    total = len(stock_list) if max_stocks is None else min(len(stock_list), max_stocks)

    # 统计计数
    success_count = 0
    error_count = 0
    skip_count = 0
    match_count = 0

    start_time = time.time()

    for i, stock in enumerate(stock_list[:total]):
        code = stock['code']
        name = stock['name']

        # 每只股票都更新进度
        if progress_bar:
            progress_bar.progress((i + 1) / total)

        if status_text:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            status_text.text(
                f"[{i+1}/{total}] 正在分析: {name} ({code}) | "
                f"速度: {speed:.1f}只/秒 | 预计剩余: {eta/60:.1f}分钟"
            )

        # 更新实时统计
        if stats_container and i % 5 == 0:
            stats_container.markdown(
                f"✅ 成功: **{success_count}** | "
                f"⏭️ 跳过: **{skip_count}** | "
                f"❌ 失败: **{error_count}** | "
                f"🎯 发现匹配: **{match_count}**"
            )

        try:
            # 获取K线数据（120天足够计算60日指标）
            df = safe_fetch_candle_data(code, days=120)

            if df is None or df.empty or len(df) < 70:
                skip_count += 1
                continue

            # 计算指标
            df = pattern.calculate_indicators(df)

            # 检查形态
            is_match, details = pattern.check_pattern(df)

            success_count += 1

            # 只要评分大于0就记录（不仅仅是完全匹配的）
            if details['score'] > 0:
                match_count += 1
                results.append({
                    '代码': code,
                    '名称': name,
                    '评分': details['score'],
                    '60日跌幅%': details['return_60d'],
                    '20日涨跌%': details['return_20d'],
                    '5日涨幅%': details['return_5d'],
                    '底部震荡%': details['bottom_range'],
                    '距底部': details['price_to_low'],
                    '5日上涨天数': details['up_days_5'],
                    '量比': details['volume_ratio'],
                    '波动比': details['vol_ratio'],
                    '现价': details['current_price'],
                    '阶段1_下跌': '✓' if details['phase1'] else '✗',
                    '阶段2_触底': '✓' if details['phase2'] else '✗',
                    '阶段3_反弹': '✓' if details['phase3'] else '✗',
                    '完全匹配': is_match
                })

        except Exception as e:
            error_count += 1
            continue

    # 最终统计更新
    if stats_container:
        stats_container.markdown(
            f"✅ 成功: **{success_count}** | "
            f"⏭️ 跳过: **{skip_count}** | "
            f"❌ 失败: **{error_count}** | "
            f"🎯 发现匹配: **{match_count}**"
        )

    # 按评分排序
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('评分', ascending=False)

    return results_df


# ==================== 页面主体 ====================

st.title("🎯 触底反弹形态挖掘")
st.caption("基于「前期下跌 → 触底 → 反弹启动」三阶段模型，挖掘潜在优势个股")

# 数据源状态检查
market_info = get_market_stock_list_info()

with st.expander("📊 市场数据库状态", expanded=False):
    if market_info['exists']:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("股票数量", f"{market_info['count']:,} 只")
        with col2:
            st.metric("更新时间", market_info['update_time'])
        with col3:
            if st.button("🔄 刷新市场库"):
                with st.spinner("正在从网络获取最新市场数据..."):
                    result = refresh_market_stock_list()
                    if result['success']:
                        st.success(f"刷新成功！共 {result['count']:,} 只股票")
                        st.rerun()
                    else:
                        st.error(f"刷新失败：{result['message']}")
    else:
        st.warning("⚠️ 市场股票库尚未初始化")
        if st.button("📥 初始化市场库"):
            with st.spinner("正在从网络获取市场数据，首次可能需要30秒..."):
                result = refresh_market_stock_list()
                if result['success']:
                    st.success(f"初始化成功！共 {result['count']:,} 只股票")
                    st.rerun()
                else:
                    st.error(f"初始化失败：{result['message']}")
        st.stop()

# ==================== K线数据库管理 ====================
st.markdown("---")
st.subheader("💾 K线数据库管理")
st.caption("预先下载全市场K线数据到本地，扫描时直接读取本地数据，速度提升10-100倍")

# 获取K线数据库统计
kline_stats = get_kline_db_stats()

# 显示数据库状态
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("已缓存股票", f"{kline_stats['total_count']:,} 只")
with col2:
    st.metric("数据库大小", f"{kline_stats['total_size_mb']:.1f} MB")
with col3:
    st.metric("今日已更新", f"{kline_stats.get('today_updated', 0):,} 只")
with col4:
    st.metric("最近更新", kline_stats.get('latest_update', '无')[:10] if kline_stats.get('latest_update') else '无')

# 数据库操作按钮
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 批量下载全市场K线", type="primary", use_container_width=True,
                 help="首次使用请点击此按钮下载全市场数据"):
        st.session_state.kline_download_mode = 'full'

with col2:
    if st.button("🔄 增量更新数据", use_container_width=True,
                 help="只更新过期的数据，速度更快"):
        st.session_state.kline_download_mode = 'incremental'

with col3:
    # 显示缓存覆盖率
    if market_info['exists'] and kline_stats['total_count'] > 0:
        coverage = kline_stats['total_count'] / market_info['count'] * 100
        st.metric("缓存覆盖率", f"{coverage:.1f}%")
    else:
        st.metric("缓存覆盖率", "0%")

# 下载参数设置（始终显示）
with st.expander("⚙️ 下载参数设置", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        max_workers = st.slider("并行线程数", 1, 10, 5, key="rebound_kline_max_workers",
                                help="线程数越多越快，但可能触发API限流")
    with col2:
        skip_existing = st.checkbox("跳过已有数据", value=False, key="rebound_kline_skip_existing",
                                    help="取消勾选可重新下载覆盖旧数据（获取更长历史）")

# 执行下载任务
if 'kline_download_mode' in st.session_state and st.session_state.kline_download_mode:
    mode = st.session_state.kline_download_mode
    st.session_state.kline_download_mode = None  # 清除状态

    # 加载市场数据
    market_data = load_market_stock_list()
    if not market_data or 'stocks' not in market_data:
        st.error("无法加载市场股票列表")
    else:
        stock_list = market_data['stocks']
        # 过滤ST等股票
        stock_list = filter_valid_stocks(stock_list)

        st.info(f"📋 待处理标的数量：{len(stock_list):,} 只")

        # 创建进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_display = st.empty()

        def download_progress_callback(current, total, success, failed, message):
            progress_bar.progress(current / total)
            status_text.text(f"[{current}/{total}] {message}")
            stats_display.markdown(
                f"✅ 成功: **{success}** | ❌ 失败: **{failed}** | "
                f"进度: **{current/total*100:.1f}%**"
            )

        # 开始下载
        start_time = datetime.now()

        # 从 session_state 获取参数
        workers = st.session_state.get('rebound_kline_max_workers', 5)
        skip = st.session_state.get('rebound_kline_skip_existing', False) if mode == 'full' else False

        if mode == 'full':
            result = batch_download_kline(
                stock_list,
                days=1460,
                max_workers=workers,
                progress_callback=download_progress_callback,
                skip_existing=skip
            )
        else:
            result = incremental_update_kline(
                stock_list,
                max_workers=workers,
                progress_callback=download_progress_callback
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        progress_bar.progress(1.0)
        status_text.text(f"✅ 完成！总耗时 {duration:.1f} 秒 ({duration/60:.1f} 分钟)")

        # 显示结果
        st.success(f"""
        **下载完成！**
        - 成功: {result['success']} 只
        - 失败: {result['failed']} 只
        - 跳过: {result.get('skipped', 0)} 只
        - 耗时: {duration:.1f} 秒
        """)

        if result.get('failed_list'):
            with st.expander(f"❌ 失败列表 ({len(result['failed_list'])} 只)"):
                for item in result['failed_list'][:50]:
                    st.text(f"{item['name']} ({item['code']}): {item['error']}")

        st.rerun()

st.markdown("---")

# ==================== 参数配置面板 ====================
st.subheader("🔧 形态识别参数")

# 初始化参数默认值到 session_state
if 'rebound_params' not in st.session_state:
    st.session_state.rebound_params = {
        'prior_decline_min': -20,
        'bottom_range_max': 10,
        'near_bottom_threshold': 5,
        'rebound_min': 2,
        'rebound_days': 3
    }

with st.expander("⚙️ 调整识别参数（可选）", expanded=False):
    st.markdown("根据市场环境调整识别灵敏度，参数越宽松匹配越多但精度下降")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**阶段1：前期下跌**")
        st.session_state.rebound_params['prior_decline_min'] = st.slider(
            "前期最小跌幅 (%)",
            min_value=-50, max_value=-5,
            value=st.session_state.rebound_params['prior_decline_min'],
            step=5,
            help="60日内的最小跌幅，越小越严格"
        )

    with col2:
        st.markdown("**阶段2：触底特征**")
        st.session_state.rebound_params['bottom_range_max'] = st.slider(
            "底部震荡幅度上限 (%)",
            min_value=5, max_value=20,
            value=st.session_state.rebound_params['bottom_range_max'],
            step=1,
            help="20日内高低点差距，越小说明底部越稳"
        )
        st.session_state.rebound_params['near_bottom_threshold'] = st.slider(
            "距底部距离上限 (%)",
            min_value=2, max_value=10,
            value=st.session_state.rebound_params['near_bottom_threshold'],
            step=1,
            help="当前价格距离20日最低点的距离"
        )

    with col3:
        st.markdown("**阶段3：反弹启动**")
        st.session_state.rebound_params['rebound_min'] = st.slider(
            "5日最小涨幅 (%)",
            min_value=0, max_value=10,
            value=st.session_state.rebound_params['rebound_min'],
            step=1,
            help="最近5日的最小涨幅"
        )
        st.session_state.rebound_params['rebound_days'] = st.slider(
            "5日内上涨天数",
            min_value=2, max_value=5,
            value=st.session_state.rebound_params['rebound_days'],
            step=1,
            help="最近5日中上涨的天数"
        )

# 从 session_state 构建参数字典
custom_params = {
    'prior_decline_min': st.session_state.rebound_params['prior_decline_min'],
    'bottom_range_max': st.session_state.rebound_params['bottom_range_max'] / 100,
    'near_bottom_threshold': 1 + st.session_state.rebound_params['near_bottom_threshold'] / 100,
    'rebound_min': st.session_state.rebound_params['rebound_min'],
    'rebound_days': st.session_state.rebound_params['rebound_days']
}

st.markdown("---")

# ==================== 扫描控制 ====================
st.subheader("🔍 市场扫描")

# 扫描模式选择
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    scan_mode = st.radio(
        "扫描模式",
        options=["⚡ 快速扫描（本地数据）", "🌐 在线扫描（实时数据）"],
        horizontal=True,
        help="快速扫描使用本地缓存数据，速度快；在线扫描获取最新数据，但速度慢"
    )

with col2:
    scan_scope = st.radio(
        "扫描范围",
        options=["全市场", "沪深主板", "创业板", "测试(100只)"],
        horizontal=True
    )

with col3:
    top_n = st.selectbox(
        "显示前N名",
        options=[10, 20, 50, 100],
        index=0
    )

# 检查本地数据库状态
is_fast_mode = "快速扫描" in scan_mode
if is_fast_mode and kline_stats['total_count'] < 100:
    st.warning("⚠️ 本地K线数据库数据不足，建议先点击上方「批量下载全市场K线」按钮下载数据")

# 扫描按钮
col1, col2 = st.columns(2)
with col1:
    start_scan = st.button("🚀 开始扫描", type="primary", use_container_width=True)
with col2:
    if is_fast_mode:
        st.caption(f"📊 本地已有 {kline_stats['total_count']:,} 只股票数据")
    else:
        st.caption("🌐 将从网络获取实时数据")

if start_scan:
    # 加载市场数据
    market_data = load_market_stock_list()
    if not market_data or 'stocks' not in market_data:
        st.error("无法加载市场股票列表")
        st.stop()

    stock_list = market_data['stocks']

    # 过滤ST等股票
    stock_list = filter_valid_stocks(stock_list)

    # 根据扫描范围过滤
    if scan_scope == "沪深主板":
        stock_list = [s for s in stock_list if s['code'].startswith(('60', '00'))]
    elif scan_scope == "创业板":
        stock_list = [s for s in stock_list if s['code'].startswith('30')]
    elif scan_scope == "测试(100只)":
        stock_list = stock_list[:100]

    # 如果是快速模式，只扫描有本地数据的股票
    if is_fast_mode:
        cached_codes = get_cached_stock_codes()
        original_count = len(stock_list)
        stock_list = [s for s in stock_list if s['code'] in cached_codes]
        st.info(f"📋 待扫描标的：{len(stock_list):,} 只（本地有数据）/ 总计 {original_count:,} 只")
    else:
        st.info(f"📋 待扫描标的数量：{len(stock_list):,} 只")

    # 创建进度显示组件
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_container = st.empty()

    # 创建形态识别器
    pattern = BottomReboundPattern(custom_params)

    # 开始扫描
    start_time = datetime.now()

    if is_fast_mode:
        # 快速扫描模式：使用本地数据库
        def fast_progress_callback(current, total, success, skipped, matched, stock_name, speed, eta):
            progress_bar.progress(current / total)
            status_text.text(
                f"[{current}/{total}] 正在分析: {stock_name} | "
                f"速度: {speed:.1f}只/秒 | 预计剩余: {eta/60:.1f}分钟"
            )
            stats_container.markdown(
                f"✅ 成功: **{success}** | "
                f"⏭️ 跳过: **{skipped}** | "
                f"🎯 发现匹配: **{matched}**"
            )

        results_df, scan_stats = scan_with_local_db(
            stock_list,
            pattern,
            progress_callback=fast_progress_callback
        )
        duration = scan_stats['duration']
    else:
        # 在线扫描模式：从网络获取数据
        results_df = scan_market_for_rebound(
            stock_list,
            pattern,
            progress_bar=progress_bar,
            status_text=status_text,
            stats_container=stats_container
        )
        duration = (datetime.now() - start_time).total_seconds()

    progress_bar.progress(1.0)
    status_text.text(f"✅ 扫描完成！总耗时 {duration:.1f} 秒 ({duration/60:.1f} 分钟)")

    # 保存结果到 session_state
    st.session_state.rebound_scan_results = results_df
    st.session_state.rebound_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==================== 结果展示 ====================
if 'rebound_scan_results' in st.session_state and not st.session_state.rebound_scan_results.empty:
    results_df = st.session_state.rebound_scan_results
    scan_time = st.session_state.get('rebound_scan_time', '未知')

    st.markdown("---")
    st.subheader(f"📈 扫描结果（扫描时间：{scan_time}）")

    # 统计信息
    total_found = len(results_df)
    full_match = results_df['完全匹配'].sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("发现标的", f"{total_found} 只")
    with col2:
        st.metric("完全匹配", f"{full_match} 只", help="同时满足三个阶段条件")
    with col3:
        avg_score = results_df['评分'].mean()
        st.metric("平均评分", f"{avg_score:.1f}")

    # 筛选条件
    st.markdown("#### 🎚️ 结果筛选")
    col1, col2, col3 = st.columns(3)

    with col1:
        min_score = st.slider("最低评分", 0, 100, 60, step=10)
    with col2:
        only_full_match = st.checkbox("仅显示完全匹配", value=False)
    with col3:
        sort_by = st.selectbox("排序方式", ["评分", "5日涨幅%", "60日跌幅%"])

    # 应用筛选
    filtered_df = results_df[results_df['评分'] >= min_score]
    if only_full_match:
        filtered_df = filtered_df[filtered_df['完全匹配'] == True]

    # 排序
    if sort_by == "评分":
        filtered_df = filtered_df.sort_values('评分', ascending=False)
    elif sort_by == "5日涨幅%":
        filtered_df = filtered_df.sort_values('5日涨幅%', ascending=False)
    else:
        filtered_df = filtered_df.sort_values('60日跌幅%', ascending=True)

    # 显示前N名
    display_df = filtered_df.head(top_n)

    st.markdown(f"#### 🏆 Top {min(top_n, len(display_df))} 触底反弹候选标的")

    # 格式化显示
    display_columns = ['代码', '名称', '评分', '现价', '60日跌幅%', '5日涨幅%',
                       '底部震荡%', '距底部', '5日上涨天数', '阶段1_下跌', '阶段2_触底', '阶段3_反弹']

    # 样式函数
    def highlight_score(val):
        if val >= 90:
            return 'background-color: #90EE90; color: black; font-weight: bold;'
        elif val >= 60:
            return 'background-color: #FFEB3B; color: black;'
        return ''

    def highlight_phase(val):
        if val == '✓':
            return 'color: green; font-weight: bold;'
        return 'color: red;'

    styled_df = display_df[display_columns].style.applymap(
        highlight_score, subset=['评分']
    ).applymap(
        highlight_phase, subset=['阶段1_下跌', '阶段2_触底', '阶段3_反弹']
    )

    st.dataframe(styled_df, use_container_width=True, height=400)

    # 联动跳转和可视化功能
    st.markdown("---")
    st.subheader("🔗 深度分析与可视化")

    stock_options = display_df['名称'].tolist()
    if stock_options:
        # 初始化 session_state
        if 'rebound_stock_select' not in st.session_state or st.session_state.rebound_stock_select not in stock_options:
            st.session_state.rebound_stock_select = stock_options[0]

        current_index = stock_options.index(st.session_state.rebound_stock_select)

        # 前后切换按钮的回调函数
        def go_prev():
            idx = stock_options.index(st.session_state.rebound_stock_select) if st.session_state.rebound_stock_select in stock_options else 0
            if idx > 0:
                st.session_state.rebound_stock_select = stock_options[idx - 1]

        def go_next():
            idx = stock_options.index(st.session_state.rebound_stock_select) if st.session_state.rebound_stock_select in stock_options else 0
            if idx < len(stock_options) - 1:
                st.session_state.rebound_stock_select = stock_options[idx + 1]

        # 布局：选择框 + 切换按钮 + 计数
        col_select, col_nav, col_count = st.columns([3, 1, 1])
        with col_select:
            selected_stock = st.selectbox(
                "选择标的进行深度分析",
                options=stock_options,
                key="rebound_stock_select",
                label_visibility="collapsed"
            )
        with col_nav:
            btn_prev, btn_next = st.columns(2)
            with btn_prev:
                st.button("◀", key="btn_prev_rebound", on_click=go_prev,
                          disabled=(current_index == 0),
                          help="上一个标的", use_container_width=True)
            with btn_next:
                st.button("▶", key="btn_next_rebound", on_click=go_next,
                          disabled=(current_index == len(stock_options) - 1),
                          help="下一个标的", use_container_width=True)
        with col_count:
            st.caption(f"第 {current_index + 1}/{len(stock_options)} 条")

        selected_row = display_df[display_df['名称'] == selected_stock]
        if not selected_row.empty:
            selected_code = selected_row.iloc[0]['代码']
            selected_name = selected_row.iloc[0]['名称']

            # 获取该股票的完整数据并绘制图表
            df_stock = read_local_kline(selected_code)
            if df_stock is not None and not df_stock.empty:
                pattern = BottomReboundPattern(custom_params)
                fig = plot_rebound_chart(selected_code, selected_name, df_stock, pattern)
                st.plotly_chart(fig, use_container_width=True)

            # 显示详细指标
            with st.expander(f"📊 {selected_name} 详细指标", expanded=True):
                row = selected_row.iloc[0]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("评分", f"{row['评分']}")
                    st.metric("现价", f"¥{row['现价']:.2f}")
                with col2:
                    st.metric("60日跌幅", f"{row['60日跌幅%']:.1f}%")
                    st.metric("20日涨跌", f"{row['20日涨跌%']:.1f}%")
                with col3:
                    st.metric("5日涨幅", f"{row['5日涨幅%']:.1f}%")
                    st.metric("5日上涨天数", f"{row['5日上涨天数']}天")
                with col4:
                    st.metric("底部震荡幅度", f"{row['底部震荡%']:.1f}%")
                    st.metric("量比", f"{row['量比']:.2f}")

                # 阶段判断说明
                st.markdown("**形态阶段判断：**")
                phase_status = []
                if row['阶段1_下跌'] == '✓':
                    phase_status.append("✅ 阶段1：前期下跌明显")
                else:
                    phase_status.append("❌ 阶段1：前期下跌不足")

                if row['阶段2_触底'] == '✓':
                    phase_status.append("✅ 阶段2：已形成底部")
                else:
                    phase_status.append("❌ 阶段2：底部尚未确认")

                if row['阶段3_反弹'] == '✓':
                    phase_status.append("✅ 阶段3：反弹已启动")
                else:
                    phase_status.append("❌ 阶段3：反弹尚未启动")

                for status in phase_status:
                    st.write(status)

            # 跳转按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📈 独立K线分析 - {selected_name}", use_container_width=True):
                    st.session_state.kline_jump_target = {
                        'code': selected_code,
                        'name': selected_name
                    }
                    st.switch_page("pages/07_kline_lab.py")
            with col2:
                if st.button(f"💡 AI资讯分析 - {selected_name}", use_container_width=True):
                    st.session_state.ai_news_jump_target = {
                        'code': selected_code,
                        'name': selected_name
                    }
                    st.switch_page("pages/01_ai_news.py")

            # 加入标的库功能
            st.markdown("---")
            st.markdown("**📥 加入本地标的库**")

            # 加载当前配置获取分类列表
            current_config = load_config()
            master_pool = current_config.get('master_pool', {})
            existing_categories = list(master_pool.keys()) if master_pool else ["核心组合", "科研/肾科", "其他"]

            # 检查是否已在标的库中
            existing_cat = None
            for cat, stocks in master_pool.items():
                if selected_code in stocks:
                    existing_cat = cat
                    break

            if existing_cat:
                st.info(f"✅ 该标的已在标的库 [{existing_cat}] 中")
            else:
                col_cat, col_add = st.columns([2, 1])
                with col_cat:
                    add_category = st.selectbox(
                        "选择分类",
                        options=existing_categories,
                        key=f"add_cat_rebound_{selected_code}"
                    )
                with col_add:
                    if st.button("➕ 加入标的库", type="primary", use_container_width=True, key=f"add_btn_rebound_{selected_code}"):
                        # 添加到标的库
                        if add_category not in master_pool:
                            master_pool[add_category] = {}
                        master_pool[add_category][selected_code] = selected_name
                        current_config['master_pool'] = master_pool
                        save_config(current_config)
                        # 同步更新 session_state
                        if 'master_pool' in st.session_state:
                            st.session_state.master_pool = master_pool
                        st.success(f"已将 {selected_name}({selected_code}) 添加到 [{add_category}]")
                        st.rerun()

    # 导出功能
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 导出全部结果 (CSV)",
            data=csv,
            file_name=f"触底反弹扫描_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        # 导出前N名
        top_csv = display_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label=f"📥 导出Top{top_n}结果 (CSV)",
            data=top_csv,
            file_name=f"触底反弹Top{top_n}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

elif 'rebound_scan_results' in st.session_state and st.session_state.rebound_scan_results.empty:
    st.warning("⚠️ 未找到符合条件的标的，请尝试调整参数后重新扫描")

else:
    # 首次进入页面，显示说明
    st.info("""
    **使用说明：**

    1. 确保市场数据库已初始化（上方显示股票数量）
    2. 可选：调整识别参数以适应当前市场环境
    3. 选择扫描范围（建议首次使用测试模式）
    4. 点击「开始扫描」按钮
    5. 等待扫描完成后查看结果

    **模型说明：**

    触底反弹形态识别基于三阶段模型：
    - **阶段1 - 前期下跌**：60日内跌幅超过20%
    - **阶段2 - 触底横盘**：20日内震荡幅度<10%，价格接近底部
    - **阶段3 - 反弹启动**：5日涨幅>2%，至少3天上涨

    评分规则：每个阶段30分，辅助指标各5分，满分100分
    """)

st.markdown("---")
st.caption("📝 **风险提示：本工具仅供研究参考，不构成投资建议。股市有风险，投资需谨慎。**")
