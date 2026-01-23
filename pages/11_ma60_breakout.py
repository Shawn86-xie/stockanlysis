"""
MA60 均线生命线突破挖掘页面

功能：
1. 基于 MA60 均线生命线突破模型扫描全市场标的
2. 识别 K 线由下向上穿越 MA60 并连续 5 日站稳的标的
3. 结合趋势斜率、乖离率、成交量动能进行综合评分
4. 提供可视化图表和深度分析
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# 复用现有服务模块
from config import load_config, save_config
from services.market_stock_list import (
    load_market_stock_list,
    get_market_stock_list_info,
    refresh_market_stock_list
)
from services.kline_database import (
    get_kline_db_stats,
    get_cached_stock_codes,
    batch_download_kline,
    incremental_update_kline,
    read_local_kline
)

# 页面配置
st.set_page_config(page_title="MA60趋势突破挖掘", layout="wide")


class MA60BreakoutPattern:
    """
    MA60 均线生命线突破模型
    核心：捕捉K线由下向上穿越60日均线并稳固5日的转折点
    """

    def __init__(self, params=None):
        """
        初始化形态参数

        Args:
            params: 可选的参数字典，用于覆盖默认参数
        """
        # 核心参数
        self.ma_period = 60               # 核心均线周期
        self.confirm_days = 5             # 突破后的确认周期
        self.slope_min = 0.0              # 均线斜率阈值（必须走平或上行，%）
        self.bias_max = 12.0              # 最大乖离率（防止追高，%）
        self.vol_z_min = 1.0              # 成交量动能要求（Z-Score）

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

        # 计算 MA60
        df['ma60'] = df['收盘'].rolling(window=self.ma_period).mean()

        # 向前填充缺失的 MA60 值（防止计算斜率时因 NaN 中断）
        df['ma60'] = df['ma60'].ffill()

        # 穿越判定
        df['is_above'] = df['收盘'] > df['ma60']

        # 5日稳固判定 (滚动求和，5天都站稳则和为5)
        df['stable_count'] = df['is_above'].rolling(window=self.confirm_days).sum()

        # 趋势斜率 (Slope) - 过去5日MA60的变化率
        df['ma60_slope'] = (df['ma60'] - df['ma60'].shift(5)) / df['ma60'].shift(5) * 100

        # 乖离率 (Bias)
        df['bias'] = (df['收盘'] - df['ma60']) / df['ma60'] * 100

        # 成交量动能 (Z-Score)
        v_m = df['成交量'].rolling(20).mean()
        v_s = df['成交量'].rolling(20).std()
        df['vol_z'] = (df['成交量'] - v_m) / (v_s + 1e-6)

        # 辅助指标：5日涨幅（用于兼容性）
        df['return_5d'] = df['收盘'].pct_change(5) * 100

        # 回抽不破检查：确认期内最低价是否在 MA60 之上
        df['low_5d'] = df['收盘'].rolling(5).min()
        df['retest_ok'] = df['low_5d'] > df['ma60']

        return df

    def check_pattern(self, df):
        """
        检查是否符合 MA60 突破形态

        Args:
            df: 已计算指标的 DataFrame

        Returns:
            tuple: (是否匹配, 详情字典)
        """
        if len(df) < 70:
            return False, {'score': 0, 'reason': '数据不足70天'}

        latest = df.iloc[-1]

        # 检查数据有效性
        required_cols = ['ma60', 'stable_count', 'ma60_slope', 'bias', 'vol_z']
        for col in required_cols:
            if pd.isna(latest[col]):
                return False, {'score': 0, 'reason': f'{col}数据无效'}

        # 条件1：5日确认穿越 (今日是第5天站稳，且5天前在均线之下)
        # 注意：stable_count 是滚动窗口内 is_above 的计数，当 stable_count == 5 时，表示最近5天都在均线上
        # 但还需要检查5天前是否在均线之下，以确保是首次突破
        confirm_breakout = (latest['stable_count'] == self.confirm_days)

        # 检查5天前是否在均线之下（如果数据足够）
        if len(df) >= self.confirm_days + 1:
            five_days_ago = df.iloc[-self.confirm_days - 1]
            confirm_breakout = confirm_breakout and (not five_days_ago['is_above'])

        # 条件2：斜率走平或向上
        slope_ok = latest['ma60_slope'] >= self.slope_min

        # 条件3：乖离率适中（不追高）
        bias_ok = latest['bias'] <= self.bias_max

        # 条件4：成交量动能
        vol_ok = latest['vol_z'] >= self.vol_z_min

        # 辅助条件：回抽不破（增强稳定性）
        retest_ok = latest.get('retest_ok', False)

        # 综合判断
        is_match = confirm_breakout and slope_ok and bias_ok and vol_ok

        # 强度评分（越高越符合）
        score = 0
        if confirm_breakout: score += 60
        if slope_ok: score += 20
        if bias_ok: score += 10
        if vol_ok: score += 10
        if retest_ok: score += 10  # 额外加分

        details = {
            'score': score,
            'slope': round(latest['ma60_slope'], 2),
            'bias': round(latest['bias'], 2),
            'vol_z': round(latest['vol_z'], 2),
            'ma60': round(latest['ma60'], 2),
            'stable_count': int(latest['stable_count']),
            'retest_ok': retest_ok,
            'current_price': round(latest['收盘'], 2),
            'is_match': is_match,
            # 以下字段用于兼容 scan_with_local_db（如必须）
            'return_60d': 0,
            'return_20d': 0,
            'return_5d': round(latest.get('return_5d', 0), 2),
            'bottom_range': 0,
            'price_to_low': 0,
            'up_days_5': 0,
            'volume_ratio': 0,
            'vol_ratio': 0,
            'phase1': False,
            'phase2': False,
            'phase3': False,
            'reason': ''
        }

        return is_match, details

    def find_historical_breakouts(self, df, min_interval=10):
        """
        扫描历史数据，找出所有符合 MA60 突破条件的时间点
        判断逻辑与 check_pattern 完全一致

        Args:
            df: 已计算指标的 DataFrame
            min_interval: 两次突破之间的最小间隔天数（避免重复标记）

        Returns:
            list: 历史突破点列表，每个元素为 {'date': 日期, 'price': 价格, 'score': 评分}
        """
        if len(df) < 70:
            return []

        breakouts = []
        last_breakout_idx = -min_interval  # 上一次突破的索引

        # 从第70天开始扫描（需要足够的历史数据计算指标）
        for i in range(70, len(df)):
            # 检查是否距离上次突破足够远
            if i - last_breakout_idx < min_interval:
                continue

            row = df.iloc[i]

            # 检查数据有效性（与 check_pattern 一致）
            if pd.isna(row['stable_count']) or pd.isna(row['ma60_slope']) or pd.isna(row['bias']) or pd.isna(row['vol_z']):
                continue

            # 条件1：5日确认穿越 (stable_count == 5 且 5天前在均线之下)
            confirm_breakout = (row['stable_count'] == self.confirm_days)

            # 检查5天前是否在均线之下
            if i >= self.confirm_days:
                five_days_ago = df.iloc[i - self.confirm_days]
                if pd.isna(five_days_ago['is_above']):
                    continue
                confirm_breakout = confirm_breakout and (not five_days_ago['is_above'])

            # 条件2：斜率走平或向上
            slope_ok = row['ma60_slope'] >= self.slope_min

            # 条件3：乖离率适中（不追高）
            bias_ok = row['bias'] <= self.bias_max

            # 条件4：成交量动能
            vol_ok = row['vol_z'] >= self.vol_z_min

            # 辅助条件：回抽不破
            retest_ok = row.get('retest_ok', False) if 'retest_ok' in row.index else False

            # 综合判断 - 必须全部满足（与 check_pattern 逻辑完全一致）
            is_match = confirm_breakout and slope_ok and bias_ok and vol_ok

            if is_match:
                # 计算评分（与 check_pattern 一致）
                score = 0
                if confirm_breakout: score += 60
                if slope_ok: score += 20
                if bias_ok: score += 10
                if vol_ok: score += 10
                if retest_ok: score += 10  # 额外加分

                breakouts.append({
                    'date': row['日期'],
                    'price': row['收盘'],
                    'ma60': row['ma60'],
                    'score': score,
                    'index': i,
                    'retest_ok': retest_ok
                })
                last_breakout_idx = i

        return breakouts


def filter_valid_stocks(stock_list):
    """
    过滤掉ST、退市、科创板等股票（与参考页面保持一致）

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

        filtered.append(stock)

    return filtered


def scan_ma60_with_local_db(stock_list, pattern, progress_callback=None):
    """
    使用本地数据库进行 MA60 突破扫描（自定义快速扫描）

    Args:
        stock_list: 股票列表
        pattern: MA60BreakoutPattern 实例
        progress_callback: 进度回调函数

    Returns:
        tuple: (results_df, stats)
    """
    import time

    results = []
    total = len(stock_list)

    success_count = 0
    skip_count = 0
    match_count = 0

    start_time = time.time()

    for i, stock in enumerate(stock_list):
        code = stock['code']
        name = stock['name']

        # 更新进度
        if progress_callback:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            progress_callback(
                i + 1, total, success_count, skip_count, match_count,
                name, speed, eta
            )

        # 从本地读取
        df = read_local_kline(code)

        if df is None or df.empty or len(df) < 70:
            skip_count += 1
            continue

        try:
            # 计算指标
            df = pattern.calculate_indicators(df)

            # 检查形态
            is_match, details = pattern.check_pattern(df)

            success_count += 1

            if details['score'] > 0:
                match_count += 1
                results.append({
                    '代码': code,
                    '名称': name,
                    '评分': details['score'],
                    'MA60': details['ma60'],
                    '现价': details['current_price'],
                    '乖离率%': details['bias'],
                    '斜率%': details['slope'],
                    '量能Z': details['vol_z'],
                    '稳固天数': details['stable_count'],
                    '回抽不破': '✓' if details['retest_ok'] else '✗',
                    '5日涨幅%': details['return_5d'],
                    '完全匹配': is_match
                })
        except Exception as e:
            skip_count += 1
            continue

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('评分', ascending=False)

    stats = {
        'total': total,
        'success': success_count,
        'skipped': skip_count,
        'matched': match_count,
        'duration': time.time() - start_time
    }

    return results_df, stats


def plot_ma60_chart(code, name, df, lookback_days=250, historical_breakouts=None):
    """
    绘制 MA60 突破图表

    Args:
        code: 股票代码
        name: 股票名称
        df: K线数据（需包含指标）
        lookback_days: 显示最近多少天的数据
        historical_breakouts: 历史突破点列表

    Returns:
        plotly.graph_objects.Figure
    """
    # 截取最近数据
    df_plot = df.tail(lookback_days).copy()
    df_plot = df_plot.reset_index(drop=True)

    # 获取显示范围内的日期范围
    plot_start_date = df_plot['日期'].iloc[0]
    plot_end_date = df_plot['日期'].iloc[-1]

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

    # 添加 MA60 均线
    fig.add_trace(go.Scatter(
        x=df_plot['日期'],
        y=df_plot['ma60'],
        mode='lines',
        name='MA60',
        line=dict(color='gold', width=3)
    ))

    # 标注历史突破点（"历史发生"区域）
    if historical_breakouts:
        # 过滤出在显示范围内的历史突破点（排除最近5天，因为那是当前确认区）
        cutoff_idx = len(df_plot) - 5 if len(df_plot) >= 5 else 0

        for i, bp in enumerate(historical_breakouts):
            bp_date = bp['date']

            # 检查是否在显示范围内
            if bp_date < plot_start_date or bp_date > plot_end_date:
                continue

            # 找到该日期在 df_plot 中的位置
            date_mask = df_plot['日期'] == bp_date
            if not date_mask.any():
                continue

            bp_idx = df_plot[date_mask].index[0]

            # 排除最近5天（当前确认区）
            if bp_idx >= cutoff_idx:
                continue

            # 获取该突破点前后5天的数据来绘制区域
            start_idx = max(0, bp_idx - 2)
            end_idx = min(len(df_plot) - 1, bp_idx + 2)

            region_data = df_plot.iloc[start_idx:end_idx + 1]
            if len(region_data) < 2:
                continue

            region_dates = region_data['日期']
            region_high = region_data['最高'].max()
            region_low = region_data['最低'].min()

            # 绘制历史突破区域（使用浅蓝色区分当前确认区）
            fig.add_trace(go.Scatter(
                x=[region_dates.iloc[0], region_dates.iloc[0], region_dates.iloc[-1], region_dates.iloc[-1], region_dates.iloc[0]],
                y=[region_low, region_high, region_high, region_low, region_low],
                fill='toself',
                fillcolor='rgba(100, 149, 237, 0.25)',  # 浅蓝色
                line=dict(color='rgba(100, 149, 237, 0.6)', width=1),
                name=f'历史发生' if i == 0 else None,  # 只有第一个显示图例
                showlegend=(i == 0),
                hoverinfo='text',
                hovertext=f"历史突破: {bp_date}<br>价格: {bp['price']:.2f}<br>评分: {bp['score']}"
            ))

            # 在区域中心添加标记点
            fig.add_trace(go.Scatter(
                x=[bp_date],
                y=[bp['price']],
                mode='markers',
                marker=dict(
                    color='cornflowerblue',
                    size=8,
                    symbol='diamond',
                    line=dict(color='white', width=1)
                ),
                showlegend=False,
                hoverinfo='text',
                hovertext=f"历史突破: {bp_date}<br>价格: {bp['price']:.2f}<br>评分: {bp['score']}"
            ))

    # 高亮突破确认区（最近5天）- 当前信号
    if len(df_plot) >= 5:
        last_5_dates = df_plot['日期'].tail(5)
        last_5_high = df_plot['最高'].tail(5).max()
        last_5_low = df_plot['最低'].tail(5).min()

        fig.add_trace(go.Scatter(
            x=[last_5_dates.iloc[0], last_5_dates.iloc[0], last_5_dates.iloc[-1], last_5_dates.iloc[-1], last_5_dates.iloc[0]],
            y=[last_5_low, last_5_high, last_5_high, last_5_low, last_5_low],
            fill='toself',
            fillcolor='rgba(255, 215, 0, 0.2)',
            line=dict(color='rgba(255, 215, 0, 0.5)', width=1),
            name='5日确认区(当前)',
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

    # 统计历史突破次数
    history_count = 0
    if historical_breakouts:
        cutoff_idx = len(df_plot) - 5 if len(df_plot) >= 5 else 0
        for bp in historical_breakouts:
            if bp['date'] >= plot_start_date and bp['date'] <= plot_end_date:
                date_mask = df_plot['日期'] == bp['date']
                if date_mask.any():
                    bp_idx = df_plot[date_mask].index[0]
                    if bp_idx < cutoff_idx:
                        history_count += 1

    # 布局设置
    title_suffix = f" | 历史发生: {history_count}次" if history_count > 0 else ""
    fig.update_layout(
        title=f'{name} ({code}) - MA60 突破分析{title_suffix}',
        xaxis_title='日期',
        yaxis_title='价格',
        template='plotly_white',
        height=500,
        xaxis_rangeslider_visible=False
    )

    return fig


# ==================== 页面主体 ====================

st.title("🧬 MA60 均线生命线突破挖掘")
st.caption("基于「5日确认穿越 + 趋势斜率 + 量能动能」模型，寻找中期趋势向上的标的")

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
        max_workers = st.slider("并行线程数", 1, 10, 5, key="kline_max_workers",
                                help="线程数越多越快，但可能触发API限流")
    with col2:
        skip_existing = st.checkbox("跳过已有数据", value=False, key="kline_skip_existing",
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
        workers = st.session_state.get('kline_max_workers', 5)
        skip = st.session_state.get('kline_skip_existing', False) if mode == 'full' else False

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
if 'ma60_params' not in st.session_state:
    st.session_state.ma60_params = {
        'slope_min': 0.0,
        'bias_max': 12.0,
        'vol_z_min': 1.0
    }

with st.expander("⚙️ 调整识别参数（可选）", expanded=False):
    st.markdown("根据市场环境调整识别灵敏度，参数越宽松匹配越多但精度下降")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**趋势斜率**")
        st.session_state.ma60_params['slope_min'] = st.slider(
            "MA60最低斜率 (%)",
            min_value=-0.5, max_value=2.0,
            value=st.session_state.ma60_params['slope_min'],
            step=0.1,
            help="MA60过去5日的斜率，正值表示上升趋势"
        )

    with col2:
        st.markdown("**乖离限制**")
        st.session_state.ma60_params['bias_max'] = st.slider(
            "最高乖离率 (%)",
            min_value=5.0, max_value=20.0,
            value=st.session_state.ma60_params['bias_max'],
            step=0.5,
            help="当前价格距离MA60的最大距离，防止追高"
        )

    with col3:
        st.markdown("**量能动能**")
        st.session_state.ma60_params['vol_z_min'] = st.slider(
            "成交量 Z-Score 阈值",
            min_value=0.0, max_value=3.0,
            value=st.session_state.ma60_params['vol_z_min'],
            step=0.1,
            help="成交量相对于20日均值的标准差倍数，越大表示放量越明显"
        )

# 从 session_state 构建参数字典
custom_params = {
    'slope_min': st.session_state.ma60_params['slope_min'],
    'bias_max': st.session_state.ma60_params['bias_max'],
    'vol_z_min': st.session_state.ma60_params['vol_z_min']
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
        # 在线模式暂不支持（需要实现网络获取）
        st.error("在线扫描模式暂未实现，请使用快速扫描（本地数据）模式")
        st.stop()

    # 创建进度显示组件
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_container = st.empty()

    # 创建形态识别器
    pattern = MA60BreakoutPattern(custom_params)

    # 开始扫描
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

    results_df, scan_stats = scan_ma60_with_local_db(
        stock_list,
        pattern,
        progress_callback=fast_progress_callback
    )
    duration = scan_stats['duration']

    progress_bar.progress(1.0)
    status_text.text(f"✅ 扫描完成！总耗时 {duration:.1f} 秒 ({duration/60:.1f} 分钟)")

    # 保存结果到 session_state
    st.session_state.ma60_scan_results = results_df
    st.session_state.ma60_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==================== 结果展示 ====================
if 'ma60_scan_results' in st.session_state and not st.session_state.ma60_scan_results.empty:
    results_df = st.session_state.ma60_scan_results
    scan_time = st.session_state.get('ma60_scan_time', '未知')

    st.markdown("---")
    st.subheader(f"📈 扫描结果（扫描时间：{scan_time}）")

    # 统计信息
    total_found = len(results_df)
    full_match = results_df['完全匹配'].sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("发现标的", f"{total_found} 只")
    with col2:
        st.metric("完全匹配", f"{full_match} 只", help="同时满足所有条件")
    with col3:
        avg_score = results_df['评分'].mean()
        st.metric("平均评分", f"{avg_score:.1f}")

    # 筛选条件
    st.markdown("#### 🎚️ 结果筛选")
    col1, col2, col3 = st.columns(3)

    with col1:
        min_score = st.slider("最低评分", 0, 110, 70, step=10)
    with col2:
        only_full_match = st.checkbox("仅显示完全匹配", value=False)
    with col3:
        sort_by = st.selectbox("排序方式", ["评分", "乖离率%", "斜率%", "5日涨幅%"])

    # 应用筛选
    filtered_df = results_df[results_df['评分'] >= min_score]
    if only_full_match:
        filtered_df = filtered_df[filtered_df['完全匹配'] == True]

    # 排序
    if sort_by == "评分":
        filtered_df = filtered_df.sort_values('评分', ascending=False)
    elif sort_by == "乖离率%":
        filtered_df = filtered_df.sort_values('乖离率%', ascending=False)
    elif sort_by == "斜率%":
        filtered_df = filtered_df.sort_values('斜率%', ascending=False)
    else:
        filtered_df = filtered_df.sort_values('5日涨幅%', ascending=False)

    # 显示前N名
    display_df = filtered_df.head(top_n)

    st.markdown(f"#### 🏆 Top {min(top_n, len(display_df))} MA60突破候选标的")

    # 格式化显示
    display_columns = ['代码', '名称', '评分', '现价', 'MA60', '乖离率%',
                       '斜率%', '量能Z', '稳固天数', '回抽不破', '5日涨幅%']

    # 样式函数
    def highlight_score(val):
        if val >= 90:
            return 'background-color: #90EE90; color: black; font-weight: bold;'
        elif val >= 70:
            return 'background-color: #FFEB3B; color: black;'
        return ''

    def highlight_phase(val):
        if val == '✓':
            return 'color: green; font-weight: bold;'
        return 'color: red;'

    styled_df = display_df[display_columns].style.applymap(
        highlight_score, subset=['评分']
    ).applymap(
        highlight_phase, subset=['回抽不破']
    )

    st.dataframe(styled_df, use_container_width=True, height=400)

    # 联动跳转和可视化功能
    st.markdown("---")
    st.subheader("🔗 深度分析与可视化")

    stock_options = display_df['名称'].tolist()
    if stock_options:
        # 初始化 session_state
        if 'ma60_stock_select' not in st.session_state or st.session_state.ma60_stock_select not in stock_options:
            st.session_state.ma60_stock_select = stock_options[0]

        current_index = stock_options.index(st.session_state.ma60_stock_select)

        # 前后切换按钮的回调函数
        def go_prev():
            idx = stock_options.index(st.session_state.ma60_stock_select) if st.session_state.ma60_stock_select in stock_options else 0
            if idx > 0:
                st.session_state.ma60_stock_select = stock_options[idx - 1]

        def go_next():
            idx = stock_options.index(st.session_state.ma60_stock_select) if st.session_state.ma60_stock_select in stock_options else 0
            if idx < len(stock_options) - 1:
                st.session_state.ma60_stock_select = stock_options[idx + 1]

        # 布局：选择框 + 切换按钮 + 计数
        col_select, col_nav, col_count = st.columns([3, 1, 1])
        with col_select:
            selected_name = st.selectbox(
                "选择标的进行深度分析",
                options=stock_options,
                key="ma60_stock_select",
                label_visibility="collapsed"
            )
        with col_nav:
            btn_prev, btn_next = st.columns(2)
            with btn_prev:
                st.button("◀", key="btn_prev_ma60", on_click=go_prev,
                          disabled=(current_index == 0),
                          help="上一个标的", use_container_width=True)
            with btn_next:
                st.button("▶", key="btn_next_ma60", on_click=go_next,
                          disabled=(current_index == len(stock_options) - 1),
                          help="下一个标的", use_container_width=True)
        with col_count:
            st.caption(f"第 {current_index + 1}/{len(stock_options)} 条")

        selected_row = display_df[display_df['名称'] == selected_name]
        if not selected_row.empty:
            selected_code = selected_row.iloc[0]['代码']
            selected_name = selected_row.iloc[0]['名称']

            # 获取该股票的完整数据并计算指标
            df_stock = read_local_kline(selected_code)
            if df_stock is not None and not df_stock.empty:
                pattern = MA60BreakoutPattern(custom_params)
                df_with_indicators = pattern.calculate_indicators(df_stock)

                # 查找历史突破点
                historical_breakouts = pattern.find_historical_breakouts(df_with_indicators)

                # 绘制图表（包含历史突破标注）
                fig = plot_ma60_chart(selected_code, selected_name, df_with_indicators,
                                      historical_breakouts=historical_breakouts)
                st.plotly_chart(fig, use_container_width=True)

                # 显示历史突破统计
                if historical_breakouts:
                    with st.expander(f"📜 历史突破记录（共 {len(historical_breakouts)} 次）", expanded=False):
                        hist_df = pd.DataFrame(historical_breakouts)
                        hist_df = hist_df[['date', 'price', 'ma60', 'score']].copy()
                        hist_df.columns = ['突破日期', '突破价格', 'MA60价格', '评分']
                        hist_df['突破价格'] = hist_df['突破价格'].apply(lambda x: f"¥{x:.2f}")
                        hist_df['MA60价格'] = hist_df['MA60价格'].apply(lambda x: f"¥{x:.2f}")
                        st.dataframe(hist_df, use_container_width=True, hide_index=True)

                # 显示详细指标
                with st.expander(f"📊 {selected_name} 详细指标", expanded=True):
                    row = selected_row.iloc[0]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("评分", f"{row['评分']}")
                        st.metric("现价", f"¥{row['现价']:.2f}")
                    with col2:
                        st.metric("MA60", f"¥{row['MA60']:.2f}")
                        st.metric("乖离率", f"{row['乖离率%']:.2f}%")
                    with col3:
                        st.metric("斜率", f"{row['斜率%']:.2f}%")
                        st.metric("量能Z", f"{row['量能Z']:.2f}")
                    with col4:
                        st.metric("稳固天数", f"{row['稳固天数']}天")
                        st.metric("5日涨幅", f"{row['5日涨幅%']:.2f}%")

                    # 形态判断说明
                    st.markdown("**形态判断：**")
                    conditions = []
                    if row['稳固天数'] >= 5:
                        conditions.append("✅ 5日确认穿越：连续5日站稳MA60上方")
                    else:
                        conditions.append(f"❌ 稳固天数不足：仅{row['稳固天数']}天")

                    if row['斜率%'] >= custom_params['slope_min']:
                        conditions.append(f"✅ 趋势斜率：{row['斜率%']:.2f}% >= {custom_params['slope_min']}%")
                    else:
                        conditions.append(f"❌ 趋势斜率不足：{row['斜率%']:.2f}% < {custom_params['slope_min']}%")

                    if row['乖离率%'] <= custom_params['bias_max']:
                        conditions.append(f"✅ 乖离适中：{row['乖离率%']:.2f}% <= {custom_params['bias_max']}%")
                    else:
                        conditions.append(f"❌ 乖离过高：{row['乖离率%']:.2f}% > {custom_params['bias_max']}%")

                    if row['量能Z'] >= custom_params['vol_z_min']:
                        conditions.append(f"✅ 量能充足：Z={row['量能Z']:.2f} >= {custom_params['vol_z_min']}")
                    else:
                        conditions.append(f"❌ 量能不足：Z={row['量能Z']:.2f} < {custom_params['vol_z_min']}")

                    if row['回抽不破'] == '✓':
                        conditions.append("✅ 回抽不破：确认期内最低价均在MA60之上")
                    else:
                        conditions.append("❌ 回抽不破：确认期内曾跌破MA60")

                    for condition in conditions:
                        st.write(condition)

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
                            key=f"add_cat_ma60_{selected_code}"
                        )
                    with col_add:
                        if st.button("➕ 加入标的库", type="primary", use_container_width=True, key=f"add_btn_ma60_{selected_code}"):
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
            file_name=f"MA60突破扫描_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        # 导出前N名
        top_csv = display_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label=f"📥 导出Top{top_n}结果 (CSV)",
            data=top_csv,
            file_name=f"MA60突破Top{top_n}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

elif 'ma60_scan_results' in st.session_state and st.session_state.ma60_scan_results.empty:
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

    MA60突破形态识别基于四维模型：
    - **5日确认穿越**：连续5个交易日收盘价在MA60之上，且5日前在MA60之下
    - **趋势斜率**：MA60均线走平或向上（斜率≥0%）
    - **乖离控制**：当前价格距离MA60不超过12%（防止追高）
    - **量能动能**：成交量Z-Score≥1（放量突破）

    **增强稳定性：**
    - **回抽不破**：确认期内最低价均在MA60之上，说明均线已从阻力转变为支撑
    - **评分规则**：确认穿越60分 + 斜率20分 + 乖离10分 + 量能10分 + 回抽不破10分 = 满分110分
    """)

st.markdown("---")
st.caption("📝 **风险提示：本工具仅供研究参考，不构成投资建议。股市有风险，投资需谨慎。**")
