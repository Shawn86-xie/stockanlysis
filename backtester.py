import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import plotly.graph_objects as go

def generate_signal_series(prices: pd.Series, 
                           window: int = 14, 
                           ma_window: int = 60, 
                           rsi_buy: float = 35.0, 
                           rsi_sell: float = 75.0) -> pd.Series:
    """
    生成基于RSI和移动平均线的历史信号序列（向量化计算）
    
    参数:
        prices: 股票价格序列，索引为日期
        window: RSI计算窗口（默认14天）
        ma_window: 移动平均窗口（默认60天）
        rsi_buy: 买入RSI阈值（默认35，RSI低于此值且价格高于MA60时买入）
        rsi_sell: 卖出RSI阈值（默认75，RSI高于此值时卖出）
    
    返回:
        信号序列：1表示持有（买入/持有），0表示空仓（现金）
    """
    # 计算收益率
    returns = prices.pct_change()
    
    # 计算RSI
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    
    # 计算移动平均线
    ma = prices.rolling(window=ma_window).mean()
    
    # 生成信号：1=持有，0=空仓
    signal = pd.Series(1, index=prices.index)  # 默认持有
    
    # 当RSI低于买入阈值且价格高于MA时买入/持有
    buy_condition = (rsi < rsi_buy) & (prices > ma)
    # 当RSI高于卖出阈值时卖出/空仓
    sell_condition = (rsi > rsi_sell)
    
    # 应用信号逻辑
    # 注意：使用前向填充，避免信号频繁切换
    signal = np.where(buy_condition, 1, 0)
    signal = np.where(sell_condition, 0, signal)
    
    # 转换为Series并处理NaN
    signal_series = pd.Series(signal, index=prices.index).ffill().fillna(1)
    
    return signal_series


def run_vectorized_backtest(price_series: pd.Series, 
                           signal_series: pd.Series,
                           initial_capital: float = 100000.0,
                           commission: float = 0.0003,
                           tax_rate: float = 0.0005) -> Tuple[pd.DataFrame, Dict]:
    """
    执行向量化回测，避免循环提高效率
    
    参数:
        price_series: 价格序列
        signal_series: 信号序列（1=持有，0=空仓）
        initial_capital: 初始资金
        commission: 佣金费率（默认万分之三）
        tax_rate: 印花税费率（默认万分之五）
    
    返回:
        (result_df, metrics_dict)
        result_df包含：价格、信号、收益率、净值、最大回撤等
        metrics_dict包含关键绩效指标
    """
    # 确保索引对齐
    df = pd.DataFrame({
        'price': price_series,
        'signal': signal_series
    }).dropna()
    
    if len(df) == 0:
        return pd.DataFrame(), {}
    
    # 1. 计算对数收益率（更准确处理复利）
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    
    # 2. 信号平移：今天的信号决定明天的收益（避免未来函数）
    df['strategy_signal'] = df['signal'].shift(1).fillna(1)
    
    # 3. 计算交易动作：信号变化时产生交易
    df['signal_change'] = df['strategy_signal'].diff().abs()
    
    # 4. 计算策略收益率（考虑交易成本）
    # 佣金：每次买入或卖出时扣除
    commission_cost = df['signal_change'] * commission
    
    # 印花税：仅在卖出时扣除（A股规则）
    sell_signal = (df['strategy_signal'].diff() < 0).astype(int)
    tax_cost = sell_signal * tax_rate
    
    # 净收益率 = 策略收益率 - 交易成本
    df['strategy_return'] = df['strategy_signal'] * df['log_return']
    df['net_return'] = df['strategy_return'] - commission_cost - tax_cost
    
    # 5. 计算净值曲线
    df['cumulative_return'] = np.exp(df['net_return'].cumsum())
    df['equity'] = df['cumulative_return'] * initial_capital
    
    # 6. 计算最大回撤（向量化计算）
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    
    # 7. 计算交易统计
    trades = []
    position = 0
    entry_price = 0
    entry_date = None
    
    for i in range(1, len(df)):
        current_signal = df['strategy_signal'].iloc[i]
        prev_signal = df['strategy_signal'].iloc[i-1]
        
        # 开仓
        if prev_signal == 0 and current_signal == 1:
            entry_date = df.index[i]
            entry_price = df['price'].iloc[i]
            position = 1
        
        # 平仓
        elif prev_signal == 1 and current_signal == 0:
            exit_date = df.index[i]
            exit_price = df['price'].iloc[i]
            
            if position == 1 and entry_date:
                trade_return = (exit_price / entry_price) - 1
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return
                })
            
            position = 0
    
    # 8. 计算关键指标
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        win_trades = trades_df[trades_df['return'] > 0]
        
        win_rate = len(win_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = win_trades['return'].mean() if len(win_trades) > 0 else 0
        
        loss_trades = trades_df[trades_df['return'] <= 0]
        avg_loss = abs(loss_trades['return'].mean()) if len(loss_trades) > 0 else 0
        
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
        
        total_return = (df['equity'].iloc[-1] / initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1 if len(df) > 252 else total_return
    else:
        win_rate = 0
        profit_loss_ratio = 0
        total_return = (df['equity'].iloc[-1] / initial_capital) - 1
        annualized_return = total_return
    
    max_drawdown = df['drawdown'].min()
    sharpe_ratio = (df['net_return'].mean() * 252) / (df['net_return'].std() * np.sqrt(252)) if df['net_return'].std() > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'num_trades': len(trades) if 'trades' in locals() else 0,
        'final_equity': df['equity'].iloc[-1]
    }
    
    return df, metrics


def plot_backtest_results(price_series: pd.Series, 
                         result_df: pd.DataFrame,
                         metrics: Dict) -> go.Figure:
    """
    绘制回测结果图表
    """
    fig = go.Figure()
    
    # 价格曲线 (主Y轴)
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        name='价格',
        line=dict(color='lightgray', width=1),
        yaxis='y1'
    ))
    
    # 买卖信号标记 (与价格共享Y轴)
    buy_signals = result_df[result_df['signal_change'] == 1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=price_series.loc[buy_signals.index],
            mode='markers',
            name='买入',
            marker=dict(symbol='triangle-up', color='green', size=10),
            yaxis='y1'
        ))
    
    # 净值曲线 (右侧副Y轴)
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df['equity'],
        name='策略净值',
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))
    
    # 回撤区域 (独立于右侧)
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df['drawdown'] * 100,
        name='回撤 (%)',
        line=dict(color='red', width=1, dash='dot'),
        yaxis='y3'
    ))
    
    # 图表布局 - 使用domain分配空间，避免position错误
    title = f"回测结果 | 总收益: {metrics['total_return']:.2%} | 最大回撤: {metrics['max_drawdown']:.2%}"
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='日期',
            domain=[0.08, 0.92]  # 留出左右边距
        ),
        yaxis1=dict(
            title='价格',
            side='left',
            showgrid=False
        ),
        yaxis2=dict(
            title='策略净值',
            side='right',
            overlaying='y',
            anchor='x',
            showgrid=False
        ),
        yaxis3=dict(
            title='回撤 (%)',
            side='right',
            overlaying='y',
            anchor='free',
            position=0.95,  # 在有效范围内 [0,1]
            showgrid=False
        ),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500,
        margin=dict(l=50, r=100, t=80, b=50)
    )
    
    return fig


def plot_divergence_chart(df: pd.DataFrame, stock_name: str) -> go.Figure:
    """
    在 Plotly 图表中绘制价格曲线及背离预警信号（增强版）
    
    增强功能：
    1. 动态位移：基于ATR计算箭头偏移，避免遮挡价格线
    2. 连线功能：连接背离的两个极值点，直观展示"方向失配"
    3. 专业样式：医疗主题配色，科研级可视化
    
    参数:
        df: 包含 'close', 'bullish_divergence', 'bearish_divergence', 'price_peak', 'price_trough',
            'rsi_peak', 'rsi_trough' 列的 DataFrame
        stock_name: 股票名称
    
    返回:
        Plotly图表对象
    """
    fig = go.Figure()
    
    # 计算ATR（平均真实波幅）用于动态位移
    def calculate_atr(prices: pd.Series, window: int = 14) -> pd.Series:
        """计算ATR指标用于动态位移"""
        high = prices  # 使用收盘价作为近似
        low = prices
        close = prices
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    # 计算ATR（如果数据量足够）
    if len(df) >= 14:
        atr_series = calculate_atr(df['close'])
        # 使用最近ATR的平均值作为基准
        if not atr_series.isna().all():
            avg_atr = atr_series.dropna().iloc[-1]
        else:
            avg_atr = df['close'].std() * 0.05  # 备用方案
    else:
        avg_atr = df['close'].std() * 0.05 if len(df) > 1 else df['close'].iloc[0] * 0.02
    
    # 1. 绘制主价格曲线
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'],
        mode='lines',
        name='收盘价',
        line=dict(color='white', width=1.5),
        opacity=0.7,
        hovertemplate='<b>收盘价</b><br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
    ))

    # 2. 标记底背离 (Bullish Divergence) - 绿色向上箭头
    bullish_pts = df[df['bullish_divergence'] == True]
    if not bullish_pts.empty:
        # 动态位移：价格 - 1.5 * ATR
        y_offset = bullish_pts['close'] * 0.98  # 基础偏移
        if avg_atr > 0:
            # 使用ATR调整，确保不遮挡价格线
            y_offset = bullish_pts['close'] - 1.5 * avg_atr
        
        fig.add_trace(go.Scatter(
            x=bullish_pts.index,
            y=y_offset,
            mode='markers+text',
            name='底背离 (买入预警)',
            marker=dict(symbol='triangle-up', size=12, color='#00ff00', line=dict(width=2, color='darkgreen')),
            text="底背离",
            textposition="bottom center",
            hovertemplate='<b>底背离信号</b><br>日期: %{x}<br>价格: %{text:.2f}<br>强度: %{customdata:.1f}<extra></extra>',
            customdata=bullish_pts['divergence_strength'] if 'divergence_strength' in bullish_pts.columns else [0] * len(bullish_pts),
            texttemplate='%{text}'
        ))

    # 3. 标记顶背离 (Bearish Divergence) - 红色向下箭头
    bearish_pts = df[df['bearish_divergence'] == True]
    if not bearish_pts.empty:
        # 动态位移：价格 + 1.5 * ATR
        y_offset = bearish_pts['close'] * 1.02  # 基础偏移
        if avg_atr > 0:
            # 使用ATR调整，确保不遮挡价格线
            y_offset = bearish_pts['close'] + 1.5 * avg_atr
        
        fig.add_trace(go.Scatter(
            x=bearish_pts.index,
            y=y_offset,
            mode='markers+text',
            name='顶背离 (卖出预警)',
            marker=dict(symbol='triangle-down', size=12, color='#ff4444', line=dict(width=2, color='darkred')),
            text="顶背离",
            textposition="top center",
            hovertemplate='<b>顶背离信号</b><br>日期: %{x}<br>价格: %{text:.2f}<br>强度: %{customdata:.1f}<extra></extra>',
            customdata=bearish_pts['divergence_strength'] if 'divergence_strength' in bearish_pts.columns else [0] * len(bearish_pts),
            texttemplate='%{text}'
        ))

    # 4. 连线功能：连接背离的两个极值点
    # 检测背离极值点（价格峰值/谷值）
    price_peaks = df[df['price_peak'].notna()]
    price_troughs = df[df['price_trough'].notna()]
    
    # 连接顶背离的极值点（价格高点）
    if len(price_peaks) >= 2:
        for i in range(1, len(price_peaks)):
            if (price_peaks.index[i] - price_peaks.index[i-1]).days <= 60:
                # 检查是否是顶背离模式（价格上升，RSI下降）
                price_inc = price_peaks['price_peak'].iloc[i] > price_peaks['price_peak'].iloc[i-1]
                rsi_dec = price_peaks['rsi_peak'].iloc[i] < price_peaks['rsi_peak'].iloc[i-1] if 'rsi_peak' in price_peaks.columns else False
                
                if price_inc and rsi_dec:
                    # 添加连接线（虚线）
                    fig.add_trace(go.Scatter(
                        x=[price_peaks.index[i-1], price_peaks.index[i]],
                        y=[price_peaks['price_peak'].iloc[i-1], price_peaks['price_peak'].iloc[i]],
                        mode='lines',
                        name='顶背离连线',
                        line=dict(color='rgba(255, 68, 68, 0.6)', width=2, dash='dash'),
                        showlegend=False,
                        hovertemplate='<b>顶背离连线</b><br>连接两个价格高点<br>价格1: %{customdata[0]:.2f}<br>价格2: %{customdata[1]:.2f}<extra></extra>',
                        customdata=[[price_peaks['price_peak'].iloc[i-1], price_peaks['price_peak'].iloc[i]]]
                    ))
    
    # 连接底背离的极值点（价格低点）
    if len(price_troughs) >= 2:
        for i in range(1, len(price_troughs)):
            if (price_troughs.index[i] - price_troughs.index[i-1]).days <= 60:
                # 检查是否是底背离模式（价格下降，RSI上升）
                price_dec = price_troughs['price_trough'].iloc[i] < price_troughs['price_trough'].iloc[i-1]
                rsi_inc = price_troughs['rsi_trough'].iloc[i] > price_troughs['rsi_trough'].iloc[i-1] if 'rsi_trough' in price_troughs.columns else False
                
                if price_dec and rsi_inc:
                    # 添加连接线（虚线）
                    fig.add_trace(go.Scatter(
                        x=[price_troughs.index[i-1], price_troughs.index[i]],
                        y=[price_troughs['price_trough'].iloc[i-1], price_troughs['price_trough'].iloc[i]],
                        mode='lines',
                        name='底背离连线',
                        line=dict(color='rgba(0, 255, 0, 0.6)', width=2, dash='dash'),
                        showlegend=False,
                        hovertemplate='<b>底背离连线</b><br>连接两个价格低点<br>价格1: %{customdata[0]:.2f}<br>价格2: %{customdata[1]:.2f}<extra></extra>',
                        customdata=[[price_troughs['price_trough'].iloc[i-1], price_troughs['price_trough'].iloc[i]]]
                    ))

    # 5. 布局优化（医疗科研主题）
    fig.update_layout(
        title=dict(
            text=f"🏥 {stock_name} 动能背离深度分析 (科研级)",
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=dict(
            text="日期",
            font=dict(size=14, color='lightgray')
        ),
        yaxis_title=dict(
            text="价格 (元)",
            font=dict(size=14, color='lightgray')
        ),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='white')
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor='rgba(17, 17, 17, 0.9)',
        paper_bgcolor='rgba(17, 17, 17, 0.9)',
        height=550,
        showlegend=True
    )
    
    # 添加网格线
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 100, 0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 100, 0.2)')
    
    return fig


def calculate_hs300_alpha(backtest_returns: pd.Series,
                         hs300_returns: pd.Series,
                         risk_free_rate: float = 0.0188) -> float:
    """
    计算相对于沪深300的Alpha
    
    参数:
        backtest_returns: 策略收益率序列
        hs300_returns: 沪深300收益率序列（需要与策略同期）
        risk_free_rate: 无风险利率（默认1.88%）
    
    返回:
        Alpha值（超额收益）
    """
    # 对齐数据
    aligned_returns = pd.concat([backtest_returns, hs300_returns], axis=1).dropna()
    if len(aligned_returns) == 0:
        return 0.0
    
    strategy_excess = aligned_returns.iloc[:, 0] - risk_free_rate / 252
    hs300_excess = aligned_returns.iloc[:, 1] - risk_free_rate / 252
    
    # 计算Beta
    covariance = strategy_excess.cov(hs300_excess)
    hs300_variance = hs300_excess.var()
    
    beta = covariance / hs300_variance if hs300_variance > 0 else 1.0
    
    # 计算Alpha
    alpha = strategy_excess.mean() - beta * hs300_excess.mean()
    
    return alpha * 252  # 年化Alpha


def rsi_grid_search(data: pd.DataFrame,
                   windows: range,
                   buy_thresholds: range,
                   sell_thresholds: range) -> pd.DataFrame:
    """
    网格搜索 RSI 最优参数
    
    参数:
        data: 包含 'close' 列的 DataFrame，索引为日期
        windows: 周期列表，如 range(6, 21, 2)
        buy_thresholds: 买入阈值列表，如 range(20, 41, 5)
        sell_thresholds: 卖出阈值列表，如 range(60, 81, 5)
    
    返回:
        按总收益率排序的 DataFrame，包含每个参数组合的表现
    """
    results = []
    
    # 计算基础对数收益率以备后用
    log_returns = np.log(data['close'] / data['close'].shift(1))
    
    for w in windows:
        # 计算该周期下的 RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        for bt in buy_thresholds:
            for st in sell_thresholds:
                # 生成信号逻辑: 1=买入/持有，0=卖出/空仓
                # RSI低于买入阈值时买入，高于卖出阈值时卖出
                signal = np.where(rsi < bt, 1, np.where(rsi > st, 0, np.nan))
                # 填充持仓状态 (ffill)
                pos = pd.Series(signal, index=data.index).ffill().fillna(0)
                
                # 计算策略收益 (滞后一天避免未来函数)
                strat_ret = pos.shift(1) * log_returns
                total_ret = np.exp(strat_ret.sum()) - 1
                
                # 计算夏普比率 (简单版)
                if strat_ret.std() != 0 and len(strat_ret) > 1:
                    sharpe = np.sqrt(252) * (strat_ret.mean() / strat_ret.std())
                else:
                    sharpe = 0
                
                # 计算交易次数
                trades = (pos.diff().abs() > 0).sum()
                
                results.append({
                    'window': w,
                    'buy_threshold': bt,
                    'sell_threshold': st,
                    'total_return': total_ret,
                    'sharpe': sharpe,
                    'num_trades': trades
                })
    
    return pd.DataFrame(results).sort_values('total_return', ascending=False)


def grid_search_with_backtest(price_series: pd.Series,
                             windows: range,
                             buy_thresholds: range,
                             sell_thresholds: range,
                             initial_capital: float = 100000.0,
                             commission: float = 0.0003,
                             tax_rate: float = 0.0005) -> pd.DataFrame:
    """
    网格搜索结合完整回测，返回详细的性能指标
    
    参数:
        price_series: 价格序列
        windows: RSI周期范围
        buy_thresholds: 买入阈值范围
        sell_thresholds: 卖出阈值范围
        initial_capital: 初始资金
        commission: 佣金费率
        tax_rate: 印花税费率
    
    返回:
        包含详细回测指标的 DataFrame
    """
    results = []
    
    for w in windows:
        for bt in buy_thresholds:
            for st in sell_thresholds:
                # 生成信号序列
                signal_series = generate_signal_series(
                    price_series,
                    window=w,
                    ma_window=60,  # 固定MA60
                    rsi_buy=bt,
                    rsi_sell=st
                )
                
                # 运行完整回测
                result_df, metrics = run_vectorized_backtest(
                    price_series,
                    signal_series,
                    initial_capital=initial_capital,
                    commission=commission,
                    tax_rate=tax_rate
                )
                
                if not result_df.empty:
                    results.append({
                        'window': w,
                        'buy_threshold': bt,
                        'sell_threshold': st,
                        'total_return': metrics['total_return'],
                        'annualized_return': metrics['annualized_return'],
                        'max_drawdown': metrics['max_drawdown'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'win_rate': metrics['win_rate'],
                        'profit_loss_ratio': metrics['profit_loss_ratio'],
                        'num_trades': metrics['num_trades'],
                        'final_equity': metrics['final_equity']
                    })
    
    return pd.DataFrame(results).sort_values('sharpe_ratio', ascending=False)


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    计算RSI指标
    
    参数:
        prices: 价格序列
        window: RSI计算窗口（默认14天）
    
    返回:
        RSI序列
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    return rsi


def detect_rsi_divergence(df: pd.DataFrame, order: int = 5, max_days_between: int = 60) -> pd.DataFrame:
    """
    背离检测算法（不使用scipy）
    
    参数:
        df: 包含 'close' 和 'rsi' 列的 DataFrame
        order: 极值点寻找的窗口大小（越大越稳健，越小越灵敏）
        max_days_between: 两个极值点之间的最大天数限制
    
    返回:
        包含背离信号的 DataFrame
    """
    df = df.copy()
    
    # 确保有足够的RSI数据
    if 'rsi' not in df.columns:
        # 计算RSI（默认14天）
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        df['rsi'] = rsi
    
    # 使用滚动窗口寻找局部极值（不使用scipy）
    # 局部高点：左右各order天的价格都比它低
    # 局部低点：左右各order天的价格都比它高
    
    df['price_peak'] = np.nan
    df['price_trough'] = np.nan
    df['rsi_peak'] = np.nan
    df['rsi_trough'] = np.nan
    
    for i in range(order, len(df) - order):
        # 检查是否为价格高点
        current_price = df['close'].iloc[i]
        left_prices = df['close'].iloc[i-order:i]
        right_prices = df['close'].iloc[i+1:i+order+1]
        
        if current_price > left_prices.max() and current_price > right_prices.max():
            df.loc[df.index[i], 'price_peak'] = current_price
        
        # 检查是否为价格低点
        if current_price < left_prices.min() and current_price < right_prices.min():
            df.loc[df.index[i], 'price_trough'] = current_price
        
        # 检查是否为RSI高点
        current_rsi = df['rsi'].iloc[i]
        left_rsi = df['rsi'].iloc[i-order:i]
        right_rsi = df['rsi'].iloc[i+1:i+order+1]
        
        if current_rsi > left_rsi.max() and current_rsi > right_rsi.max():
            df.loc[df.index[i], 'rsi_peak'] = current_rsi
        
        # 检查是否为RSI低点
        if current_rsi < left_rsi.min() and current_rsi < right_rsi.min():
            df.loc[df.index[i], 'rsi_trough'] = current_rsi
    
    # 提取非空的极值点
    peaks = df.dropna(subset=['price_peak', 'rsi_peak']).copy()
    troughs = df.dropna(subset=['price_trough', 'rsi_trough']).copy()
    
    # 初始化背离列
    df['bearish_divergence'] = False
    df['bullish_divergence'] = False
    df['divergence_strength'] = 0.0
    
    # 检测顶背离 (Bearish Divergence)
    if len(peaks) >= 2:
        for i in range(1, len(peaks)):
            # 检查时间间隔
            time_diff = (peaks.index[i] - peaks.index[i-1]).days
            if 0 < time_diff <= max_days_between:
                price_inc = peaks['price_peak'].iloc[i] > peaks['price_peak'].iloc[i-1]
                rsi_dec = peaks['rsi_peak'].iloc[i] < peaks['rsi_peak'].iloc[i-1]
                
                if price_inc and rsi_dec:
                    # 计算背离强度（价格增长百分比 vs RSI下降百分比）
                    price_change = (peaks['price_peak'].iloc[i] / peaks['price_peak'].iloc[i-1] - 1) * 100
                    rsi_change = (peaks['rsi_peak'].iloc[i] / peaks['rsi_peak'].iloc[i-1] - 1) * 100
                    strength = abs(price_change) + abs(rsi_change)
                    
                    # 要求第一个RSI峰值处于超买区 (>70)
                    if peaks['rsi_peak'].iloc[i-1] > 70:
                        df.loc[peaks.index[i], 'bearish_divergence'] = True
                        df.loc[peaks.index[i], 'divergence_strength'] = strength
    
    # 检测底背离 (Bullish Divergence)
    if len(troughs) >= 2:
        for i in range(1, len(troughs)):
            # 检查时间间隔
            time_diff = (troughs.index[i] - troughs.index[i-1]).days
            if 0 < time_diff <= max_days_between:
                price_dec = troughs['price_trough'].iloc[i] < troughs['price_trough'].iloc[i-1]
                rsi_inc = troughs['rsi_trough'].iloc[i] > troughs['rsi_trough'].iloc[i-1]
                
                if price_dec and rsi_inc:
                    # 计算背离强度
                    price_change = (troughs['price_trough'].iloc[i] / troughs['price_trough'].iloc[i-1] - 1) * 100
                    rsi_change = (troughs['rsi_trough'].iloc[i] / troughs['rsi_trough'].iloc[i-1] - 1) * 100
                    strength = abs(price_change) + abs(rsi_change)
                    
                    # 要求第一个RSI谷值处于超卖区 (<30)
                    if troughs['rsi_trough'].iloc[i-1] < 30:
                        df.loc[troughs.index[i], 'bullish_divergence'] = True
                        df.loc[troughs.index[i], 'divergence_strength'] = strength
    
    return df


def plot_backtest_with_divergence(price_series: pd.Series, 
                                 result_df: pd.DataFrame,
                                 metrics: Dict,
                                 divergence_df: pd.DataFrame = None) -> go.Figure:
    """
    绘制回测结果图表（包含背离信号）
    """
    fig = go.Figure()
    
    # 价格曲线 (主Y轴)
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        name='价格',
        line=dict(color='lightgray', width=1),
        yaxis='y1'
    ))
    
    # 买卖信号标记 (与价格共享Y轴)
    buy_signals = result_df[result_df['signal_change'] == 1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=price_series.loc[buy_signals.index],
            mode='markers',
            name='买入',
            marker=dict(symbol='triangle-up', color='green', size=10),
            yaxis='y1'
        ))
    
    # 背离信号标记
    if divergence_df is not None:
        # 顶背离标记 (红色向下箭头)
        bearish_div = divergence_df[divergence_df['bearish_divergence']]
        if not bearish_div.empty:
            fig.add_trace(go.Scatter(
                x=bearish_div.index,
                y=price_series.loc[bearish_div.index],
                mode='markers',
                name='顶背离',
                marker=dict(symbol='arrow-down', color='red', size=15, line=dict(width=2, color='darkred')),
                yaxis='y1',
                hovertemplate='<b>顶背离信号</b><br>价格: %{y:.2f}<br>日期: %{x}<br>强度: %{text}',
                text=[f'{s:.1f}' for s in bearish_div['divergence_strength']]
            ))
        
        # 底背离标记 (绿色向上箭头)
        bullish_div = divergence_df[divergence_df['bullish_divergence']]
        if not bullish_div.empty:
            fig.add_trace(go.Scatter(
                x=bullish_div.index,
                y=price_series.loc[bullish_div.index],
                mode='markers',
                name='底背离',
                marker=dict(symbol='arrow-up', color='green', size=15, line=dict(width=2, color='darkgreen')),
                yaxis='y1',
                hovertemplate='<b>底背离信号</b><br>价格: %{y:.2f}<br>日期: %{x}<br>强度: %{text}',
                text=[f'{s:.1f}' for s in bullish_div['divergence_strength']]
            ))
    
    # 净值曲线 (右侧副Y轴)
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df['equity'],
        name='策略净值',
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))
    
    # 回撤区域 (独立于右侧)
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df['drawdown'] * 100,
        name='回撤 (%)',
        line=dict(color='red', width=1, dash='dot'),
        yaxis='y3'
    ))
    
    # 图表布局 - 使用domain分配空间，避免position错误
    title = f"回测结果 | 总收益: {metrics['total_return']:.2%} | 最大回撤: {metrics['max_drawdown']:.2%}"
    if divergence_df is not None:
        total_div = len(bearish_div) + len(bullish_div)
        if total_div > 0:
            title += f" | 背离信号: {total_div}个"
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='日期',
            domain=[0.08, 0.92]  # 留出左右边距
        ),
        yaxis1=dict(
            title='价格',
            side='left',
            showgrid=False
        ),
        yaxis2=dict(
            title='策略净值',
            side='right',
            overlaying='y',
            anchor='x',
            showgrid=False
        ),
        yaxis3=dict(
            title='回撤 (%)',
            side='right',
            overlaying='y',
            anchor='free',
            position=0.95,  # 在有效范围内 [0,1]
            showgrid=False
        ),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500,
        margin=dict(l=50, r=100, t=80, b=50)
    )
    
    return fig
