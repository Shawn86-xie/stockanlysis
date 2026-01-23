"""
行情数据服务模块：负责所有行情数据获取和处理的逻辑
将 data_fetcher.py 中的核心数据获取函数抽离到此服务层
"""
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import time
import streamlit as st

# 导入本地市场标的库模块
from services.market_stock_list import (
    safe_search_local_stock,
    get_market_stock_list_info,
    refresh_market_stock_list,
    init_market_stock_list_if_needed
)


class MarketDataServiceError(Exception):
    """行情数据服务异常"""
    pass


def fetch_stock_data(codes, names):
    """
    获取股票历史数据（优化版：优先从本地Parquet读取）
    
    为什么要使用Parquet + session_state：
    1. Parquet是列式存储，读取速度快，压缩率高，适合时间序列数据
    2. 本地存储避免了每次联网请求，大幅提升加载速度
    3. session_state可以跨rerun缓存数据，避免重复I/O
    4. 首次使用才从网络拉取，后续使用秒级读取
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
    
    Returns:
        DataFrame，列为股票名称，索引为日期
    
    Raises:
        MarketDataServiceError: 当数据获取失败时抛出
    """
    try:
        # 导入data_fetcher中的fetch_stock_data（已优化版本）
        from data_fetcher import fetch_stock_data as fetch_stock_data_optimized
        return fetch_stock_data_optimized(codes, names, incremental_update=False)
    except Exception as e:
        raise MarketDataServiceError(f"获取股票数据失败: {e}")


def fetch_candle_data(code, days=365):
    """
    获取股票的OHLC数据（开盘、最高、最低、收盘、成交量）

    优化策略（三级缓存）：
    1. 优先从 session_state 内存缓存读取（最快，毫秒级）
    2. 其次从本地 Parquet 文件读取（次快，约100ms）
    3. 最后从网络 API 获取（最慢，约2-3秒）

    Args:
        code: 股票代码
        days: 历史天数，默认365天

    Returns:
        DataFrame，包含日期、开盘、最高、最低、收盘、成交量

    Raises:
        MarketDataServiceError: 当数据获取失败时抛出
    """
    import os
    from pathlib import Path

    # === 第一级缓存：session_state 内存缓存 ===
    cache_key = f'_candle_cache_{code}'
    if cache_key in st.session_state:
        # 检查缓存是否过期（5分钟有效期）
        cache_time_key = f'_candle_cache_time_{code}'
        if cache_time_key in st.session_state:
            cache_age = (datetime.now() - st.session_state[cache_time_key]).total_seconds()
            if cache_age < 300:  # 5分钟内有效
                return st.session_state[cache_key]

    # === 第二级缓存：本地 Parquet 文件（OHLCV完整数据）===
    candle_dir = Path("market_data/candle")
    candle_dir.mkdir(parents=True, exist_ok=True)
    candle_filepath = candle_dir / f"{code}_ohlcv.parquet"

    if candle_filepath.exists():
        try:
            df = pd.read_parquet(candle_filepath)
            # 检查数据是否需要更新（最后日期是否为今天或昨天）
            if not df.empty:
                last_date = pd.to_datetime(df['日期']).max().date()
                today = datetime.now().date()
                # 如果最后日期是今天或昨天（或周末时是上周五），使用缓存
                days_diff = (today - last_date).days
                is_weekend = datetime.now().weekday() >= 5
                if days_diff <= 1 or (is_weekend and days_diff <= 3):
                    # 存入 session_state 缓存
                    st.session_state[cache_key] = df
                    st.session_state[f'_candle_cache_time_{code}'] = datetime.now()
                    return df
        except Exception as e:
            print(f"读取本地K线缓存失败: {e}")

    # === 第三级：从网络获取 ===
    max_retries = 3
    retry_delay = 1  # 秒

    for attempt in range(max_retries):
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")

            # 确保返回的是DataFrame且有需要的列
            if not isinstance(df, pd.DataFrame) or df.empty:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise MarketDataServiceError(f"获取 {code} 的K线数据失败: 返回空DataFrame")

            # 重命名列
            df = df.rename(columns={
                '日期': '日期',
                '开盘': '开盘',
                '最高': '最高',
                '最低': '最低',
                '收盘': '收盘',
                '成交量': '成交量'
            })

            # 确保数据类型正确
            numeric_cols = ['开盘', '最高', '最低', '收盘', '成交量']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['日期'] = pd.to_datetime(df['日期'])

            result = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]

            # 保存到本地 Parquet 缓存
            try:
                result.to_parquet(candle_filepath, index=False)
            except Exception as e:
                print(f"保存K线缓存失败: {e}")

            # 存入 session_state 缓存
            st.session_state[cache_key] = result
            st.session_state[f'_candle_cache_time_{code}'] = datetime.now()

            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise MarketDataServiceError(f"获取 {code} 的K线数据失败: {e}")


def fetch_latest_prices(codes, names, force_network=False):
    """
    获取股票的最新价格（双轨制：优先实时快照接口，失败时回退历史接口）
    
    双轨制策略：
    1. 优先使用东方财富实时快照接口（ak.stock_zh_a_spot_em）获取交易时间内的最新价格
    2. 实时快照失败时，回退到历史数据接口（ak.stock_zh_a_hist）+ 本地Parquet缓存
    3. 支持强制网络刷新模式，穿透所有缓存获取实时数据
    
    性能优势：
    1. 交易时间内能获取真正的实时价格（而非昨天收盘价）
    2. 实时快照接口一次性返回全市场数据，避免多次请求
    3. 历史接口+本地Parquet提供稳定的兜底方案
    4. 智能缓存机制平衡实时性和性能
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
        force_network: 是否强制穿透所有缓存，直接联网获取最新快照（默认False）
    
    Returns:
        DataFrame，列名为股票名称，索引为日期（最新日期），只有一行数据
    
    Raises:
        MarketDataServiceError: 当数据获取失败时抛出
    """
    try:
        # 1. 检查缓存是否有效（仅在非强制模式下生效）
        cache_key = f'_latest_prices_cache_{",".join(sorted(codes))}'
        cache_time_key = f'_latest_prices_cache_time_{",".join(sorted(codes))}'
        
        current_time = datetime.now()
        cache_valid = False
        
        # 仅在非强制模式下检查缓存
        if not force_network and (cache_key in st.session_state and 
            cache_time_key in st.session_state):
            cache_age = (current_time - st.session_state[cache_time_key]).total_seconds()
            if cache_age < 120:  # 2分钟缓存
                df_latest = st.session_state[cache_key]
                print(f"使用缓存的最新价格数据（{cache_age:.0f}秒前）")
                return df_latest
        
        # 2. 尝试从实时快照接口获取最新价格（优先）- 根据 force_network 决定是否启用
        latest_prices = {}
        spot_success = False
        
        # 只有在强制网络模式下才尝试实时快照接口（用于06_live_portfolio的实时刷新）
        if force_network:
            try:
                # 使用东方财富实时快照接口（支持交易时间内的实时价格）
                spot_df = ak.stock_zh_a_spot_em()
                
                # 为每个股票代码匹配最新价格
                for code, name in zip(codes, names):
                    # 匹配6位代码
                    match = spot_df[spot_df['代码'] == code]
                    if not match.empty:
                        price = match.iloc[0]['最新价']
                        # 停牌处理：如果最新价为0或NaN，取昨收
                        if price == 0 or pd.isna(price):
                            price = match.iloc[0]['昨收']
                        
                        if price is not None and not pd.isna(price) and price > 0:
                            latest_prices[name] = price
                            print(f"从实时快照获取 {code} 的最新价格: {price}")
                            spot_success = True
                        else:
                            print(f"实时快照中 {code} 的价格无效: {price}")
                    else:
                        print(f"实时快照中未找到 {code} 的数据")
            except Exception as e:
                print(f"实时快照接口获取失败，将回退历史接口: {e}")
        else:
            # 非强制网络模式下，跳过实时快照接口，直接使用本地历史数据
            print("非强制网络模式，跳过实时快照接口，直接使用本地历史数据")
        
        # 3. 兜底逻辑：如果实时快照未成功获取所有股票，使用历史接口+本地Parquet
        if not spot_success or len(latest_prices) != len(codes):
            for code, name in zip(codes, names):
                # 如果已经通过实时快照获取了该股票的价格，跳过
                if name in latest_prices:
                    continue
                    
                price = None
                
                try:
                    # 直接从本地Parquet读取数据（不自动触发增量更新）
                    # 增量更新应由用户手动触发，避免页面切换时的不必要等待
                    from market_store import MarketStore
                    store = MarketStore()

                    # 读取该股票的历史数据（从本地Parquet，不触发更新）
                    hist_data = store.read_stock_data(code, update_if_needed=False)
                    
                    if not hist_data.empty:
                        # 获取最新收盘价（可能是昨天的）
                        price = hist_data['close'].iloc[-1]
                        print(f"从本地Parquet获取 {code} 的收盘价: {price}")
                    else:
                        # 如果本地没有数据，则尝试联网获取最近一天的收盘价
                        end_date = datetime.now().strftime("%Y%m%d")
                        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")
                        df = ak.stock_zh_a_hist(
                            symbol=code, 
                            period="daily", 
                            start_date=start_date, 
                            end_date=end_date, 
                            adjust="qfq"
                        )
                        if not df.empty:
                            price = df.iloc[-1]['收盘']
                            print(f"从历史接口获取 {code} 的收盘价: {price}")
                            
                            # 将获取的数据保存到本地Parquet（避免下次再联网）
                            temp_df = pd.DataFrame({
                                'date': [pd.to_datetime(df.iloc[-1]['日期'])],
                                'close': [price],
                                'code': [code]
                            }).set_index('date')
                            filepath = store._get_filepath(code)
                            temp_df.to_parquet(filepath)
                except Exception as e:
                    print(f"获取 {code} 历史价格失败: {e}")
                    # 继续尝试其他股票
                
                if price is not None and not pd.isna(price) and price > 0:
                    latest_prices[name] = price
                else:
                    # 如果仍然无法获取价格，使用NaN
                    latest_prices[name] = np.nan
        
        # 4. 确保所有股票都有价格（即使是NaN）
        for name in names:
            if name not in latest_prices:
                latest_prices[name] = np.nan
        
        # 创建DataFrame，索引为当前时间
        df_latest = pd.DataFrame([latest_prices], index=[current_time])
        
        # 5. 更新缓存（仅在非强制模式下或强制模式成功获取后）
        if not force_network or (force_network and not df_latest.empty):
            st.session_state[cache_key] = df_latest
            st.session_state[cache_time_key] = current_time
        
        return df_latest
    except Exception as e:
        raise MarketDataServiceError(f"获取最新价格失败: {e}")


def search_stock_info(query, use_local=True, limit=20):
    """
    根据股票代码或名称模糊查询股票信息

    优先使用本地市场标的库进行搜索，提高搜索效率，避免频繁网络请求。
    如果本地库不存在或use_local=False，则从网络获取。

    Args:
        query: 股票代码或名称（支持模糊匹配）
        use_local: 是否优先使用本地市场库搜索（默认True）
        limit: 返回结果数量限制

    Returns:
        字典列表，每个字典包含股票代码和名称

    Raises:
        MarketDataServiceError: 当查询失败时抛出
    """
    # 如果查询为空，返回空列表
    if not query or len(query.strip()) == 0:
        return []

    query = query.strip()

    # 优先使用本地库搜索
    if use_local:
        local_results = safe_search_local_stock(query, limit)
        if local_results:
            return local_results
        # 检查本地库是否存在
        info = get_market_stock_list_info()
        if info['exists']:
            # 本地库存在但没有匹配结果，直接返回空
            return []

    # 本地库不存在或use_local=False，从网络获取
    try:
        stock_list = ak.stock_info_a_code_name()

        # 模糊匹配
        results = []
        for _, row in stock_list.iterrows():
            code = str(row['code'])
            name = str(row['name'])

            # 匹配代码（精确或部分匹配）
            code_match = query in code
            # 匹配名称（中文模糊匹配）
            name_match = query in name

            if code_match or name_match:
                results.append({
                    'code': code,
                    'name': name
                })
                # 限制返回结果
                if len(results) >= limit:
                    break

        return results
    except Exception as e:
        raise MarketDataServiceError(f"股票信息查询失败: {e}")


def get_portfolio_summary(portfolio_list, latest_prices):
    """
    计算真实持仓的盈亏情况

    Args:
        portfolio_list: 来自 config['portfolio']
        latest_prices: 当前获取的实时股价 DataFrame

    Returns:
        tuple: (summary_df, total_market_value, total_profit_ratio)

    Raises:
        MarketDataServiceError: 当计算失败时抛出
    """
    summary_data = []
    total_cost = 0.0
    total_market_value = 0.0

    for trade in portfolio_list:
        name = trade['name']
        code = trade.get('code', 'N/A')

        # 检查数据完整性
        if trade['buy_price'] <= 0:
            print(f"警告：{name}的买入价格异常，跳过该记录")
            continue

        # 获取最新收盘价
        if name in latest_prices.columns:
            current_p = latest_prices[name].iloc[-1]
        else:
            # 如果没有该股票的数据，跳过
            print(f"警告：{name}没有价格数据，跳过该记录")
            continue

        # 检查是否为停牌（价格为NaN、0或负数）
        is_suspended = False
        status_mark = ""
        if pd.isna(current_p) or current_p <= 0:
            is_suspended = True
            status_mark = " 🚫停牌"
            # 停牌时尝试从历史数据获取最后交易日收盘价
            # 如果无法获取，则使用成本价作为兜底估值
            try:
                from market_store import MarketStore
                store = MarketStore()
                hist_data = store.read_stock_data(code, update_if_needed=False)
                if not hist_data.empty and 'close' in hist_data.columns:
                    last_close = hist_data['close'].iloc[-1]
                    if not pd.isna(last_close) and last_close > 0:
                        current_p = last_close
                        print(f"警告：{name}（{code}）停牌，使用最后交易日收盘价 {current_p:.2f}")
                    else:
                        current_p = trade['buy_price']
                        print(f"警告：{name}（{code}）停牌且无有效历史价格，使用成本价 {current_p:.2f} 估值")
                else:
                    current_p = trade['buy_price']
                    print(f"警告：{name}（{code}）停牌且无历史数据，使用成本价 {current_p:.2f} 估值")
            except Exception as e:
                current_p = trade['buy_price']
                print(f"警告：{name}（{code}）停牌且获取历史价格失败，使用成本价 {current_p:.2f} 估值: {e}")

        # 计算核心指标
        cost = trade['buy_price'] * trade['quantity']
        market_val = current_p * trade['quantity']
        profit = market_val - cost
        profit_ratio = (current_p / trade['buy_price']) - 1 if trade['buy_price'] > 0 else 0

        total_cost += cost
        total_market_value += market_val

        summary_data.append({
            "代码": code,  # 添加股票代码便于后续编辑
            "资产名称": name + status_mark,  # 停牌标记
            "成本价": trade['buy_price'],
            "现价": current_p,
            "持仓量": trade['quantity'],
            "成本": cost,
            "市值": market_val,
            "盈亏额": profit,
            "盈亏比": profit_ratio
        })

    total_profit_ratio = (total_market_value / total_cost - 1) if total_cost > 0 else 0
    return pd.DataFrame(summary_data), total_market_value, total_profit_ratio


def get_recent_10_days(data):
    """
    返回最近10个交易日的价格数据
    
    Args:
        data: DataFrame或字典，索引为日期，列为股票名称
    
    Returns:
        DataFrame，最近10个交易日的数据
    
    Raises:
        MarketDataServiceError: 当数据处理失败时抛出
    """
    # 确保data是DataFrame
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            raise MarketDataServiceError(f"无法将数据转换为DataFrame: {e}")
    
    # 检查DataFrame是否为空
    if data.empty:
        return pd.DataFrame()
    
    # 返回最近10行
    return data.tail(10) if len(data) >= 10 else data


def safe_fetch_stock_data(codes, names):
    """
    安全版本的 fetch_stock_data，捕获异常并返回空 DataFrame
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
    
    Returns:
        DataFrame: 成功时返回数据，失败时返回空 DataFrame
    """
    try:
        return fetch_stock_data(codes, names)
    except MarketDataServiceError:
        return pd.DataFrame()


def safe_fetch_latest_prices(codes, names, force_network=False):
    """
    安全版本的 fetch_latest_prices，捕获异常并返回空 DataFrame
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
        force_network: 是否强制穿透所有缓存，直接联网获取最新快照（默认False）
    
    Returns:
        DataFrame: 成功时返回数据，失败时返回空 DataFrame
    """
    try:
        return fetch_latest_prices(codes, names, force_network)
    except MarketDataServiceError:
        return pd.DataFrame(columns=names)


def safe_fetch_candle_data(code, days=365):
    """
    安全版本的 fetch_candle_data，捕获异常并返回空 DataFrame
    
    Args:
        code: 股票代码
        days: 历史天数，默认365天
    
    Returns:
        DataFrame: 成功时返回数据，失败时返回空 DataFrame
    """
    try:
        return fetch_candle_data(code, days)
    except MarketDataServiceError:
        return pd.DataFrame()


def safe_search_stock_info(query, use_local=True, limit=20):
    """
    安全版本的 search_stock_info，捕获异常并返回空列表

    Args:
        query: 股票代码或名称
        use_local: 是否优先使用本地市场库搜索（默认True）
        limit: 返回结果数量限制

    Returns:
        list: 成功时返回结果列表，失败时返回空列表
    """
    try:
        return search_stock_info(query, use_local, limit)
    except MarketDataServiceError:
        return []


def safe_get_signals(data, best_weights, stop_loss, latest_prices=None):
    """
    安全版本的 get_signals，捕获异常并返回空 DataFrame
    
    Args:
        data: 股票价格 DataFrame 或字典
        best_weights: 最优权重字典 {股票名称: 权重}
        stop_loss: 止损阈值（负数）
        latest_prices: 可选，包含最新价格的DataFrame，用于计算当日涨跌
    
    Returns:
        DataFrame: 成功时返回信号数据，失败时返回空 DataFrame
    """
    try:
        from data_fetcher import get_signals
        return get_signals(data, best_weights, stop_loss, latest_prices)
    except Exception as e:
        print(f"获取交易信号失败: {e}")
        return pd.DataFrame()
