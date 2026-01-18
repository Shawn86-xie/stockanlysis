"""
FILE: data_fetcher.py
ROLE: 数据抓取引擎，负责从 AkShare API 获取股票数据，实现本地 Parquet 缓存、增量更新、数据完整性校验和实时行情获取。
LOGIC: 
    1. 优先从本地 Parquet 读取历史数据，减少网络请求。
    2. 实现数据完整性检查和自动重试机制，提高系统鲁棒性。
    3. 提供实时行情获取接口，支持缓存和兜底逻辑。
    4. 支持批量股票数据获取和增量更新。
DEPENDENCIES: pandas, akshare, numpy, streamlit, market_store, math_engine
"""

import pandas as pd
import akshare as ak
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import time

# 导入本地行情仓库
from market_store import get_market_store, ensure_data_initialized

# 导入数学引擎
from math_engine import calculate_screening_score

def fetch_stock_data(codes, names, incremental_update=False, max_retry=3):
    """
    获取股票历史数据（优化版：优先从本地Parquet读取，带完整性检查和自动重试）
    
    为什么要使用Parquet + session_state：
    1. Parquet是列式存储，读取速度快，压缩率高，适合时间序列数据
    2. 本地存储避免了每次联网请求，大幅提升加载速度
    3. session_state可以跨rerun缓存数据，避免重复I/O
    4. 首次使用才从网络拉取，后续使用秒级读取
    
    新增的数据完整性检查：
    1. 在分析前检查每只股票的数据完整性
    2. 如果数据不完整，自动尝试重新下载
    3. 多次重试失败后才报错，提高系统鲁棒性
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
        incremental_update: 是否执行增量更新（默认False，仅当用户明确要求时执行）
        max_retry: 最大重试次数，默认3次
    Returns:
        DataFrame，列为股票名称，索引为日期
    """
    # 获取本地行情仓库实例
    store = get_market_store()
    
    # 第一步：确保所有股票都有本地数据（带重试，使用并行优化）
    for retry in range(max_retry):
        print(f"第{retry+1}次尝试初始化数据...")

        # ✅ 使用批量并行初始化（性能优化）
        results = store.batch_init_stock_data(
            codes=codes,
            names=names,
            days=365,
            force_update=(retry > 0),
            max_workers=5  # 5线程并行，平衡性能和API限流风险
        )

        # 检查是否全部成功
        all_success = all(results.values())

        if all_success:
            break
        elif retry < max_retry - 1:
            wait_time = 2 ** retry  # 指数退避
            print(f"数据初始化失败，{wait_time}秒后重试...")
            time.sleep(wait_time)
    
    # 第二步：从本地Parquet读取数据（根据参数决定是否更新）
    data = store.read_multiple_stocks(codes, names, update_if_needed=incremental_update)
    
    # 第三步：检查数据完整性
    valid_data = {}
    missing_stocks = []
    
    for code, name in zip(codes, names):
        if name in data.columns:
            # 检查该列是否有足够的数据（至少10个交易日）
            stock_data = data[name].dropna()
            if len(stock_data) >= 10:
                valid_data[name] = stock_data
            else:
                print(f"股票 {name}({code}) 数据不足，仅 {len(stock_data)} 条记录，尝试重新下载...")
                missing_stocks.append((code, name))
        else:
            print(f"股票 {name}({code}) 在数据中缺失，尝试重新下载...")
            missing_stocks.append((code, name))
    
    # 第四步：如果有缺失或不完整的数据，尝试重新下载（使用并行）
    if missing_stocks:
        print(f"发现 {len(missing_stocks)} 只股票数据缺失或不完整，尝试重新下载...")

        # ✅ 使用批量并行重新下载
        missing_codes = [code for code, name in missing_stocks]
        missing_names = [name for code, name in missing_stocks]

        results = store.batch_init_stock_data(
            codes=missing_codes,
            names=missing_names,
            days=365,
            force_update=True,  # 强制重新初始化
            max_workers=3  # 缺失数据通常较少，用3线程即可
        )
        
        # 重新读取数据
        data = store.read_multiple_stocks(codes, names, update_if_needed=False)
    
    # 第五步：最终检查，如果数据仍然为空，尝试兜底获取
    if data.empty:
        print("警告：本地数据为空，尝试从AkShare获取（兜底逻辑）...")
        data = _fallback_fetch_from_akshare(codes, names)
        
        # 如果兜底获取成功，保存到本地
        if not data.empty:
            print("兜底获取成功，保存数据到本地...")
            for code, name in zip(codes, names):
                if name in data.columns:
                    # 提取该股票的数据
                    stock_df = data[[name]].copy()
                    stock_df = stock_df.dropna()
                    if not stock_df.empty:
                        # 转换为MarketStore格式并保存
                        df_to_save = pd.DataFrame({
                            'close': stock_df[name],
                            'code': code
                        }, index=stock_df.index)
                        filepath = store._get_filepath(code)
                        try:
                            df_to_save.to_parquet(filepath)
                            print(f"已保存 {name}({code}) 数据到本地，共 {len(df_to_save)} 条记录")
                        except Exception as e:
                            print(f"保存 {name}({code}) 数据失败: {e}")
    
    # 第六步：过滤掉仍然缺失的股票
    final_data = pd.DataFrame()
    for name in names:
        if name in data.columns:
            if final_data.empty:
                final_data = data[[name]].copy()
            else:
                final_data[name] = data[name]
    
    if final_data.empty:
        print("错误：无法获取任何股票数据")
        return pd.DataFrame()
    
    print(f"成功获取 {len(final_data.columns)}/{len(names)} 只股票的数据，共 {len(final_data)} 个交易日")
    return final_data

def _fallback_fetch_from_akshare(codes, names):
    """
    兜底函数：从AkShare获取股票历史数据（当本地数据不可用时）
    
    注意：此函数仅在本地数据完全缺失时使用
    使用st.cache_data缓存，避免重复请求
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
    Returns:
        DataFrame，列为股票名称，索引为日期
    """
    @st.cache_data(ttl=3600)
    def _fetch(codes, names):
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        dfs = []
        for code, name in zip(codes, names):
            try:
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                # 确保返回的是DataFrame
                if not isinstance(df, pd.DataFrame):
                    print(f"获取 {code}({name}) 数据失败: 返回类型不是DataFrame")
                    continue
                df = df[['日期', '收盘']].rename(columns={'日期': 'Date', '收盘': name})
                # 确保收盘价列是数值类型
                df[name] = pd.to_numeric(df[name], errors='coerce')
                df['Date'] = pd.to_datetime(df['Date'])
                dfs.append(df.set_index('Date'))
            except Exception as e:
                print(f"获取 {code}({name}) 数据失败: {e}")
                continue
        
        # 确保返回DataFrame，即使为空
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, axis=1).dropna()
        
        # 确保结果是DataFrame
        if not isinstance(result, pd.DataFrame):
            print("警告: 合并后的结果不是DataFrame，返回空DataFrame")
            return pd.DataFrame()
        
        return result
    
    return _fetch(codes, names)

def get_signals(data, best_weights, stop_loss, latest_prices=None):
    """
    生成交易信号与风险提醒
    Args:
        data: 股票价格 DataFrame 或字典
        best_weights: 最优权重字典 {股票名称: 权重}
        stop_loss: 止损阈值（负数）
        latest_prices: 可选，包含最新价格的DataFrame，用于计算当日涨跌
    Returns:
        DataFrame 包含各标的信号
    """
    signals = []
    
    # 确保data是DataFrame
    if not isinstance(data, pd.DataFrame):
        # 尝试转换为DataFrame
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            print(f"无法将数据转换为DataFrame: {e}")
            return pd.DataFrame()
    
    # 检查DataFrame是否为空
    if data.empty:
        return pd.DataFrame()
    
    for name in data.columns:
        try:
            prices = data[name]
            # 确保价格序列是数值类型
            prices = pd.to_numeric(prices, errors='coerce').dropna()
            # 检查数据长度是否足够
            if len(prices) < 2:
                continue
            
            # 确定当前价格和昨日收盘价
            curr_p = float(prices.iloc[-1])
            prev_p = float(prices.iloc[-2])
            
            # 如果提供了最新价格，使用最新价格计算当日涨跌
            daily_change = 0.0
            if latest_prices is not None and not latest_prices.empty and name in latest_prices.columns:
                latest_price = latest_prices[name].iloc[-1]
                if not pd.isna(latest_price) and latest_price > 0:
                    # 使用最新价格计算涨跌
                    curr_p = float(latest_price)
                    daily_change = (curr_p - prev_p) / prev_p
                else:
                    # 如果最新价格无效，使用历史数据计算
                    daily_change = prices.pct_change().iloc[-1]
            else:
                # 使用历史数据计算
                daily_change = prices.pct_change().iloc[-1]
            
            if pd.isna(daily_change):
                daily_change = 0
                
            # 移动平均线
            ma60 = prices.rolling(window=60).mean().iloc[-1] if len(prices) >= 60 else curr_p
            # RSI 简易计算
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            
            # 避免除零错误
            if len(prices) >= 14 and loss.iloc[-1] != 0:
                rsi = 100 - (100 / (1 + (gain.iloc[-1] / loss.iloc[-1])))
            else:
                rsi = 50  # 默认值
            
            advice = "💎 持有"
            if rsi < 35 and curr_p > ma60:
                advice = "✅ 建议买入"
            elif rsi > 75:
                advice = "🚨 建议减仓"
            
            # 风险提醒
            risk_tag = "正常"
            if daily_change <= stop_loss:
                risk_tag = f"‼️ 触及止损 ({daily_change:.1%})"
                
            signals.append({
                "标的": name,
                "价格": f"{curr_p:.2f}",
                "当日涨跌": f"{daily_change:.2%}",
                "RSI": f"{rsi:.1f}",
                "建议": advice,
                "风险": risk_tag,
                "最优配比": f"{best_weights.get(name, 0):.1%}"
            })
        except Exception as e:
            print(f"计算 {name} 信号时出错: {e}")
            continue
    return pd.DataFrame(signals)

def compute_portfolio_stats(returns, risk_free_rate=0.0188):
    """
    计算投资组合统计量：年化收益率、协方差矩阵
    Args:
        returns: 日收益率 DataFrame
        risk_free_rate: 无风险利率
    Returns:
        mean_ret, cov_mat
    """
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    return mean_ret, cov_mat

def monte_carlo_simulation(mean_ret, cov_mat, n_sim=3000, risk_free_rate=0.0188):
    """
    蒙特卡洛模拟生成随机权重，计算收益、波动率和夏普比率
    Args:
        mean_ret: 年化收益率序列
        cov_mat: 协方差矩阵
        n_sim: 模拟次数
        risk_free_rate: 无风险利率
    Returns:
        DataFrame 包含每次模拟的 Ret, Vol, Sharpe 及权重
    """
    n_assets = len(mean_ret)
    sim_res = []
    for _ in range(n_sim):
        w = np.random.random(n_assets)
        w /= np.sum(w)
        ret = np.sum(mean_ret * w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
        sharpe = (ret - risk_free_rate) / vol
        sim_res.append([ret, vol, sharpe] + list(w))
    columns = ['Ret', 'Vol', 'Sharpe'] + list(mean_ret.index)
    return pd.DataFrame(sim_res, columns=columns)

def get_recent_10_days(data):
    """
    返回最近10个交易日的价格数据
    Args:
        data: DataFrame或字典，索引为日期，列为股票名称
    Returns:
        DataFrame，最近10个交易日的数据
    """
    # 确保data是DataFrame
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            print(f"无法将数据转换为DataFrame: {e}")
            return pd.DataFrame()
    
    # 检查DataFrame是否为空
    if data.empty:
        return pd.DataFrame()
    
    # 返回最近10行
    return data.tail(10) if len(data) >= 10 else data

def analyze_portfolio(holdings, data, best_weights, stop_loss):
    """
    分析持仓并给出调仓建议
    Args:
        holdings: 字典 {股票名称: 持仓数量}
        data: 股票价格 DataFrame
        best_weights: 最优权重字典 {股票名称: 权重}
        stop_loss: 止损阈值
    Returns:
        包含持仓市值、权重、建议、调仓数量等的 DataFrame
    """
    analysis = []
    total_value = 0
    for name, qty in holdings.items():
        if name in data.columns:
            price = data[name].iloc[-1]
            value = price * qty
            total_value += value
            analysis.append({
                "标的": name,
                "持仓数量": qty,
                "当前价格": f"{price:.2f}",
                "持仓市值": f"{value:.2f}",
                "当前权重": "",  # 稍后计算
                "建议权重": f"{best_weights.get(name, 0):.1%}",
                "调仓建议": "",
                "调仓数量": 0  # 以100股为单位的调仓数量
            })
    # 计算当前权重和调仓数量
    for item in analysis:
        value = float(item["持仓市值"])
        item["当前权重"] = f"{(value / total_value):.1%}" if total_value > 0 else "0.0%"
        # 生成调仓建议
        curr_weight = float(item["当前权重"].strip('%')) / 100
        sugg_weight = float(item["建议权重"].strip('%')) / 100
        diff = sugg_weight - curr_weight
        if diff > 0.05:
            # 计算增持金额，然后转换为100股的整数倍
            target_value = total_value * sugg_weight
            current_value = value
            add_value = target_value - current_value
            # 转换为股数（按100取整）
            add_shares = int(round(add_value / float(item["当前价格"]) / 100) * 100)
            if add_shares > 0:
                item["调仓建议"] = f"增持 {add_shares}股"
                item["调仓数量"] = add_shares
            else:
                item["调仓建议"] = "保持"
        elif diff < -0.05:
            # 计算减持金额，然后转换为100股的整数倍
            target_value = total_value * sugg_weight
            current_value = value
            reduce_value = current_value - target_value
            reduce_shares = int(round(reduce_value / float(item["当前价格"]) / 100) * 100)
            if reduce_shares > 0:
                item["调仓建议"] = f"减持 {reduce_shares}股"
                item["调仓数量"] = -reduce_shares
            else:
                item["调仓建议"] = "保持"
        else:
            item["调仓建议"] = "保持"
    return pd.DataFrame(analysis), total_value


def get_portfolio_summary(portfolio_list, latest_prices):
    """
    计算真实持仓的盈亏情况
    portfolio_list: 来自 config['portfolio']
    latest_prices: 当前获取的实时股价 DataFrame
    """
    summary_data = []
    total_cost = 0.0
    total_market_value = 0.0

    for trade in portfolio_list:
        name = trade['name']
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
        
        # 计算核心指标
        cost = trade['buy_price'] * trade['quantity']
        market_val = current_p * trade['quantity']
        profit = market_val - cost
        profit_ratio = (current_p / trade['buy_price']) - 1 if trade['buy_price'] > 0 else 0
        
        total_cost += cost
        total_market_value += market_val
        
        summary_data.append({
            "资产名称": name,
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


def calculate_deviation(current_weights, optimal_weights):
    """
    计算当前持仓与最优配置之间的偏离度
    current_weights: 字典 {资产名称: 当前权重}
    optimal_weights: 字典 {资产名称: 最优权重}
    """
    deviation = 0.0
    deviations = {}
    
    for name in set(current_weights.keys()) | set(optimal_weights.keys()):
        curr = current_weights.get(name, 0)
        opt = optimal_weights.get(name, 0)
        dev = abs(curr - opt)
        deviations[name] = dev
        deviation += dev
    
    return deviation, deviations


def search_stock_info(query):
    """
    根据股票代码或名称模糊查询股票信息
    Args:
        query: 股票代码或名称（支持模糊匹配）
    Returns:
        字典列表，每个字典包含股票代码和名称
    """
    try:
        import akshare as ak
        
        # 获取A股股票列表
        stock_list = ak.stock_info_a_code_name()
        
        # 如果查询为空，返回空列表
        if not query or len(query.strip()) == 0:
            return []
        
        query = query.strip()
        
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
                # 限制最多返回10个结果
                if len(results) >= 10:
                    break
        
        return results
    except Exception as e:
        print(f"股票信息查询失败: {e}")
        # 返回一些常见股票的示例数据（备用）
        if query:
            return [
                {'code': '000001', 'name': '平安银行'},
                {'code': '000002', 'name': '万科A'},
                {'code': '600036', 'name': '招商银行'},
                {'code': '601318', 'name': '中国平安'},
                {'code': '000858', 'name': '五粮液'},
                {'code': '000333', 'name': '美的集团'},
                {'code': '002594', 'name': '比亚迪'},
                {'code': '600276', 'name': '恒瑞医药'},
                {'code': '300347', 'name': '泰格医药'},
                {'code': '603259', 'name': '药明康德'}
            ]
        return []


@st.cache_data(ttl=3600)  # 增加到1小时缓存，减少重复请求
def fetch_candle_data(code, days=365):
    """
    获取股票的OHLC数据（开盘、最高、最低、收盘、成交量）
    优化：增加重试机制和更长的缓存时间
    Args:
        code: 股票代码
        days: 历史天数，默认365天
    Returns:
        DataFrame，包含日期、开盘、最高、最低、收盘、成交量
    """
    max_retries = 3
    retry_delay = 1  # 秒
    
    for attempt in range(max_retries):
        try:
            import akshare as ak
            from datetime import datetime, timedelta
            
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
            
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            
            # 确保返回的是DataFrame且有需要的列
            if not isinstance(df, pd.DataFrame) or df.empty:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                print(f"获取 {code} 的K线数据失败: 返回空DataFrame")
                return pd.DataFrame()
            
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
            
            return df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"获取 {code} 数据失败，第{attempt+1}次重试: {e}")
                time.sleep(retry_delay)
                continue
            print(f"获取 {code} 的K线数据失败: {e}")
            return pd.DataFrame()


@st.cache_data(ttl=600)  # 缓存10分钟，减少实时行情请求
def fetch_latest_prices(codes, names):
    """
    获取股票的最新价格（实时行情），如果实时行情获取失败，则使用最近一个交易日的收盘价
    优化：批量处理请求，增加缓存时间
    Args:
        codes: 股票代码列表
        names: 股票名称列表
    Returns:
        DataFrame，列名为股票名称，索引为日期（最新日期），只有一行数据
    """
    try:
        import akshare as ak
        from datetime import datetime, timedelta
        
        # 首先尝试获取实时行情（一次性获取所有股票）
        spot_df = ak.stock_zh_a_spot()
        # 将代码映射到名称
        code_to_name = dict(zip(codes, names))
        
        latest_prices = {}
        for code, name in zip(codes, names):
            price = None
            
            # 构建市场代码映射表（先一次性构建，避免重复计算）
            if code.startswith('6'):
                market_code = f'sh{code}'
            else:
                market_code = f'sz{code}'
            
            # 使用向量化查找提高效率
            if market_code in spot_df['代码'].values:
                spot_row = spot_df[spot_df['代码'] == market_code].iloc[0]
                price = spot_row['最新价']
                # 如果最新价为0或NaN，可能表示停牌或数据异常，则使用昨收
                if pd.isna(price) or price == 0:
                    price = spot_row['昨收']
            
            # 如果实时行情没有找到或价格无效，则尝试获取最近一天的收盘价
            if price is None or pd.isna(price) or price == 0:
                # 批量获取历史数据（减少单独请求）
                try:
                    end_date = datetime.now().strftime("%Y%m%d")
                    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                    if not df.empty:
                        price = df.iloc[-1]['收盘']
                except Exception as e:
                    print(f"获取 {code} 最近收盘价失败: {e}")
            
            if price is not None and not pd.isna(price) and price > 0:
                latest_prices[name] = price
            else:
                # 如果仍然无法获取价格，使用NaN
                latest_prices[name] = np.nan
        
        # 创建DataFrame，索引为当前日期
        df_latest = pd.DataFrame([latest_prices], index=[datetime.now().date()])
        return df_latest
    except Exception as e:
        print(f"获取实时行情失败: {e}")
        # 如果全部失败，返回一个空的DataFrame
        return pd.DataFrame(columns=names)


@st.cache_data(ttl=3600)
def rank_master_pool(master_pool, weights=None):
    """
    批量诊断逻辑：遍历master_pool中的所有分类和股票，计算综合评分并排序
    
    Args:
        master_pool: 字典，键为分类名称，值为股票代码列表或股票信息字典
                    例如：{'分类1': ['000001', '000002'], '分类2': ['600036']}
                    或者：{'分类1': [{'code': '000001', 'name': '平安银行'}, ...]}
        weights: 权重字典，格式如 {'k': 0.4, 'r2': 0.3, 'dist': 0.2, 'vol': 0.1}
                默认使用均衡稳健型: k:0.35, r2:0.35, dist:0.20, vol:0.10
    
    Returns:
        DataFrame，按'综合评分'降序排列，包含以下列：
        - 分类: 股票所属分类
        - 代码: 股票代码
        - 名称: 股票名称（如果有）
        - 斜率(k): 阻力线斜率
        - 归一化斜率(k_rel)
        - R²: 拟合优度
        - 距上轨距离(dist_to_upper)
        - 距离权重(dist_weight)
        - 量能强度(vol_intensity)
        - 量能权重(vol_weight)
        - 综合评分(final_score)
    """
    import streamlit as st
    from datetime import datetime, timedelta
    
    results = []
    
    # 遍历master_pool中的所有分类
    for category, stocks in master_pool.items():
        print(f"处理分类: {category}, 共 {len(stocks)} 只股票")
        
        # 处理股票列表，可能是字符串代码或字典
        for stock in stocks:
            try:
                # 解析股票信息
                if isinstance(stock, dict):
                    code = stock.get('code')
                    name = stock.get('name', code)
                else:
                    code = str(stock)
                    name = code
                
                if not code:
                    continue
                
                # 获取最近60个交易日的历史数据
                # 使用fetch_candle_data获取OHLC数据
                df_candle = fetch_candle_data(code, days=120)  # 获取120天以确保有60个交易日
                if df_candle.empty or len(df_candle) < 60:
                    print(f"股票 {name}({code}) 数据不足，跳过")
                    continue
                
                # 准备calculate_screening_score所需的DataFrame
                # 需要'close'和'volume'列
                df = pd.DataFrame({
                    'close': df_candle['收盘'],
                    'volume': df_candle['成交量']
                })
                
                # 调用评分函数，传递权重参数
                score_result = calculate_screening_score(df, window=60, weights=weights)
                
                # 添加结果
                results.append({
                    '分类': category,
                    '代码': code,
                    '名称': name,
                    '斜率(k)': score_result.get('k', 0),
                    '归一化斜率(k_rel)': score_result.get('k_rel', 0),
                    'R²': score_result.get('r_squared', 0),
                    '距上轨距离': score_result.get('dist_to_upper', 0),
                    '距离权重': score_result.get('dist_weight', 0),
                    '量能强度': score_result.get('vol_intensity', 0),
                    '量能权重': score_result.get('vol_weight', 0),
                    '综合评分': score_result.get('final_score', 0)
                })
                
                print(f"  股票 {name}({code}) 评分: {score_result.get('final_score', 0):.4f}")
                
            except Exception as e:
                print(f"处理股票 {stock} 时出错: {e}")
                # 继续处理其他股票，不影响全局
                continue
    
    # 如果没有结果，返回空DataFrame
    if not results:
        return pd.DataFrame()
    
    # 转换为DataFrame并排序
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('综合评分', ascending=False).reset_index(drop=True)
    
    print(f"批量诊断完成，共处理 {len(result_df)} 只股票")
    return result_df


def finalize_daily_data():
    """
    盘后结案：将热数据合并到冷数据中
    
    时间触发：判断当前时间是否在16:00之后。
    操作：将今日的热缓冲区数据（Snapshot）读取出来，打上 is_finalized=True 标签。
    合并：读入历史 Parquet，追加今日数据，进行 drop_duplicates(keep='last')。
    归档：覆盖写入冷库文件，并清理内存中的临时热数据。
    
    Returns:
        结案结果字典
    """
    # 导入storage_manager，避免循环导入
    from storage_manager import finalize_all_daily_data
    
    # 从配置加载股票列表
    from config import load_config
    config = load_config()
    
    # 获取master_pool中的股票
    master_pool = config.get('master_pool', {})
    all_codes = []
    for category, stocks in master_pool.items():
        for code, name in stocks.items():
            all_codes.append(code)
    
    if not all_codes:
        return {'success': False, 'message': '配置文件中没有找到股票数据'}
    
    # 执行批量结案
    results = finalize_all_daily_data(all_codes)
    
    return {
        'success': results['success'] > 0,
        'message': f"盘后结案完成: 成功 {results['success']} / 失败 {results['failed']} / 总计 {results['total']}",
        'details': results['details']
    }


def get_aligned_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    高精度缝合算法：实现逻辑锚点校验、复权一致性检查和日期连续性审计
    
    算法步骤：
    1. 加载冷数据 (Cold Storage)
    2. 获取实时快照 (Live Snapshot)
    3. 执行“对齐审计” (Alignment Audit)：校验昨收价是否匹配
    4. 时间戳归一化 (Timestamp Normalization)：确保格式一致
    5. 安全缝合 (Safe Stitching)：合并数据并去重
    
    Args:
        ticker: 股票代码
        days: 历史天数
        
    Returns:
        对齐后的DataFrame
    """
    import streamlit as st
    
    # 1. 加载冷数据 (Cold Storage)
    from storage_manager import get_storage_manager
    storage_manager = get_storage_manager()
    df_history = storage_manager.load_from_cold_storage(ticker)
    
    if df_history.empty:
        # 如果没有本地数据，则初始化
        from market_store import get_market_store
        store = get_market_store()
        store.init_stock_data(ticker, days=days, force_update=False)
        df_history = storage_manager.load_from_cold_storage(ticker)
    
    # 2. 获取实时快照 (Live Snapshot)
    try:
        import akshare as ak
        from datetime import datetime
        
        # 获取实时行情
        spot_df = ak.stock_zh_a_spot()
        if ticker.startswith('6'):
            market_code = f'sh{ticker}'
        else:
            market_code = f'sz{ticker}'
        
        snapshot = {}
        if market_code in spot_df['代码'].values:
            spot_row = spot_df[spot_df['代码'] == market_code].iloc[0]
            snapshot['last_close'] = spot_row['昨收']
            snapshot['current_price'] = spot_row['最新价']
            snapshot['date'] = datetime.now().date().isoformat()
        else:
            # 如果实时行情获取失败，使用最近历史数据
            if not df_history.empty:
                snapshot['last_close'] = df_history['close'].iloc[-1]
                snapshot['current_price'] = df_history['close'].iloc[-1]
                snapshot['date'] = datetime.now().date().isoformat()
            else:
                # 无法获取快照，返回历史数据
                return df_history
    except Exception as e:
        print(f"获取实时快照失败: {e}")
        # 返回历史数据
        return df_history
    
    # 3. 执行“对齐审计” (Alignment Audit)
    if not df_history.empty:
        # 校验昨收价是否匹配
        local_last_close = float(df_history['close'].iloc[-1])
        api_prev_close = float(snapshot['last_close'])
        
        if abs(local_last_close - api_prev_close) > 1e-4:
            # 触发“重同步”警报
            print(f"检测到 {ticker} 数据基准偏移，正在进行复权校准...")
            
            # 强制重新同步：删除本地文件并重新下载全量数据
            from storage_manager import get_storage_manager
            storage_manager = get_storage_manager()
            resync_result = storage_manager.gap_filling_update(ticker, force_resync=True)
            
            if resync_result['success']:
                # 重新加载冷数据
                df_history = storage_manager.load_from_cold_storage(ticker)
                print(f"强制重新同步完成: {ticker}")
            else:
                print(f"强制重新同步失败: {resync_result['message']}")
                # 继续使用原有数据，但标记对齐问题
    
    # 4. 时间戳归一化 (Timestamp Normalization)
    # 确保今日实时行的时间戳格式与历史行完全一致
    today_date = pd.to_datetime(snapshot['date']).normalize()
    
    # 5. 安全缝合 (Safe Stitching)
    # 创建今日数据行
    today_row = pd.DataFrame({
        'close': [snapshot['current_price']],
        'code': [ticker],
        'is_hot_data': [True]
    }, index=[today_date])
    
    # 合并数据
    if df_history.empty:
        df_combined = today_row
    else:
        df_combined = pd.concat([df_history, today_row])
        # 去重，保留最后出现的记录
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined = df_combined.sort_index()
    
    return df_combined


def show_data_confidence_dashboard():
    """
    数据置信度审计面板：显示时间轴完整度、基准对齐度、信噪比审计
    
    在Streamlit页面中调用此函数来显示数据置信度面板
    """
    import streamlit as st
    from config import load_config
    from storage_manager import get_storage_manager, get_data_audit_status
    
    # 加载配置
    config = load_config()
    
    # 获取master_pool中的股票
    master_pool = config.get('master_pool', {})
    all_codes = []
    for category, stocks in master_pool.items():
        for code, name in stocks.items():
            all_codes.append(code)
    
    if not all_codes:
        st.warning("配置文件中没有找到股票数据")
        return
    
    # 获取数据审计状态
    audit_status = get_data_audit_status(all_codes)
    
    # 计算审计指标
    total_stocks = len(audit_status)

    # 1. 时间轴完整度
    today = datetime.now().date()
    # ✅ 使用交易日判断：获取最近的交易日，而非简单的昨天
    from data_integrity_checker import get_last_trading_day
    last_trading_day = get_last_trading_day(today - timedelta(days=1))

    timeline_complete_count = 0
    for code, status in audit_status.items():
        last_date = status.get('last_cold_date')
        # ✅ 检查是否包含最近交易日或今天的数据
        if last_date and (last_date >= last_trading_day or last_date == today):
            timeline_complete_count += 1

    timeline_completeness = timeline_complete_count / total_stocks if total_stocks > 0 else 0
    
    # 2. 基准对齐度
    alignment_pass_count = 0
    for code, status in audit_status.items():
        status_str = status.get('status_bar', '')
        if '🟢' in status_str or ('🟡' in status_str and '数据完整' in status_str):
            alignment_pass_count += 1
    
    alignment_score = alignment_pass_count / total_stocks if total_stocks > 0 else 0
    
    # 3. 信噪比审计（脏数据清洗记录）
    # 暂时简化：检查是否有异常数据警告
    dirty_data_count = 0
    for code, status in audit_status.items():
        # 这里可以扩展为实际的异常检测逻辑
        # 暂时使用状态判断
        status_str = status.get('status_bar', '')
        if '🔴' in status_str:
            dirty_data_count += 1
    
    # 显示审计面板
    st.subheader("🔬 数据置信度审计面板")
    st.caption('循证医学风格的"数据报告"：确保每个$R^2$、每个突破信号都建立在真实、对齐的物理数据之上')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 时间轴完整度
        st.metric(
            label="📅 时间轴完整度",
            value=f"{timeline_completeness:.1%}",
            delta=f"{timeline_complete_count}/{total_stocks} 只股票",
            delta_color="normal" if timeline_completeness > 0.9 else "off"
        )
        st.caption("检查是否存在漏报的交易日")
    
    with col2:
        # 基准对齐度
        st.metric(
            label="🎯 基准对齐度",
            value="PASS" if alignment_score > 0.95 else "FAIL",
            delta=f"{alignment_pass_count}/{total_stocks} 只股票",
            delta_color="normal" if alignment_score > 0.95 else "off"
        )
        st.caption("昨收价与今昨收是否匹配")
    
    with col3:
        # 信噪比审计
        st.metric(
            label="📊 信噪比审计",
            value=f"{dirty_data_count} 个异常",
            delta="异常跳空数据点",
            delta_color="inverse" if dirty_data_count == 0 else "normal"
        )
        st.caption("是否存在跳空缺口超过20%的脏数据")
    
    # 详细状态表格
    with st.expander("📋 详细审计状态", expanded=False):
        # 准备表格数据
        table_data = []
        for code, status in list(audit_status.items())[:20]:  # 限制显示前20只
            # 找到股票名称
            name = code
            for category, stocks in master_pool.items():
                if code in stocks:
                    name = stocks[code]
                    break
            
            # 判断各项指标
            last_date = status.get('last_cold_date')
            # ✅ 使用交易日判断
            timeline_status = "✅" if last_date and (last_date >= last_trading_day or last_date == today) else "❌"
            
            status_str = status.get('status_bar', '')
            alignment_status = "✅" if '🟢' in status_str or ('🟡' in status_str and '数据完整' in status_str) else "❌"
            
            noise_status = "✅" if '🔴' not in status_str else "⚠️"
            
            table_data.append({
                "代码": code,
                "名称": name,
                "时间轴完整度": timeline_status,
                "基准对齐度": alignment_status,
                "信噪比审计": noise_status,
                "最后更新日期": last_date or "N/A"
            })
        
        if table_data:
            import pandas as pd
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True, hide_index=True)
        else:
            st.info("暂无审计数据")
    
    # 审计说明
    st.info("""
    **审计指标说明：**
    
    1. **时间轴完整度**：检查每只股票的本地数据是否包含最新交易日（昨日或今日）。  
       漏报的交易日会导致技术指标（如移动平均线、$R^2$拟合）失真。
    
    2. **基准对齐度**：验证本地最后收盘价与实时昨收价是否匹配（差异<0.001）。  
       不匹配通常表示发生除权除息或数据源调整，需触发强制重新同步。
    
    3. **信噪比审计**：检测异常跳空数据（涨跌幅>20%且无公告的离群点）。  
       这些"脏数据"会污染统计分析，需被拦截并记录。
    
    **循证价值**：通过三维握手校验，确保历史数据与实时数据在"缝合点"无缝对接，  
   为量化分析提供可靠的数据基础。
    """)
