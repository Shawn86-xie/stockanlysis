import pandas as pd
import akshare as ak
import numpy as np
from datetime import datetime, timedelta

def fetch_stock_data(codes, names):
    """
    获取股票历史数据
    Args:
        codes: 股票代码列表
        names: 股票名称列表
    Returns:
        DataFrame，列为股票名称，索引为日期
    """
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

def get_signals(data, best_weights, stop_loss):
    """
    生成交易信号与风险提醒
    Args:
        data: 股票价格 DataFrame 或字典
        best_weights: 最优权重字典 {股票名称: 权重}
        stop_loss: 止损阈值（负数）
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
                
            # 使用pandas的pct_change计算日涨跌，更安全
            daily_change = prices.pct_change().iloc[-1]
            if pd.isna(daily_change):
                daily_change = 0
                
            curr_p = float(prices.iloc[-1])
            prev_p = float(prices.iloc[-2])
            
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
