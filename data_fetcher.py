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
            df = df[['日期', '收盘']].rename(columns={'日期': 'Date', '收盘': name})
            df['Date'] = pd.to_datetime(df['Date'])
            dfs.append(df.set_index('Date'))
        except Exception as e:
            print(f"获取 {code}({name}) 数据失败: {e}")
            continue
    return pd.concat(dfs, axis=1).dropna() if dfs else pd.DataFrame()

def get_signals(data, best_weights, stop_loss):
    """
    生成交易信号与风险提醒
    Args:
        data: 股票价格 DataFrame
        best_weights: 最优权重字典 {股票名称: 权重}
        stop_loss: 止损阈值（负数）
    Returns:
        DataFrame 包含各标的信号
    """
    signals = []
    for name in data.columns:
        prices = data[name]
        curr_p = prices.iloc[-1]
        prev_p = prices.iloc[-2]
        daily_change = (curr_p - prev_p) / prev_p
        
        ma60 = prices.rolling(window=60).mean().iloc[-1]
        # RSI 简易计算
        delta = prices.diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss).iloc[-1]))
        
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
        data: DataFrame，索引为日期，列为股票名称
    Returns:
        DataFrame，最近10个交易日的数据
    """
    return data.tail(10)

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
