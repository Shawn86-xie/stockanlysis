"""
统一数据服务模块：为所有页面提供统一、可靠的数据获取接口

此模块封装了数据获取、验证和缓存逻辑，确保所有页面的数据调用遵循相同的标准，
提高数据调用的合理性和一致性。

核心功能：
1. 统一的数据获取接口，支持历史数据、实时价格、K线数据等
2. 内置数据完整性验证，确保数据质量
3. 智能缓存管理，平衡性能与实时性
4. 错误处理和重试机制，提高系统鲁棒性
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import time
import logging

# 导入现有服务
from services.market_data_service import (
    safe_fetch_stock_data,
    safe_fetch_latest_prices,
    safe_fetch_candle_data,
    safe_get_signals,
    get_recent_10_days
)
from data_fetcher import (
    compute_portfolio_stats,
    monte_carlo_simulation,
    analyze_portfolio,
    calculate_deviation
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataServiceError(Exception):
    """数据服务异常"""
    pass


def get_analysis_data(stock_codes: list, stock_names: list, incremental_update: bool = False) -> pd.DataFrame:
    """
    获取分析数据（统一入口）
    
    此函数封装了数据获取的所有逻辑，包括：
    1. 检查缓存是否有效
    2. 获取历史数据（优先本地Parquet）
    3. 数据完整性验证
    4. 错误处理和重试
    
    Args:
        stock_codes: 股票代码列表
        stock_names: 股票名称列表
        incremental_update: 是否执行增量更新
    
    Returns:
        pd.DataFrame: 历史价格数据，索引为日期，列为股票名称
    
    Raises:
        DataServiceError: 当数据获取失败时抛出
    """
    # 生成缓存键
    cache_key = f"analysis_data_{'_'.join(sorted(stock_codes))}"
    
    # 检查缓存是否有效（15分钟）
    if cache_key in st.session_state:
        cache_data = st.session_state[cache_key]
        cache_time = cache_data.get('timestamp', datetime.min)
        cache_age = (datetime.now() - cache_time).total_seconds()
        
        # 缓存有效且不需要增量更新
        if cache_age < 900 and not incremental_update:  # 15分钟
            logger.info(f"使用缓存数据，年龄: {cache_age:.0f}秒")
            return cache_data['data']
    
    # 获取数据
    try:
        logger.info(f"开始获取 {len(stock_codes)} 只股票的历史数据")
        data = safe_fetch_stock_data(stock_codes, stock_names)
        
        if data.empty:
            raise DataServiceError("获取股票数据失败，返回空DataFrame")
        
        # 数据完整性验证
        missing_stocks = [name for name in stock_names if name not in data.columns]
        if missing_stocks:
            logger.warning(f"以下股票数据缺失: {missing_stocks}")
            # 移除缺失的股票
            data = data[[name for name in data.columns if name in stock_names]]
        
        # 检查数据长度
        if len(data) < 10:
            raise DataServiceError(f"数据长度不足，仅 {len(data)} 条记录")
        
        # 缓存数据
        st.session_state[cache_key] = {
            'data': data,
            'timestamp': datetime.now(),
            'codes': stock_codes,
            'names': stock_names
        }
        
        logger.info(f"数据获取成功，共 {len(data)} 条记录，{len(data.columns)} 只股票")
        return data
        
    except Exception as e:
        logger.error(f"获取分析数据失败: {e}")
        raise DataServiceError(f"数据获取失败: {str(e)}")


def get_portfolio_statistics(data: pd.DataFrame, risk_free_rate: float = 0.0188, sim_num: int = 3000):
    """
    获取投资组合统计信息（统一计算）
    
    Args:
        data: 历史价格数据
        risk_free_rate: 无风险利率
        sim_num: 蒙特卡洛模拟次数
    
    Returns:
        tuple: (returns, mean_ret, cov_mat, sim_res, best_p)
    """
    # 计算收益率
    returns = data.pct_change().dropna()
    
    if returns.empty:
        raise DataServiceError("无法计算收益率，数据不足")
    
    # 计算投资组合统计
    mean_ret, cov_mat = compute_portfolio_stats(returns, risk_free_rate)
    
    # 蒙特卡洛模拟
    sim_res = monte_carlo_simulation(mean_ret, cov_mat, sim_num, risk_free_rate)
    best_p = sim_res.iloc[sim_res['Sharpe'].idxmax()]
    
    return returns, mean_ret, cov_mat, sim_res, best_p


def get_latest_prices(stock_codes: list, stock_names: list, force_network: bool = False) -> pd.DataFrame:
    """
    获取最新价格（统一接口）
    
    Args:
        stock_codes: 股票代码列表
        stock_names: 股票名称列表
        force_network: 是否强制穿透缓存
    
    Returns:
        pd.DataFrame: 最新价格数据
    """
    return safe_fetch_latest_prices(stock_codes, stock_names, force_network)


def get_trading_signals(data: pd.DataFrame, best_weights: dict, stop_loss: float, 
                        latest_prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    获取交易信号（统一接口）
    
    Args:
        data: 历史价格数据
        best_weights: 最优权重字典
        stop_loss: 止损阈值
        latest_prices: 最新价格数据
    
    Returns:
        pd.DataFrame: 交易信号数据
    """
    return safe_get_signals(data, best_weights, stop_loss, latest_prices)


def get_candle_data(stock_code: str, days: int = 365) -> pd.DataFrame:
    """
    获取K线数据（统一接口）
    
    Args:
        stock_code: 股票代码
        days: 历史天数
    
    Returns:
        pd.DataFrame: K线数据
    """
    return safe_fetch_candle_data(stock_code, days)


def validate_data_integrity(data: pd.DataFrame, min_days: int = 10) -> dict:
    """
    数据完整性验证
    
    Args:
        data: 待验证的数据
        min_days: 最小交易日要求
    
    Returns:
        dict: 验证结果
    """
    results = {
        'is_valid': True,
        'missing_dates': [],
        'nan_columns': [],
        'insufficient_data': [],
        'messages': []
    }
    
    if data.empty:
        results['is_valid'] = False
        results['messages'].append('数据为空')
        return results
    
    # 检查日期连续性
    if isinstance(data.index, pd.DatetimeIndex):
        date_diff = data.index.to_series().diff().dropna()
        if len(date_diff) > 0:
            max_gap = date_diff.max()
            if max_gap > timedelta(days=7):
                results['messages'].append(f'检测到日期缺口: {max_gap.days}天')
    
    # 检查NaN值
    nan_columns = data.columns[data.isna().any()].tolist()
    if nan_columns:
        results['nan_columns'] = nan_columns
        results['messages'].append(f'以下列包含NaN值: {nan_columns}')
    
    # 检查数据长度
    for column in data.columns:
        if len(data[column].dropna()) < min_days:
            results['insufficient_data'].append(column)
    
    if results['insufficient_data']:
        results['messages'].append(f'以下股票数据不足{min_days}天: {results["insufficient_data"]}')
    
    # 综合判断
    if results['nan_columns'] or results['insufficient_data']:
        results['is_valid'] = False
    
    return results


def clear_data_cache():
    """
    清除数据缓存
    
    用于系统维护或数据重置后刷新缓存
    """
    cache_keys = [key for key in st.session_state.keys() if key.startswith('analysis_data_')]
    for key in cache_keys:
        del st.session_state[key]
    logger.info(f"已清除 {len(cache_keys)} 个数据缓存")


def get_data_audit_report(stock_codes: list, stock_names: list) -> dict:
    """
    生成数据审计报告
    
    Args:
        stock_codes: 股票代码列表
        stock_names: 股票名称列表
    
    Returns:
        dict: 审计报告
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'stocks_analyzed': len(stock_codes),
        'data_quality': {},
        'recommendations': []
    }
    
    try:
        # 获取数据
        data = get_analysis_data(stock_codes, stock_names)
        
        # 验证数据完整性
        validation = validate_data_integrity(data)
        
        report['data_quality'] = {
            'total_records': len(data),
            'total_columns': len(data.columns),
            'date_range': {
                'start': data.index.min().strftime('%Y-%m-%d') if len(data) > 0 else None,
                'end': data.index.max().strftime('%Y-%m-%d') if len(data) > 0 else None
            },
            'validation_results': validation
        }
        
        # 生成建议
        if not validation['is_valid']:
            report['recommendations'].append('数据完整性存在问题，建议重新获取数据')
        
        if len(data.columns) < len(stock_names):
            missing = set(stock_names) - set(data.columns)
            report['recommendations'].append(f'以下股票数据缺失: {list(missing)}')
        
        if len(data) < 60:
            report['recommendations'].append('历史数据不足60个交易日，可能影响技术指标计算')
        
    except Exception as e:
        report['error'] = str(e)
        report['recommendations'].append('数据获取失败，请检查网络连接或股票代码')
    
    return report


# 快捷函数，便于页面调用
def initialize_page_data(analysis_stocks: dict, config: dict, user_settings: dict):
    """
    页面数据初始化（统一入口）
    
    此函数封装了页面初始化所需的所有数据获取逻辑，
    确保所有页面使用相同的初始化流程。
    
    Args:
        analysis_stocks: 股票字典 {code: name}
        config: 配置字典
        user_settings: 用户设置字典
    
    Returns:
        dict: 包含所有必要数据的字典
    """
    # 提取股票代码和名称
    stock_codes = list(analysis_stocks.keys())
    stock_names = list(analysis_stocks.values())
    
    # 获取分析数据
    data = get_analysis_data(stock_codes, stock_names)
    
    # 获取投资组合统计
    risk_free_rate = config.get("risk_free_rate", 0.0188)
    sim_num = user_settings.get('sim_num', 3000)
    returns, mean_ret, cov_mat, sim_res, best_p = get_portfolio_statistics(
        data, risk_free_rate, sim_num
    )
    
    # 获取最新价格
    latest_prices = get_latest_prices(stock_codes, stock_names)
    
    # 生成交易信号
    stop_loss = user_settings.get('stop_loss', -0.05)
    signals = get_trading_signals(data, best_p.to_dict(), stop_loss, latest_prices)
    
    return {
        'data': data,
        'returns': returns,
        'mean_ret': mean_ret,
        'cov_mat': cov_mat,
        'sim_res': sim_res,
        'best_p': best_p,
        'latest_prices': latest_prices,
        'signals': signals,
        'stock_codes': stock_codes,
        'stock_names': stock_names,
        'analysis_stocks': analysis_stocks
    }
