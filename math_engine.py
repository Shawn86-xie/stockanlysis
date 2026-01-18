"""
FILE: math_engine.py
ROLE: 数学核心引擎，计算趋势线拟合、拟合优度 R²、动能斜率 k、量能强度等量化指标，支持多因子权重调优。
LOGIC: 
    1. 使用多项式拟合技术分析价格趋势线（支撑线和压力线）。
    2. 计算拟合优度 R² 评估趋势线的可靠性。
    3. 结合量能强度和距离上轨距离等因子，计算综合评分。
    4. 支持多因子权重调优，适应不同市场风格。
DEPENDENCIES: numpy, scipy.signal.find_peaks
"""

import numpy as np
from scipy.signal import find_peaks


def fit_trend_line(df_slice, type='peak', degree=2):
    """
    对选定片段进行峰谷提取并拟合曲线
    type: 'peak' 代表压力线, 'valley' 代表支撑线
    degree: 1为线性(速度), 2为二次曲线(加速度)
    """
    prices = df_slice['最高'].values if type == 'peak' else df_slice['最低'].values
    # 使用自然序号作为 X 轴用于拟合
    x_axis = np.arange(len(prices))
    
    # 1. 寻找局部极值点 (使用 scipy 提升科学性)
    if type == 'peak':
        indices, _ = find_peaks(prices, distance=5)
    else:
        indices, _ = find_peaks(-prices, distance=5)
        
    # 二次拟合至少需要3个点，线性需要2个点
    min_points = degree + 1
    if len(indices) < min_points:
        return None, None, None  # 点数不足无法拟合

    # 2. 准备拟合数据
    x_fit = indices
    y_fit = prices[indices]
    
    # 3. 执行多项式拟合 (关键修改)
    # np.polyfit 返回的是系数数组，从最高次幂开始 [a, b, c]
    coeffs = np.polyfit(x_fit, y_fit, degree)
    
    # 构建拟合函数
    poly_func = np.poly1d(coeffs)
    
    # 生成整段区间的拟合曲线数据
    y_pred = poly_func(x_axis)
    
    # 获取核心系数：如果是二次曲线，取第一个系数 'a' (曲率)
    # 如果是线性，取第一个系数 'b' (斜率)
    core_coeff = coeffs[0]
    
    return y_pred, core_coeff, len(indices)  # 返回拟合线、核心系数和点数


def detect_peaks_valleys(df_slice):
    """
    检测选定区间的局部峰谷点，返回峰点和谷点的索引和价格
    """
    high_prices = df_slice['最高'].values
    low_prices = df_slice['最低'].values
    
    # 寻找局部峰值（压力点）
    peak_indices, _ = find_peaks(high_prices, distance=5)
    # 寻找局部谷值（支撑点）
    valley_indices, _ = find_peaks(-low_prices, distance=5)
    
    peaks = {
        'indices': peak_indices,
        'prices': high_prices[peak_indices] if len(peak_indices) > 0 else np.array([])
    }
    valleys = {
        'indices': valley_indices,
        'prices': low_prices[valley_indices] if len(valley_indices) > 0 else np.array([])
    }
    
    return peaks, valleys


def calculate_screening_score(df, window=30, weights=None):
    """
    标的筛选评分核心函数（支持多因子权重调优）
    输入: 
        df: 包含 'close' 和 'volume' 的 DataFrame
        window: 分析窗口长度
        weights: 权重字典，格式如 {'k': 0.4, 'r2': 0.3, 'dist': 0.2, 'vol': 0.1}
                默认使用均衡稳健型: k:0.35, r2:0.35, dist:0.20, vol:0.10
    输出: 包含所有中间指标和最终评分的字典
    """
    # 1. 输入处理：接收包含 'close' 和 'volume' 的 DataFrame
    if 'close' not in df.columns or 'volume' not in df.columns:
        raise ValueError("DataFrame must contain 'close' and 'volume' columns")
    
    # 设置默认权重（均衡稳健型）
    if weights is None:
        weights = {'k': 0.35, 'r2': 0.35, 'dist': 0.20, 'vol': 0.10}
    
    # 取最近 window 天的数据
    df_slice = df.tail(window).copy()
    
    # 为了使用现有函数，需要创建 '最高' 和 '最低' 列（都使用 close）
    df_slice['最高'] = df_slice['close']
    df_slice['最低'] = df_slice['close']
    
    # 2. 趋势拟合：调用 fit_trend_line 拟合阻力线（使用线性拟合 degree=1）
    resistance_line, k, point_count = fit_trend_line(df_slice, type='peak', degree=1)
    
    # 如果拟合失败（点数不足），返回0分和空指标
    if resistance_line is None or k is None:
        return {
            'k': 0,
            'k_rel': 0,
            'r_squared': 0,
            'dist_to_upper': 0,
            'dist_weight': 0,
            'vol_intensity': 0,
            'vol_weight': 0,
            'final_score': 0
        }
    
    # 3. 计算 R²（拟合优度）
    # 获取峰值点索引和价格
    high_prices = df_slice['最高'].values
    peak_indices, _ = find_peaks(high_prices, distance=5)
    
    if len(peak_indices) < 2:  # 至少需要2个点才能计算 R²
        r_squared = 0
    else:
        # 实际峰值点价格
        y_actual = high_prices[peak_indices]
        # 拟合线在峰值点的值
        y_predicted = resistance_line[peak_indices]
        # 计算 R²
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 4. 核心指标计算（原始指标）
    # 归一化斜率 (Relative Slope)
    price_mean = np.mean(df_slice['close'].values)
    k_rel = k / price_mean if price_mean != 0 else 0
    
    # 距上轨距离 (Dist to Upper)
    latest_close = df_slice['close'].iloc[-1]
    latest_resistance = resistance_line[-1]
    dist_to_upper = (latest_resistance - latest_close) / latest_close if latest_close != 0 else 0
    
    # 距离权重：距离越靠近上轨（但在2%以内）权重越高
    if 0 <= dist_to_upper <= 0.02:
        dist_weight = 1 - (dist_to_upper / 0.02)  # 距离为0时权重1，距离2%时权重0
    else:
        dist_weight = 0
    
    # 量能强度 (Vol Intensity)
    volume_series = df_slice['volume'].values
    if len(volume_series) >= 5:
        vol_intensity = volume_series[-1] / np.mean(volume_series[-5:]) if np.mean(volume_series[-5:]) != 0 else 0
    else:
        vol_intensity = 0
    
    # 量能权重：将强度归一化到0-1之间，限制最大强度为2
    vol_weight = min(vol_intensity, 2) / 2
    
    # 5. 指标标准化（Min-Max 标准化，防止某个指标因数值过大而主导结果）
    # 注意：这里我们只对原始指标进行标准化，而不是对权重进行标准化
    # 实际应用中，可能需要跨股票进行标准化，但这里我们仅对当前股票的指标进行归一化到0-1区间
    # 由于各指标本身已经在合理范围内，我们跳过跨股票标准化，仅使用原始指标
    
    # 6. 综合评分逻辑（使用传入的权重）
    # 如果 k < 0（非上升趋势），基准分为0
    if k < 0:
        final_score = 0
    else:
        # 综合得分 = k_rel * w_k + r_squared * w_r2 + dist_weight * w_dist + vol_weight * w_vol
        final_score = (k_rel * weights.get('k', 0.35) + 
                      r_squared * weights.get('r2', 0.35) + 
                      dist_weight * weights.get('dist', 0.20) + 
                      vol_weight * weights.get('vol', 0.10))
    
    # 7. 返回结果
    return {
        'k': k,
        'k_rel': k_rel,
        'r_squared': r_squared,
        'dist_to_upper': dist_to_upper,
        'dist_weight': dist_weight,
        'vol_intensity': vol_intensity,
        'vol_weight': vol_weight,
        'final_score': final_score,
        'weights_used': weights  # 返回使用的权重，便于调试
    }
