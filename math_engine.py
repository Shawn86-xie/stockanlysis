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