import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression


def fit_trend_line(df_slice, type='peak'):
    """
    对选定片段进行峰谷提取并拟合直线
    type: 'peak' 代表压力线, 'valley' 代表支撑线
    """
    prices = df_slice['最高'].values if type == 'peak' else df_slice['最低'].values
    x_axis = np.arange(len(prices)).reshape(-1, 1)
    
    # 1. 寻找局部极值点 (使用 scipy 提升科学性)
    if type == 'peak':
        indices, _ = find_peaks(prices, distance=5)
    else:
        indices, _ = find_peaks(-prices, distance=5)
        
    if len(indices) < 2:
        return None, None  # 点数不足无法拟合

    # 2. 准备拟合数据
    x_fit = indices.reshape(-1, 1)
    y_fit = prices[indices]
    
    # 3. 执行线性回归
    model = LinearRegression().fit(x_fit, y_fit)
    y_pred = model.predict(x_axis)  # 生成拟合线序列
    
    return y_pred, model.coef_[0]  # 返回拟合线和斜率


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