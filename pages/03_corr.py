import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 认证保护
from auth import protect_page
protect_page()

# 导入自定义模块
from config import load_config
from services.data_service import (
    initialize_page_data,
    get_recent_10_days,
    DataServiceError
)
from ai_analyzer import deepseek_analyze
from backtester import generate_signal_series, run_vectorized_backtest, plot_backtest_results, plot_backtest_with_divergence, rsi_grid_search, grid_search_with_backtest, detect_rsi_divergence, plot_divergence_chart, calculate_rsi
from math_engine import fit_trend_line, detect_peaks_valleys
from font_utils import setup_chinese_font

# 设置中文字体
font_name, font_path = setup_chinese_font()
if font_name:
    print(f"已使用中文字体: {font_name}")
else:
    print("使用默认字体配置")
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(page_title="板块相关性分析", layout="wide")

# 检查是否有选中的标的
if 'selected_stocks' not in st.session_state or not st.session_state.selected_stocks:
    st.warning("请先返回主页面选择标的并开始分析。")
    st.stop()

# 获取必要的变量
analysis_stocks = st.session_state.selected_stocks
config = load_config()
ds_key = config.get("deepseek_api_key", "")
news_count = config.get("news_count", 5)
stop_loss_val = st.session_state.user_settings.get('stop_loss', -0.05) if 'user_settings' in st.session_state else -0.05
sim_num = st.session_state.user_settings.get('sim_num', 3000) if 'user_settings' in st.session_state else 3000

# 数据接力函数：检查session_state中是否已有现成数据，避免重复初始化
def get_ready_data():
    """
    数据接力：如果主控台已经加载了数据，直接使用；否则调用服务层初始化。
    返回包含所有必要数据的字典。
    """
    # 检查内存中是否已有完整数据（接力模式）
    has_complete_data = ('analysis_data' in st.session_state and
                         'returns' in st.session_state and
                         'best_p' in st.session_state)

    if has_complete_data:
        st.success("✅ 使用主控台已加载的数据（数据接力模式）")
        return {
            'data': st.session_state.analysis_data,
            'returns': st.session_state.returns,
            'mean_ret': st.session_state.get('mean_ret', None),
            'cov_mat': st.session_state.get('cov_mat', None),
            'sim_res': st.session_state.get('sim_res', None),
            'best_p': st.session_state.best_p
        }

    # 如果没有现成数据，调用服务层初始化
    with st.spinner("从中心库调阅病历中..."):
        try:
            page_data = initialize_page_data(analysis_stocks, config, st.session_state.user_settings)

            # 保存到session_state，供后续使用
            st.session_state.analysis_data = page_data['data']
            st.session_state.returns = page_data['returns']
            st.session_state.mean_ret = page_data.get('mean_ret')
            st.session_state.cov_mat = page_data.get('cov_mat')
            st.session_state.sim_res = page_data.get('sim_res')
            st.session_state.best_p = page_data['best_p']

            return page_data

        except DataServiceError as e:
            st.error(f"数据初始化失败: {e}")
            st.stop()
        except Exception as e:
            st.error(f"未知错误: {e}")
            st.stop()

# 获取数据（使用接力模式）
page_data = get_ready_data()

# 提取数据
data = page_data['data']
returns = page_data['returns']
mean_ret = page_data.get('mean_ret')
cov_mat = page_data.get('cov_mat')
sim_res = page_data.get('sim_res')
best_p = page_data['best_p']

# 页面内容开始
st.subheader("资产相关性矩阵 (防范共振风险)")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(returns.corr(), annot=True, cmap='RdYlGn', center=0, ax=ax_corr)
st.pyplot(fig_corr)
