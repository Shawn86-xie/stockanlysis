import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 导入自定义模块
from config import load_config
from services.data_service import (
    initialize_page_data,
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
st.set_page_config(page_title="模拟交易", layout="wide")

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

# 更新selected_names为实际的股票名称列表
selected_names = list(analysis_stocks.values())

# 页面内容开始
st.subheader("📈 模拟交易 - 持仓分析与调仓建议")
st.write("请在下方输入您当前的持仓数量（股），系统将根据最优权重给出调仓建议。")

# 初始化持仓输入
holdings = {}
col1, col2, col3 = st.columns(3)
selected_names = list(analysis_stocks.values())
for idx, name in enumerate(selected_names):
    # 每列放置几个输入框
    with [col1, col2, col3][idx % 3]:
        holdings[name] = st.number_input(
            f"{name} 持仓数量",
            min_value=0,
            value=0,
            step=100,
            key=f"hold_{name}"
        )

if st.button("分析持仓并生成调仓建议", type="primary"):
    if sum(holdings.values()) == 0:
        st.warning("您还没有输入任何持仓，请输入持仓数量。")
    else:
        # 检查best_p中是否存在所有选中的股票
        valid_names = [name for name in selected_names if name in best_p.index]
        missing_names = [name for name in selected_names if name not in best_p.index]
        
        if missing_names:
            st.warning(f"以下股票在最优组合计算中被排除（可能由于数据缺失）：{', '.join(missing_names)}")
        
        if len(valid_names) == 0:
            st.error("没有有效的股票权重数据，无法生成调仓建议。请检查数据获取情况。")
            st.stop()
        
        # 只使用有效的股票名称
        best_weights_dict = best_p[valid_names].to_dict()
        
        # 确保data只包含有效股票
        data_valid = data[valid_names].copy() if len(valid_names) < len(data.columns) else data
        
        # 调用分析函数
        analysis_df, total_value = analyze_portfolio(holdings, data_valid, best_weights_dict, stop_loss_val)
        
        st.metric("持仓总市值", f"{total_value:,.2f} 元")
        
        st.subheader("持仓详情与调仓建议")
        st.dataframe(analysis_df)
        
        # 可视化当前权重与建议权重对比
        st.subheader("权重对比")
        compare_df = pd.DataFrame({
            '标的': analysis_df['标的'],
            '当前权重': analysis_df['当前权重'].str.rstrip('%').astype('float') / 100,
            '建议权重': analysis_df['建议权重'].str.rstrip('%').astype('float') / 100
        })
        compare_df = compare_df.set_index('标的')
        st.bar_chart(compare_df * 100)  # 转换为百分比显示
        
        # 总结调仓动作
        st.subheader("📋 调仓操作总结")
        for _, row in analysis_df.iterrows():
            if row['调仓建议'] != '保持':
                st.info(f"{row['标的']}: {row['调仓建议']}")
