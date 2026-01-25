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
st.set_page_config(page_title="权重优化实验", layout="wide")

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
# 过滤可用的股票：只保留在best_p中存在的股票（即在data中成功获取的股票）
available_stocks = [name for name in analysis_stocks.values() if name in best_p.index]
if not available_stocks:
    st.error("没有可用的股票数据进行分析。")
    st.stop()

col_l, col_r = st.columns(2)
with col_l:
    st.metric("预期年化收益", f"{best_p['Ret']:.2%}")
    st.caption("📈 **预期年化收益**：基于历史数据计算的该投资组合在未来一年的预期收益率，已考虑复利效应。")
    fig_pie, ax_pie = plt.subplots()
    best_p[available_stocks].plot.pie(autopct='%1.1f%%', ax=ax_pie, cmap='Pastel1', title="AI 推荐配比")
    st.pyplot(fig_pie)
    st.caption("🥧 **AI推荐配比**：通过蒙特卡洛模拟找到的夏普比率最高的投资组合权重分配。")
with col_r:
    st.metric("夏普比率 (性价比)", f"{best_p['Sharpe']:.2f}")
    st.caption("⚖️ **夏普比率**：衡量每承担一单位风险所获得的超额收益，数值越高表示风险调整后收益越好。")
    
    # 显示前10个优选组合
    with st.expander("📊 查看前10个优选组合", expanded=False):
        st.caption("""
        **指标解释**：
        - **排名**：按夏普比率从高到低排序
        - **Ret**：预期年化收益率
        - **Vol**：预期年化波动率（风险）
        - **Sharpe**：夏普比率（风险调整后收益）
        - **各股票列**：在该组合中的权重分配
        """)
        # 按夏普比率降序排序，取前10
        top10 = sim_res.sort_values('Sharpe', ascending=False).head(10).copy()
        # 重置索引并添加排名
        top10 = top10.reset_index(drop=True)
        top10.insert(0, '排名', range(1, len(top10) + 1))
        
        # 格式化显示
        display_df = top10[['排名', 'Ret', 'Vol', 'Sharpe'] + available_stocks].copy()
        # 将收益和波动率转换为百分比字符串
        display_df['Ret'] = display_df['Ret'].apply(lambda x: f"{x:.2%}")
        display_df['Vol'] = display_df['Vol'].apply(lambda x: f"{x:.2%}")
        display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.2f}")
        # 将各股票权重转换为百分比字符串
        for name in available_stocks:
            display_df[name] = display_df[name].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_df, use_container_width=True)
        st.caption(f"共模拟 {sim_num} 次，展示了夏普比率最高的10个组合")
    
    fig_ef, ax_ef = plt.subplots()
    ax_ef.scatter(sim_res.Vol, sim_res.Ret, c=sim_res.Sharpe, cmap='viridis', alpha=0.3)
    ax_ef.scatter(best_p['Vol'], best_p['Ret'], color='red', marker='*', s=200)
    st.pyplot(fig_ef)
    st.caption("📊 **有效前沿图**：每个点代表一个随机投资组合，红色星号标记夏普比率最高的组合。")
