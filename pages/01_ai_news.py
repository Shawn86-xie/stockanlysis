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
    DataServiceError
)
from ai_analyzer import deepseek_analyze
from backtester import generate_signal_series, run_vectorized_backtest, plot_backtest_results, plot_backtest_with_divergence, rsi_grid_search, grid_search_with_backtest, detect_rsi_divergence, plot_divergence_chart, calculate_rsi
from math_engine import fit_trend_line, detect_peaks_valleys
from font_utils import setup_chinese_font
from services.news_service import batch_fetch_and_analyze, NewsServiceError

# 设置中文字体
font_name, font_path = setup_chinese_font()
if font_name:
    print(f"已使用中文字体: {font_name}")
else:
    print("使用默认字体配置")
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(page_title="AI资讯深度研判", layout="wide")

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
st.subheader("DeepSeek专业资讯评分")
st.info("AI资讯分析需要调用外部API，耗时较长，请手动点击按钮获取。")

# 初始化session_state存储新闻和分析结果
if 'news_results' not in st.session_state:
    st.session_state.news_results = {}

# 检查 API Key 配置
if not ds_key:
    st.warning("⚠️ 未配置 DeepSeek API Key，无法进行 AI 分析。请在 config.json 中配置 deepseek_api_key。")

# 按钮触发获取新闻
if st.button("🔍 获取最新新闻并分析", type="primary", disabled=not ds_key):
    with st.spinner("正在获取新闻并分析..."):
        try:
            # 调用新闻服务批量获取并分析新闻
            results = batch_fetch_and_analyze(analysis_stocks, news_count, ds_key)

            # 将结果存储到 session_state
            for stock_code, result in results.items():
                st.session_state.news_results[stock_code] = result

            # 显示成功消息
            if results:
                st.success(f"✅ 成功获取并分析了 {len(results)} 个股票的新闻")
            else:
                st.warning("⚠️ 未获取到任何新闻数据")

        except NewsServiceError as e:
            st.error(f"新闻分析服务出错：{e}")
        except Exception as e:
            st.error(f"未知错误：{e}")
            import traceback
            st.error(f"详细错误信息：\n{traceback.format_exc()}")

# 显示已存储的结果
for c, news_entry in st.session_state.news_results.items():
    if news_entry['name'] in analysis_stocks.values():  # 只显示当前选中的标的
        with st.expander(f"📌 {news_entry['name']} ({c}) - 资讯洞察"):
            for item in news_entry['news']:
                # 使用两列布局：左侧标题和摘要，右侧评分和日期
                col_left, col_right = st.columns([3, 1])
                with col_left:
                    st.markdown(f"#### {item['title']}")
                    st.caption(f"**发布日期**: {item['date']}")
                    st.write(f"**摘要**: {item['summary']}")
                    st.write(f"**分析**: {item['reason']}")
                    if item['url']:
                        st.caption(f"[原文链接]({item['url']})")
                with col_right:
                    st.markdown(f":{item['color']}[{item['stars']}]")
                    st.caption(f"**性质**: {item['nature']}")
