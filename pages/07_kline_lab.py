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
from services.market_data_service import safe_fetch_candle_data
from ai_analyzer import deepseek_analyze
from backtester import generate_signal_series, run_vectorized_backtest, plot_backtest_results, plot_backtest_with_divergence, rsi_grid_search, grid_search_with_backtest, detect_rsi_divergence, plot_divergence_chart, calculate_rsi
from math_engine import fit_trend_line, detect_peaks_valleys
from font_utils import setup_chinese_font

# Plotly 图表配置 (符合 Streamlit 2026 标准)
PLOTLY_CONFIG = {
    'displayModeBar': True,    # 显示工具栏
    'displaylogo': False,      # 隐藏 Plotly Logo
    'modeBarButtonsToRemove': ['lasso2d'],  # 保留select2d框选工具
    'responsive': True         # 自适应容器
}

# 设置中文字体
font_name, font_path = setup_chinese_font()
if font_name:
    print(f"已使用中文字体: {font_name}")
else:
    print("使用默认字体配置")
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(page_title="独立K线分析！", layout="wide")

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

st.subheader("📈 独立 K 线分析")
st.caption("独立技术分析模块，手动选择标的并启动K线分析，不影响主程序状态。")

def render_independent_kline(stock_pool):
    st.subheader("🔍 个股独立技术分析 (手动模式)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_name = st.selectbox("选择分析标的", options=list(stock_pool.values()), key="ind_kline_select")
        code = [c for c, n in stock_pool.items() if n == selected_name][0]
    with col2:
        run_kline = st.checkbox("🚩 启动独立分析引擎", key="kline_active_flag")
    
    if run_kline:
        with st.spinner(f"正在调取 {selected_name} 全量历史数据..."):
            df = safe_fetch_candle_data(code)
            
            if df.empty:
                st.error(f"无法获取 {selected_name} 的K线数据，请稍后重试。")
            else:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # 创建基础K线图
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   vertical_spacing=0.03, row_width=[0.2, 0.8])
                
                fig.add_trace(go.Candlestick(
                    x=df['日期'], open=df['开盘'], high=df['最高'], low=df['最低'], close=df['收盘'],
                    name='Price', increasing_line_color='#ff4444', decreasing_line_color='#00cc00'
                ), row=1, col=1)
                
                ma60 = df['收盘'].rolling(window=60).mean()
                fig.add_trace(go.Scatter(x=df['日期'], y=ma60, line=dict(color='orange', width=1.5), name='MA60'), row=1, col=1)
                
                fig.add_trace(go.Bar(x=df['日期'], y=df['成交量'], marker_color='gray', opacity=0.5, name='Volume'), row=2, col=1)
                
                fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode='x unified')
                
                # 添加交互式框选提示和拟合模型选择
                st.info("📊 交互式趋势拟合分析：使用图表工具栏的框选工具（Box Select）选择一段区间，系统将自动绘制拟合线。")
                
                # 拟合模型选择器
                fit_degree = st.radio("拟合模型选择", [1, 2], 
                                     format_func=lambda x: "线性趋势 (速度)" if x==1 else "二次曲线 (动能加速度)", 
                                     horizontal=True, key="fit_degree_selector")
                
                # 渲染图表并捕获选择事件
                event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", config=PLOTLY_CONFIG)
                
                # 如果用户进行了框选
                if event and "selection" in event and len(event["selection"]["points"]) > 0:
                    # 获取选区索引范围
                    selected_points = event["selection"]["points"]
                    idx_start = selected_points[0]["point_index"]
                    idx_end = selected_points[-1]["point_index"]
                    df_slice = df.iloc[idx_start:idx_end+1]
                    
                    # 计算峰谷拟合 (传入 degree 参数)
                    peak_line, p_coeff, p_count = fit_trend_line(df_slice, 'peak', degree=fit_degree)
                    valley_line, v_coeff, v_count = fit_trend_line(df_slice, 'valley', degree=fit_degree)
                    
                    # 定义图例标签前缀和系数名称
                    prefix = "曲率a" if fit_degree == 2 else "斜率k"
                    coeff_name = "曲率" if fit_degree == 2 else "斜率"
                    
                    # 将拟合线叠加到原图
                    if peak_line is not None:
                        fig.add_trace(go.Scatter(
                            x=df_slice['日期'], y=peak_line, 
                            name=f"阻力曲线 ({prefix}:{p_coeff:.4f}, 点数:{p_count})",
                            line=dict(color='red', dash='dash', width=2)
                        ), row=1, col=1)
                    if valley_line is not None:
                        fig.add_trace(go.Scatter(
                            x=df_slice['日期'], y=valley_line, 
                            name=f"支撑曲线 ({prefix}:{v_coeff:.4f}, 点数:{v_count})",
                            line=dict(color='green', dash='dash', width=2)
                        ), row=1, col=1)
                    
                    # 重新渲染图表
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                    
                    # 显示拟合结果分析
                    st.subheader("📈 拟合结果分析")
                    if peak_line is not None:
                        if fit_degree == 2:
                            # 二次曲线模式：曲率分析
                            if p_coeff > 0:
                                st.success(f"阻力线曲率为正 ({p_coeff:.4f})，凹面向上(∪形)，表明向下的动能在衰减，或者向上的动能在增强。")
                            elif p_coeff < 0:
                                st.warning(f"阻力线曲率为负 ({p_coeff:.4f})，凹面向下(∩形)，表明向上的动能在衰减(见顶信号)，或者向下的动能在增强。")
                            else:
                                st.info(f"阻力线曲率接近零 ({p_coeff:.4f})，接近直线趋势，动能稳定。")
                        else:
                            # 线性模式：斜率分析
                            if p_coeff > 0:
                                st.success(f"阻力线斜率为正 ({p_coeff:.2f})，表明在选区内压力位呈上升趋势。")
                            else:
                                st.warning(f"阻力线斜率为负 ({p_coeff:.2f})，表明在选区内压力位呈下降趋势。")
                    
                    if valley_line is not None:
                        if fit_degree == 2:
                            # 二次曲线模式：曲率分析
                            if v_coeff > 0:
                                st.success(f"支撑线曲率为正 ({v_coeff:.4f})，凹面向上(∪形)，表明向下的动能在衰减，或者向上的动能在增强。")
                            elif v_coeff < 0:
                                st.warning(f"支撑线曲率为负 ({v_coeff:.4f})，凹面向下(∩形)，表明向上的动能在衰减(见顶信号)，或者向下的动能在增强。")
                            else:
                                st.info(f"支撑线曲率接近零 ({v_coeff:.4f})，接近直线趋势，动能稳定。")
                        else:
                            # 线性模式：斜率分析
                            if v_coeff > 0:
                                st.success(f"支撑线斜率为正 ({v_coeff:.2f})，表明在选区内支撑位呈上升趋势。")
                            else:
                                st.warning(f"支撑线斜率为负 ({v_coeff:.2f})，表明在选区内支撑位呈下降趋势。")
                    
                    # 平行通道判定
                    coeff_diff = 0.0
                    if peak_line is not None and valley_line is not None:
                        coeff_diff = abs(p_coeff - v_coeff)
                        if fit_degree == 2:
                            # 二次曲线：曲率相近判定
                            if coeff_diff < 0.01:  # 曲率相近
                                st.info("阻力线与支撑线曲率相近，形成平行弯曲通道，适合进行高抛低吸策略。")
                            else:
                                st.info("阻力线与支撑线曲率差异明显，表明弯曲程度不同，动能变化不对称。")
                        else:
                            # 线性：斜率相近判定
                            if coeff_diff < 0.1:  # 斜率相近
                                st.info("阻力线与支撑线斜率相近，形成平行通道，适合进行高抛低吸策略。")
                            else:
                                st.info("阻力线与支撑线斜率差异明显，表明趋势通道正在扩大或收缩。")
                    
                    # 突破判定
                    latest_close = None
                    latest_peak = None
                    if not df_slice.empty and peak_line is not None:
                        latest_close = df_slice['收盘'].iloc[-1]
                        latest_peak = peak_line[-1]
                        if latest_close > latest_peak:
                            st.success(f"最新收盘价 {latest_close:.2f} 已突破阻力线 {latest_peak:.2f}，可能形成突破信号。")
                        
                    # 二次曲线模式下的额外分析：趋势加速/减速
                    if fit_degree == 2 and peak_line is not None:
                        if p_coeff > 0:
                            st.info("阻力线曲率为正，凹面向上，突破后上涨动能可能加速。")
                        elif p_coeff < 0:
                            st.warning("阻力线曲率为负，凹面向下，突破后上涨动能可能减速。")
                    
                    # AI深度分析
                    st.subheader("🤖 AI深度分析")
                    if ds_key:
                        if st.button("🔍 启动AI深度分析", type="secondary", key="ai_fit_analysis"):
                            with st.spinner("AI正在分析拟合结果，请稍候..."):
                                try:
                                    from openai import OpenAI
                                    client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
                                    
                                    # 构建分析提示词 - 修复格式化字符串错误
                                    model_type = "二次曲线（加速度）" if fit_degree == 2 else "线性（速度）"
                                    coeff_unit = "曲率a" if fit_degree == 2 else "斜率k"
                                    
                                    # 格式化系数显示
                                    p_coeff_formatted = f"{p_coeff:.4f}" if fit_degree == 2 else f"{p_coeff:.2f}"
                                    v_coeff_formatted = f"{v_coeff:.4f}" if fit_degree == 2 else f"{v_coeff:.2f}"
                                    coeff_diff_formatted = f"{coeff_diff:.4f}" if fit_degree == 2 else f"{coeff_diff:.2f}"
                                    
                                    # 突破信号
                                    if not df_slice.empty and peak_line is not None:
                                        breakthrough_status = "已突破阻力线" if latest_close > latest_peak else "未突破阻力线"
                                        breakthrough_text = f"最新收盘价{breakthrough_status}"
                                    else:
                                        breakthrough_text = "未检测到突破信号"
                                    
                                    analysis_prompt = f"""作为资深金融分析师，请对以下股票技术分析拟合结果进行深度解读：
                                    
标的股票：{selected_name}
拟合模型：{model_type}
阻力线核心系数：{p_coeff_formatted} ({coeff_unit})
支撑线核心系数：{v_coeff_formatted} ({coeff_unit})
系数差异：{coeff_diff_formatted}
{breakthrough_text}
选区数据范围：{len(df_slice)}个交易日

请基于以上技术指标，提供：
1. 技术形态解读（50字内）
2. 多空动能评估（30字内）
3. 具体操作建议（20字内）
4. 风险提示（20字内）"""
                                    
                                    response = client.chat.completions.create(
                                        model="deepseek-chat",
                                        messages=[
                                            {"role": "system", "content": "你是资深金融分析师，擅长技术分析，用简洁专业的中文回答。"},
                                            {"role": "user", "content": analysis_prompt}
                                        ],
                                        temperature=0.3,
                                        max_tokens=500
                                    )
                                    
                                    ai_analysis = response.choices[0].message.content
                                    
                                    # 显示分析结果
                                    st.success("AI分析完成！")
                                    st.markdown(f"**🤖 AI分析结果：**")
                                    st.info(ai_analysis)
                                    
                                except Exception as e:
                                    st.error(f"AI分析失败：{e}")
                    else:
                        st.warning("⚠️ 如需AI深度分析，请在侧边栏配置DeepSeek API密钥。")

                st.caption(f"提示：当前正在对 {selected_name} 进行技术面独立审计。")

if analysis_stocks:
    stock_pool = analysis_stocks
else:
    first_cat = list(st.session_state.master_pool.keys())[0]
    stock_pool = st.session_state.master_pool[first_cat]

render_independent_kline(stock_pool)
