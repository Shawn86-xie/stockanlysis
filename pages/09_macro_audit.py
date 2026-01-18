import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import json

# 添加项目根目录到路径，以便导入macro_fetcher
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '..')))
from macro_fetcher import fetch_china_epu, fetch_ivix_data, fetch_margin_data
from ai_analyzer import analyze_macro_environment

# Plotly 图表配置 (符合 Streamlit 2026 标准)
PLOTLY_CONFIG = {
    'displayModeBar': True,    # 显示工具栏
    'displaylogo': False,      # 隐藏 Plotly Logo
    'modeBarButtonsToRemove': ['lasso2d'],  # 保留select2d框选工具
    'responsive': True         # 自适应容器
}

# 设置页面配置
st.set_page_config(
    page_title="宏观压力审计",
    page_icon="📊",
    layout="wide"
)

# 页面标题
st.title("📊 宏观压力审计")
st.markdown("---")

# 加载数据
@st.cache_data(ttl=3600)
def load_all_data():
    """加载所有宏观数据"""
    epu_data = fetch_china_epu()
    ivix_data = fetch_ivix_data()
    margin_data = fetch_margin_data()
    return epu_data, ivix_data, margin_data

# 显示加载状态
with st.spinner("正在加载宏观数据..."):
    epu_df, ivix_df, margin_df = load_all_data()

# ==================== 宏观指标每日自动审计 ====================
# 检查并更新今日宏观指标到数据库
from database_manager import get_database_manager

db_manager = get_database_manager()

# 检查今日宏观指标是否已更新
today_indicators = db_manager.check_macro_data_today(["EPU", "iVIX", "MarginBalance"])
need_update = []

for indicator, exists in today_indicators.items():
    if not exists:
        need_update.append(indicator)

# 如果有指标需要更新，执行更新
if need_update:
    with st.sidebar.expander("🔍 宏观指标自动审计", expanded=False):
        st.sidebar.info(f"检测到{len(need_update)}个指标需要更新: {', '.join(need_update)}")
        
        # 更新EPU指标
        if "EPU" in need_update and not epu_df.empty:
            try:
                current_epu = epu_df['EPU_Index'].iloc[-1]
                epu_percentile = epu_df['Percentile'].iloc[-1]
                db_manager.save_macro_indicator("EPU", current_epu, percentile=epu_percentile)
                st.sidebar.success("✓ EPU指标已更新")
            except Exception as e:
                st.sidebar.error(f"EPU指标更新失败: {e}")
        
        # 更新iVIX指标
        if "iVIX" in need_update and not ivix_df.empty:
            try:
                current_ivix = ivix_df['ivix'].iloc[-1]
                # 计算iVIX分位（过去一年）
                ivix_last_year = ivix_df['ivix'].tail(252) if len(ivix_df) >= 252 else ivix_df['ivix']
                ivix_percentile = (ivix_last_year < current_ivix).sum() / len(ivix_last_year) if len(ivix_last_year) > 0 else 0.5
                db_manager.save_macro_indicator("iVIX", current_ivix, percentile=ivix_percentile)
                st.sidebar.success("✓ iVIX指标已更新")
            except Exception as e:
                st.sidebar.error(f"iVIX指标更新失败: {e}")
        
        # 更新融资余额指标
        if "MarginBalance" in need_update and not margin_df.empty:
            try:
                current_margin = margin_df['fin_balance'].iloc[-1]
                # 计算融资余额分位（过去一年）
                margin_last_year = margin_df['fin_balance'].tail(252) if len(margin_df) >= 252 else margin_df['fin_balance']
                margin_percentile = (margin_last_year < current_margin).sum() / len(margin_last_year) if len(margin_last_year) > 0 else 0.5
                db_manager.save_macro_indicator("MarginBalance", current_margin, percentile=margin_percentile)
                st.sidebar.success("✓ 融资余额指标已更新")
            except Exception as e:
                st.sidebar.error(f"融资余额指标更新失败: {e}")
        
        # 显示数据审计状态
        st.sidebar.caption(f"数据审计时间: {datetime.now().strftime('%H:%M:%S')}")

# ==================== 数据审计侧边栏 ====================
st.sidebar.header("📂 数据审计面板")
st.sidebar.markdown("**循证透明，三层校验**")

# 导入数据置信度审计功能
try:
    from data_fetcher import show_data_confidence_dashboard
    
    # 直接在侧边栏显示数据置信度审计面板
    with st.sidebar.expander("🔬 数据置信度审计", expanded=False):
        show_data_confidence_dashboard()
        
    # 提供刷新按钮
    if st.sidebar.button("🔄 刷新数据审计", key="refresh_audit"):
        st.cache_data.clear()
        st.rerun()
        
except Exception as e:
    st.sidebar.error(f"数据置信度审计加载失败: {str(e)}")

# ==================== 顶部：宏观压力概览 ====================
st.header("🏥 宏观压力概览")

# 计算关键指标
# 1. EPU当前分位
epu_current_percentile = epu_df['Percentile'].iloc[-1] if not epu_df.empty else 0.5
epu_current_value = epu_df['EPU_Index'].iloc[-1] if not epu_df.empty else 0

# 2. iVIX当前值
ivix_current = ivix_df['ivix'].iloc[-1] if not ivix_df.empty else 20

# 3. 两融余额环比变化
if not margin_df.empty and len(margin_df) >= 2:
    latest_fin_balance = margin_df['fin_balance'].iloc[-1]
    prev_fin_balance = margin_df['fin_balance'].iloc[-2]
    if prev_fin_balance != 0:
        margin_ratio = (latest_fin_balance - prev_fin_balance) / prev_fin_balance * 100
    else:
        margin_ratio = 0
else:
    margin_ratio = 0
    latest_fin_balance = margin_df['fin_balance'].iloc[-1] if not margin_df.empty else 0

# 创建三列展示关键指标
col1, col2, col3 = st.columns(3)

with col1:
    # EPU分位指标卡
    st.metric(
        label="📈 经济政策不确定性 (EPU)",
        value=f"{epu_current_percentile:.1%}",
        delta=f"分位排名",
        delta_color="inverse" if epu_current_percentile > 0.5 else "normal"
    )
    st.caption(f"当前指数值: {epu_current_value:.1f}")

with col2:
    # iVIX指标卡
    ivix_status = "高压" if ivix_current > 30 else "中压" if ivix_current > 20 else "低压"
    st.metric(
        label="😨 恐慌指数 (iVIX)",
        value=f"{ivix_current:.2f}",
        delta=ivix_status,
        delta_color="inverse" if ivix_current > 25 else "normal"
    )
    st.caption("阈值: 20(低压) | 30(高压) | 40(极度恐慌)")

with col3:
    # 两融余额环比指标卡
    st.metric(
        label="💰 两融余额环比",
        value=f"{latest_fin_balance:,.0f} 亿元",
        delta=f"{margin_ratio:+.2f}%",
        delta_color="inverse" if margin_ratio < -1 else "normal"
    )
    st.caption("融资余额变化率")

# 临床诊断结论
st.subheader("🏥 今日环境诊断结论")

# 根据指标综合判断
diagnosis = ""
if epu_current_percentile > 0.8 and ivix_current > 30:
    diagnosis = "🔴 高压避险环境：经济政策不确定性高，市场恐慌情绪显著，建议保守操作，控制仓位。"
elif epu_current_percentile > 0.6 or ivix_current > 25:
    diagnosis = "🟠 中压谨慎环境：存在一定不确定性或恐慌情绪，建议谨慎投资，适度配置防御性资产。"
elif epu_current_percentile < 0.4 and ivix_current < 20 and margin_ratio > 0:
    diagnosis = "🟢 低压扩张环境：政策环境稳定，市场情绪平稳，融资活跃，可适度增加风险暴露。"
else:
    diagnosis = "🟡 平衡观察环境：各项指标处于中性区间，建议维持现有策略，关注后续变化。"

st.info(diagnosis)

# ==================== 计算分位数据（供AI分析使用） ====================
# 计算iVIX过去一年分位
if not ivix_df.empty:
    ivix_last_year = ivix_df['ivix'].tail(252) if len(ivix_df) >= 252 else ivix_df['ivix']
    ivix_percentile = (ivix_last_year < ivix_current).sum() / len(ivix_last_year)
else:
    ivix_percentile = 0.5

# 计算融资余额分位
if not margin_df.empty:
    balance_last_year = margin_df['fin_balance'].tail(252) if len(margin_df) >= 252 else margin_df['fin_balance']
    margin_percentile = (balance_last_year < margin_df['fin_balance'].iloc[-1]).sum() / len(balance_last_year) if len(balance_last_year) > 0 else 0.5
else:
    margin_percentile = 0.5

# ==================== AI综合分析 ====================
st.subheader("🤖 AI综合分析")

# 加载config.json获取DeepSeek API Key
try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        deepseek_api_key = config.get('deepseek_api_key', '')
except Exception as e:
    print(f"读取config.json失败: {e}")
    deepseek_api_key = ''

# 准备宏观指标数据
macro_indicators = {
    'epu_current_value': epu_current_value,
    'epu_current_percentile': epu_current_percentile,
    'ivix_current': ivix_current,
    'ivix_percentile': ivix_percentile,
    'latest_fin_balance': latest_fin_balance,
    'margin_ratio': margin_ratio,
    'margin_percentile': margin_percentile,
    'diagnosis': diagnosis
}

# 执行AI分析
if deepseek_api_key:
    with st.spinner("🤖 AI正在分析宏观环境..."):
        ai_analysis = analyze_macro_environment(deepseek_api_key, macro_indicators)
    
    # 显示总体评估
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # 根据评估等级显示不同的表情和颜色
        assessment = ai_analysis.get('overall_assessment', '不确定')
        confidence = ai_analysis.get('confidence_score', 0)
        
        if assessment == '良好':
            emoji = "🟢"
            color = "green"
        elif assessment == '谨慎':
            emoji = "🟡"
            color = "yellow"
        elif assessment == '危险':
            emoji = "🔴"
            color = "red"
        else:
            emoji = "⚪"
            color = "gray"
        
        st.markdown(f"<h3 style='text-align: center; color: {color};'>{emoji} {assessment}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>置信度: <b>{confidence}%</b></p>", unsafe_allow_html=True)
    
    with col2:
        # 显示市场时机建议
        timing = ai_analysis.get('market_timing', '无法评估')
        timing_map = {
            '积极加仓': '🚀',
            '适度建仓': '📈',
            '保持观望': '👀',
            '减仓避险': '🛡️',
            '无法评估': '❓'
        }
        timing_emoji = timing_map.get(timing, '❓')
        st.markdown(f"<h4 style='text-align: center;'>{timing_emoji} {timing}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 0.9em;'>AI建议的市场时机</p>", unsafe_allow_html=True)
    
    with col3:
        # 显示技术分析摘要
        tech_analysis = ai_analysis.get('technical_analysis', '无技术分析')
        # 使用st.info来确保深色主题下的可读性
        st.info(f"{tech_analysis[:200]}{'...' if len(tech_analysis) > 200 else ''}")
    
    st.markdown("---")
    
    # 创建两列显示关键洞察和投资建议
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🔍 关键市场洞察")
        insights = ai_analysis.get('key_insights', [])
        for i, insight in enumerate(insights[:3], 1):
            st.markdown(f"{i}. {insight}")
    
    with col_right:
        st.markdown("#### 💡 投资建议")
        recommendations = ai_analysis.get('investment_recommendations', [])
        for i, rec in enumerate(recommendations[:3], 1):
            st.markdown(f"{i}. {rec}")
    
    st.markdown("---")
    
    # 显示风险警告
    st.markdown("#### ⚠️ 风险警告")
    warnings = ai_analysis.get('risk_warnings', [])
    for i, warning in enumerate(warnings[:2], 1):
        st.warning(f"{i}. {warning}")
    
else:
    st.warning("⚠️ 未配置DeepSeek API Key，无法进行AI分析。请在config.json中配置deepseek_api_key。")

# ==================== 融资买入占比可视化 ====================
st.header("📊 融资买入占比分析")

# 检查融资买入占比数据
if not margin_df.empty:
    # 确保有足够的日期数据
    margin_ratio_df = margin_df.copy()
    margin_ratio_df['date'] = pd.to_datetime(margin_ratio_df['date'])
    margin_ratio_df = margin_ratio_df.sort_values('date')
    
    # 获取最近90天的数据
    recent_data = margin_ratio_df.tail(90).copy()
    
    # 创建融资买入占比柱状图
    fig_margin = go.Figure()
    
    # 根据阈值设置颜色
    colors = []
    for ratio in recent_data['fin_buy_ratio']:
        if ratio > 12:
            colors.append('red')  # 过热
        elif ratio < 7:
            colors.append('blue')  # 冰点
        else:
            colors.append('green')  # 正常
    
    fig_margin.add_trace(go.Bar(
        x=recent_data['date'],
        y=recent_data['fin_buy_ratio'],
        name='融资买入占比',
        marker_color=colors,
        text=[f'{ratio:.1f}%' for ratio in recent_data['fin_buy_ratio']],
        textposition='auto',
    ))
    
    # 添加阈值线
    fig_margin.add_hline(y=12, line_dash="dash", line_color="red", 
                         annotation_text="过热阈值 12%", 
                         annotation_position="bottom right")
    fig_margin.add_hline(y=7, line_dash="dash", line_color="blue", 
                         annotation_text="冰点阈值 7%", 
                         annotation_position="top right")
    
    # 更新布局
    fig_margin.update_layout(
        title="融资买入占比 (最近90天)",
        xaxis_title="日期",
        yaxis_title="融资买入占比 (%)",
        height=400,
        hovermode="x unified",
        showlegend=False
    )
    
    st.plotly_chart(fig_margin, config=PLOTLY_CONFIG)
    
    # 显示统计信息
    current_ratio = margin_ratio_df['fin_buy_ratio'].iloc[-1] if len(margin_ratio_df) > 0 else 0
    ratio_status = "🔥 过热" if current_ratio > 12 else "❄️ 冰点" if current_ratio < 7 else "✅ 正常"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="当前融资买入占比",
            value=f"{current_ratio:.2f}%",
            delta=ratio_status
        )
    with col2:
        st.metric(
            label="90日均值",
            value=f"{recent_data['fin_buy_ratio'].mean():.2f}%",
            delta=f"{'高于' if recent_data['fin_buy_ratio'].mean() > 9 else '低于'}中性线"
        )
    with col3:
        st.metric(
            label="90日标准差",
            value=f"{recent_data['fin_buy_ratio'].std():.2f}%",
            delta="波动率"
        )
    
    # 解释说明
    st.markdown("""
    **融资买入占比说明：**
    - **🔴 > 12%**: 市场过热，融资买入活跃度异常高，需警惕风险
    - **🔵 < 7%**: 市场冰点，融资买入意愿低迷，可能存在超跌机会
    - **🟢 7%-12%**: 正常区间，市场情绪平稳
    
    **循证价值**：该指标通过计算全市场融资买入额与总成交额的比值，真实反映杠杆资金的市场参与度。
    高占比通常预示市场情绪过热，低占比则可能反映市场过度悲观。
    """)
    
    # 添加融资余额趋势图（双轴）
    st.subheader("融资余额与占比趋势")
    
    fig_balance = go.Figure()
    
    # 添加融资余额线（主坐标轴）
    fig_balance.add_trace(go.Scatter(
        x=recent_data['date'],
        y=recent_data['fin_balance'],
        name="融资余额",
        line=dict(color='orange', width=2),
        yaxis="y1"
    ))
    
    # 添加融资买入占比线（次坐标轴）
    fig_balance.add_trace(go.Scatter(
        x=recent_data['date'],
        y=recent_data['fin_buy_ratio'],
        name="融资买入占比",
        line=dict(color='purple', width=2, dash='dash'),
        yaxis="y2"
    ))
    
    # 更新布局
    fig_balance.update_layout(
        title="融资余额与买入占比趋势",
        xaxis_title="日期",
        yaxis=dict(
            title=dict(text="融资余额 (亿元)", font=dict(color="orange")),
            tickfont=dict(color="orange")
        ),
        yaxis2=dict(
            title=dict(text="融资买入占比 (%)", font=dict(color="purple")),
            tickfont=dict(color="purple"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0, y=1.1, orientation="h"),
        height=400,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_balance, config=PLOTLY_CONFIG)

else:
    st.warning("暂无融资买入占比数据")

st.markdown("---")

# ==================== 中部：可视化图表 ====================
st.header("📈 宏观指标可视化")

# 准备数据用于EPU与上证指数对比图（这里需要获取上证指数数据）
# 由于akshare可能不在依赖中，我们这里先使用模拟数据，实际使用时可以接入真实数据
@st.cache_data(ttl=3600)
def get_shanghai_index():
    """获取上证指数数据（模拟）"""
    try:
        import akshare as ak
        # 获取上证指数历史数据
        df_index = ak.stock_zh_index_hist(symbol="000001", period="daily")
        df_index = df_index[['日期', '收盘']].rename(columns={'日期': 'date', '收盘': 'close'})
        df_index['date'] = pd.to_datetime(df_index['date'])
        return df_index
    except:
        # 如果akshare不可用，生成模拟数据
        dates = pd.date_range(end=datetime.now(), periods=365*2, freq='D')
        values = 3000 + np.cumsum(np.random.normal(0, 20, len(dates)))
        df_index = pd.DataFrame({'date': dates, 'close': values})
        return df_index

# 创建两列用于图表
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("EPU vs 上证指数对比")
    
    # 获取上证指数数据
    with st.spinner("加载上证指数数据..."):
        sh_index_df = get_shanghai_index()
    
    # 准备EPU数据（转换为与指数相同的时间尺度）
    # EPU是月度数据，我们需要将其与指数对齐
    epu_for_chart = epu_df.copy()
    epu_for_chart.index = pd.to_datetime(epu_for_chart.index)
    
    # 创建双轴图表
    fig1 = go.Figure()
    
    # 添加上证指数线
    fig1.add_trace(go.Scatter(
        x=sh_index_df['date'],
        y=sh_index_df['close'],
        name="上证指数",
        line=dict(color='blue', width=2),
        yaxis="y1"
    ))
    
    # 添加EPU线（使用次坐标轴）
    fig1.add_trace(go.Scatter(
        x=epu_for_chart.index,
        y=epu_for_chart['EPU_Index'],
        name="EPU指数",
        line=dict(color='red', width=2, dash='dash'),
        yaxis="y2"
    ))
    
    # 更新布局
    fig1.update_layout(
        title="EPU与上证指数历史对比",
        xaxis_title="日期",
        yaxis=dict(
            title=dict(text="上证指数", font=dict(color="blue")),
            tickfont=dict(color="blue")
        ),
        yaxis2=dict(
            title=dict(text="EPU指数", font=dict(color="red")),
            tickfont=dict(color="red"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0, y=1.1, orientation="h"),
        height=400,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig1, config=PLOTLY_CONFIG)

with chart_col2:
    st.subheader("恐慌指数 (iVIX) 仪表盘")
    
    # 创建仪表盘图
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ivix_current,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "iVIX 恐慌指数"},
        delta={'reference': 20, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'green'},
                {'range': [20, 30], 'color': 'yellow'},
                {'range': [30, 40], 'color': 'orange'},
                {'range': [40, 50], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': ivix_current
            }
        }
    ))
    
    fig2.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    st.plotly_chart(fig2, config=PLOTLY_CONFIG)
    
    # 添加iVIX阈值说明
    st.caption("""
    **阈值说明：**
    - 🟢 < 20: 市场情绪稳定，低压区间
    - 🟡 20-30: 市场情绪波动，关注风险
    - 🟠 30-40: 市场恐慌情绪上升，高压区间
    - 🔴 > 40: 极度恐慌，市场可能超跌
    """)

st.markdown("---")

# ==================== 底部：情绪分位计算器 ====================
st.header("🧮 情绪分位计算器")

st.markdown("自动计算当前各项指标在过去一年（252个交易日）中的百分分位，标记'过热'或'超跌'区域。")

# 计算各项指标的分位
# 1. iVIX分位
if not ivix_df.empty:
    ivix_last_year = ivix_df['ivix'].tail(252) if len(ivix_df) >= 252 else ivix_df['ivix']
    ivix_percentile = (ivix_last_year < ivix_current).sum() / len(ivix_last_year)
else:
    ivix_percentile = 0.5

# 2. 融资买入占比分位
if not margin_df.empty:
    margin_last_year = margin_df['fin_buy_ratio'].tail(252) if len(margin_df) >= 252 else margin_df['fin_buy_ratio']
    margin_percentile = (margin_last_year < margin_df['fin_buy_ratio'].iloc[-1]).sum() / len(margin_last_year) if len(margin_last_year) > 0 else 0.5
else:
    margin_percentile = 0.5

# 3. 融资余额分位
if not margin_df.empty:
    balance_last_year = margin_df['fin_balance'].tail(252) if len(margin_df) >= 252 else margin_df['fin_balance']
    balance_percentile = (balance_last_year < margin_df['fin_balance'].iloc[-1]).sum() / len(balance_last_year) if len(balance_last_year) > 0 else 0.5
else:
    balance_percentile = 0.5

# 创建分位显示表格
percentile_data = pd.DataFrame({
    '指标': ['恐慌指数 (iVIX)', '融资买入占比', '融资余额'],
    '当前值': [f"{ivix_current:.2f}", f"{margin_df['fin_buy_ratio'].iloc[-1]:.1f}%" if not margin_df.empty else "N/A", f"{latest_fin_balance:,.0f} 亿元"],
    '过去一年分位': [f"{ivix_percentile:.1%}", f"{margin_percentile:.1%}", f"{balance_percentile:.1%}"],
    '状态': [
        "🔥 过热" if ivix_percentile > 0.9 else "💧 超跌" if ivix_percentile < 0.1 else "⚖️ 正常",
        "🔥 过热" if margin_percentile > 0.9 else "💧 超跌" if margin_percentile < 0.1 else "⚖️ 正常",
        "🔥 过热" if balance_percentile > 0.9 else "💧 超跌" if balance_percentile < 0.1 else "⚖️ 正常"
    ]
})

# 显示表格
st.dataframe(percentile_data, width='stretch', hide_index=True)

# 添加解释说明
st.markdown("""
**分位解释：**
- **🔥 过热**: 指标位于过去一年的前10%（分位 > 90%），可能表示市场情绪过热
- **💧 超跌**: 指标位于过去一年的后10%（分位 < 10%），可能表示市场情绪过度悲观
- **⚖️ 正常**: 指标处于正常区间（10% ≤ 分位 ≤ 90%）

**使用建议：**
1. 当多个指标显示"过热"时，考虑降低仓位，增加防御性配置
2. 当多个指标显示"超跌"时，可能提供左侧布局机会
3. 结合顶部诊断结论，制定综合投资策略
""")

# ==================== 系统维护与数据重置 ====================
st.header("🛠️ 系统维护与数据重置")

with st.expander("三级重置逻辑 (Triple-Level Reset)", expanded=False):
    st.markdown("""
    **三级重置逻辑：**
    
    1. **Level 1: 缓存重置 (Cache Purge)**  
       - 仅清理 Hot Buffer (JSON) 和 Streamlit 缓存。  
       - 适用场景：解决盘中实时数据跳动或显示异常。
    
    2. **Level 2: 标的重置 (Targeted Reset)**  
       - 删除特定标的的 .parquet 历史文件。  
       - 适用场景：某个标的发生重大除权除息导致“失焦”，需重新同步。
    
    3. **Level 3: 系统初始化 (System Factory Reset)**  
       - 清空所有 Parquet、SQLite 表及 AI 报告。  
       - 适用场景：系统架构重大升级，需从零开始重新构建证据链。
    """)
    
    st.info("**安全协议：先备份，后销毁 (Backup-Before-Wipe)**  \n在删除前，系统会自动将当前 data/ 目录压缩存入 data/archive/backup_[timestamp].zip。")

# 重置选项
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 数据存储重置")
    
    # 全市场同步
    st.markdown("---")
    st.markdown("**🔄 全市场数据同步**")
    st.caption("同步所有 Master Pool 标的的历史数据，执行自动补缺和握手校验")
    
    force_update = st.checkbox("强制更新（忽略缓存）", value=False, key="force_update_sync")
    
    if st.button("🚀 执行全市场同步", type="primary", key="full_market_sync"):
        with st.spinner("正在执行全市场同步，请稍候..."):
            try:
                from services.sync_manager import run_full_market_sync
                summary = run_full_market_sync(force_update=force_update)
                
                if summary['success']:
                    st.success(f"全市场同步完成: {summary['message']}")
                    
                    # 显示详细结果
                    with st.expander("📊 同步详情", expanded=False):
                        st.write(f"**总计**: {summary['total']} 只标的")
                        st.write(f"**成功**: {summary['success_count']} 只")
                        st.write(f"**失败**: {summary['failed_count']} 只")
                        st.write(f"**新增记录**: {summary['total_records']} 条")
                        
                        if summary['failed_stocks']:
                            st.warning("**失败的标的**:")
                            for code, name, msg in summary['failed_stocks']:
                                st.write(f"- {name}({code}): {msg}")
                else:
                    st.error(f"全市场同步失败: {summary['message']}")
                    
            except Exception as e:
                st.error(f"同步执行失败: {str(e)}")
    
    # 同步状态检查
    if st.button("📊 检查同步状态", type="secondary", key="check_sync_status"):
        with st.spinner("正在检查同步状态..."):
            try:
                from services.sync_manager import get_sync_status
                status = get_sync_status()
                
                st.info(f"**同步状态报告** (生成时间: {status['report_time']})")
                st.write(f"**Master Pool 标的总数**: {status['total_master']}")
                st.write(f"**缺失数据标的**: {status['missing_master']}")
                
                st.write("**数据新鲜度分布**:")
                fresh_counts = status['fresh_counts']
                st.write(f"- 今日数据: {fresh_counts['today']}")
                st.write(f"- 昨日数据: {fresh_counts['yesterday']}")
                st.write(f"- 3日内数据: {fresh_counts['within_3_days']}")
                st.write(f"- 1周内数据: {fresh_counts['within_week']}")
                st.write(f"- 较旧数据: {fresh_counts['older']}")
                st.write(f"- 缺失数据: {fresh_counts['missing']}")
                
            except Exception as e:
                st.error(f"状态检查失败: {str(e)}")
    
    st.markdown("---")
    st.markdown("**三级重置逻辑**")
    
    # Level 1: 缓存重置
    if st.button("🔄 Level 1: 缓存重置", type="secondary", key="reset_cache"):
        from storage_manager import get_storage_manager
        storage_manager = get_storage_manager()
        result = storage_manager.purge_data(scope='hot_buffer', create_backup=True)
        if result['success']:
            st.success(f"缓存重置完成: {result['message']}")
            if result.get('backup_path'):
                st.info(f"备份已创建: {result['backup_path']}")
        else:
            st.error(f"缓存重置失败: {result['message']}")
    
    # Level 2: 标的重置
    st.markdown("---")
    st.markdown("**Level 2: 标的重置**")
    ticker_to_reset = st.text_input("输入股票代码进行重置", placeholder="例如: 000001")
    if st.button("🎯 Level 2: 重置指定标的", type="secondary", key="reset_ticker"):
        if ticker_to_reset:
            from storage_manager import get_storage_manager
            storage_manager = get_storage_manager()
            result = storage_manager.purge_data(scope='ticker', ticker=ticker_to_reset, create_backup=True)
            if result['success']:
                st.success(f"股票 {ticker_to_reset} 重置完成: {result['message']}")
                if result.get('backup_path'):
                    st.info(f"备份已创建: {result['backup_path']}")
            else:
                st.error(f"重置失败: {result['message']}")
        else:
            st.warning("请输入股票代码")
    
    # Level 3: 系统初始化
    st.markdown("---")
    st.markdown("**Level 3: 系统初始化**")
    st.warning("⚠️ 此操作将清空所有本地数据，包括历史Parquet文件、热缓冲区和Streamlit缓存。")
    confirm_reset = st.checkbox("我确认理解此操作将删除所有本地数据", key="confirm_reset_all")
    if st.button("💥 Level 3: 系统初始化", type="primary", key="reset_all", disabled=not confirm_reset):
        from storage_manager import get_storage_manager
        storage_manager = get_storage_manager()
        result = storage_manager.purge_data(scope='all', create_backup=True)
        if result['success']:
            st.success(f"系统初始化完成: {result['message']}")
            if result.get('backup_path'):
                st.info(f"备份已创建: {result['backup_path']}")
        else:
            st.error(f"系统初始化失败: {result['message']}")

with col2:
    st.subheader("🗃️ 数据库重置")
    
    # 数据库表重置
    st.markdown("**重置数据库表**")
    tables_to_reset = st.multiselect(
        "选择要重置的表",
        options=['macro_indicators', 'ai_reports', 'stock_data_status'],
        default=['macro_indicators', 'ai_reports']
    )
    
    if st.button("🗑️ 重置选定数据库表", type="secondary", key="reset_db_tables"):
        if tables_to_reset:
            from database_manager import get_database_manager
            db_manager = get_database_manager()
            result = db_manager.reset_tables(tables=tables_to_reset)
            if result['success']:
                st.success(f"数据库表重置完成: {result['message']}")
                st.info(f"重置了 {len(result['tables_reset'])} 个表，删除了 {result['rows_deleted']} 行数据")
                if result.get('backup_path'):
                    st.info(f"备份已创建: {result['backup_path']}")
            else:
                st.error(f"数据库表重置失败: {result['message']}")
        else:
            st.warning("请选择至少一个表")
    
    # 健康自检
    st.markdown("---")
    st.subheader("🩺 健康自检")
    if st.button("🔍 执行健康自检", type="secondary", key="health_check"):
        with st.spinner("正在执行健康自检..."):
            # 检查目录结构
            import os
            from pathlib import Path
            
            check_results = []
            
            # 1. 检查目录结构
            required_dirs = ['market_data', 'temp_hot_data', 'backups']
            for dir_path in required_dirs:
                if Path(dir_path).exists():
                    check_results.append(f"✅ 目录 {dir_path} 存在")
                else:
                    check_results.append(f"⚠️ 目录 {dir_path} 不存在")
            
            # 2. 检查数据库连接
            try:
                from database_manager import get_database_manager
                db_manager = get_database_manager()
                db_info = db_manager.get_table_info('macro_indicators')
                if db_info:
                    check_results.append(f"✅ 数据库连接正常")
                else:
                    check_results.append(f"⚠️ 数据库表信息获取失败")
            except Exception as e:
                check_results.append(f"❌ 数据库连接失败: {str(e)}")
            
            # 3. 检查API Key
            try:
                import json
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                if config.get('deepseek_api_key'):
                    check_results.append(f"✅ DeepSeek API Key 配置正常")
                else:
                    check_results.append(f"⚠️ DeepSeek API Key 未配置")
            except Exception as e:
                check_results.append(f"❌ 配置文件读取失败: {str(e)}")
            
            # 4. 检查数据文件
            try:
                from storage_manager import get_storage_manager
                storage_manager = get_storage_manager()
                # 获取配置中的股票列表
                from config import load_config
                config = load_config()
                master_pool = config.get('master_pool', {})
                if master_pool:
                    check_results.append(f"✅ 配置文件加载正常，包含 {len(master_pool)} 个分类")
                else:
                    check_results.append(f"⚠️ 配置文件中的master_pool为空")
            except Exception as e:
                check_results.append(f"❌ 数据文件检查失败: {str(e)}")
            
            # 显示检查结果
            st.markdown("**健康自检结果:**")
            for result in check_results:
                if result.startswith('✅'):
                    st.success(result)
                elif result.startswith('⚠️'):
                    st.warning(result)
                else:
                    st.error(result)
            
            # 总体评估
            success_count = sum(1 for r in check_results if r.startswith('✅'))
            warning_count = sum(1 for r in check_results if r.startswith('⚠️'))
            error_count = sum(1 for r in check_results if r.startswith('❌'))
            
            if error_count == 0 and warning_count == 0:
                st.success(f"🎉 系统健康状态: 优秀 ({success_count}/{len(check_results)} 项通过)")
            elif error_count == 0:
                st.info(f"📋 系统健康状态: 良好 ({success_count}/{len(check_results)} 项通过, {warning_count} 项警告)")
            else:
                st.error(f"🚨 系统健康状态: 异常 ({success_count}/{len(check_results)} 项通过, {warning_count} 项警告, {error_count} 项错误)")

# 重置后提示
st.info("""
**重置后操作提示：**
1. 执行重置后，系统将恢复至初始冷启动状态。
2. 请重新执行数据同步以获取最新数据。
3. 建议在重置后立即运行一次"健康自检"以确保系统完整性。
4. 所有重置操作都会自动创建备份，可在 `data/archive/` 目录下找到。
""")

st.markdown("---")

# 添加数据更新时间
st.caption(f"数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("注：宏观数据更新频率较低，EPU为月度数据，iVIX和两融数据为日度数据。")
