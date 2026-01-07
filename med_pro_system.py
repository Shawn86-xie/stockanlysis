import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 导入自定义模块
from config import load_config, save_config
from data_fetcher import fetch_stock_data, get_signals, compute_portfolio_stats, monte_carlo_simulation, get_recent_10_days, analyze_portfolio
from ai_analyzer import deepseek_analyze

# --- 环境与中文字体配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="2026 医疗量化看板-模块化版", layout="wide")

# --- 加载配置 ---
config = load_config()

# --- 标的库定义 (Session State) ---
if 'master_pool' not in st.session_state:
    # 从配置加载标的库，如果不存在则使用默认值
    config_master_pool = config.get('master_pool')
    if config_master_pool:
        st.session_state.master_pool = config_master_pool
    else:
        st.session_state.master_pool = {
            "核心组合": {
                "300347": "泰格医药",
                "688506": "百利天恒",
                "603259": "药明康德",
                "600276": "恒瑞医药",
                "301293": "三博脑科",
                "000538": "云南白药",
                "600521": "华海药业"
            },
            "科研/肾科": {
                "300453": "三鑫医疗",
                "002223": "鱼跃医疗",
                "603108": "润达医疗"
            }
        }

# --- 加载用户保存的选择和持仓 ---
if 'user_settings' not in st.session_state:
    st.session_state.user_settings = config.get('user_settings', {})

# --- Streamlit 侧边栏 ---
st.sidebar.title("🤖 决策中心 (2026)")
ds_key = st.sidebar.text_input("DeepSeek Key", value=config.get("deepseek_api_key", ""), type="password",
                               help="用于访问DeepSeek API的密钥，获取AI分析的新闻资讯。")
st.sidebar.markdown("---")

# 标的库管理
with st.sidebar.expander("📚 标的库管理", expanded=False):
    st.write("在这里添加或删除标的。")
    
    # 添加新标的
    st.subheader("添加新标的")
    
    # 股票代码搜索
    search_query = st.text_input("🔍 搜索股票代码或名称", placeholder="输入代码或名称，如 '300347' 或 '泰格医药'", key="stock_search")
    
    search_results = []
    if search_query and len(search_query.strip()) > 0:
        from data_fetcher import search_stock_info
        search_results = search_stock_info(search_query.strip())
    
    # 搜索结果展示
    selected_stock = None
    if search_results:
        st.info(f"找到 {len(search_results)} 个匹配的股票")
        # 创建选择列表
        stock_options = [f"{item['code']} - {item['name']}" for item in search_results]
        selected_option = st.selectbox("选择股票", options=stock_options, key="stock_select")
        
        if selected_option:
            # 解析选择的股票
            code, name = selected_option.split(" - ", 1)
            selected_stock = {'code': code, 'name': name}
            st.success(f"已选择: {name} ({code})")
    
    col_code, col_name, col_cat = st.columns(3)
    with col_code:
        if selected_stock:
            new_code = st.text_input("股票代码", value=selected_stock['code'], placeholder="例如: 000001", key="new_code")
        else:
            new_code = st.text_input("股票代码", value="", placeholder="例如: 000001", key="new_code")
    with col_name:
        if selected_stock:
            new_name = st.text_input("股票名称", value=selected_stock['name'], placeholder="例如: 平安银行", key="new_name")
        else:
            new_name = st.text_input("股票名称", value="", placeholder="例如: 平安银行", key="new_name")
    with col_cat:
        new_category = st.selectbox("分类", ["核心组合", "科研/肾科", "其他"], key="new_category")
    
    if st.button("➕ 添加标的", type="secondary", use_container_width=True):
        if new_code and new_name:
            # 确保分类存在
            if new_category not in st.session_state.master_pool:
                st.session_state.master_pool[new_category] = {}
            
            # 检查是否已存在
            if new_code in st.session_state.master_pool[new_category]:
                st.warning(f"股票代码 {new_code} 已在 {new_category} 中存在！")
            else:
                # 添加标的
                st.session_state.master_pool[new_category][new_code] = new_name
                st.success(f"已添加 {new_name}({new_code}) 到 {new_category}")
                # 重新加载页面
                st.rerun()
        else:
            st.warning("请填写股票代码和名称")
    
    st.subheader("删除标的")
    # 显示现有标的供删除
    for cat, stocks in list(st.session_state.master_pool.items()):
        st.write(f"**{cat}**")
        for c, n in list(stocks.items()):
            if st.button(f"🗑️ 删除 {n} ({c})", key=f"del_{c}", use_container_width=True):
                del st.session_state.master_pool[cat][c]
                # 如果分类为空，删除分类
                if not st.session_state.master_pool[cat]:
                    del st.session_state.master_pool[cat]
                st.success(f"已删除 {n}({c})")
                st.rerun()
    
    # 保存标的库按钮
    st.subheader("保存标的库")
    if st.button("💾 保存标的库到配置文件", type="primary", use_container_width=True):
        config['master_pool'] = st.session_state.master_pool
        save_config(config)
        st.success("标的库已保存到配置文件！")
        st.info("应用将重新加载以使用新的标的库...")
        st.rerun()

st.sidebar.markdown("---")

# 风险参数设置（先定义，以便在保存按钮中使用）
stop_loss_val = st.sidebar.slider("风险提醒阈值 (止损)", -0.10, -0.01, config.get("stop_loss_threshold", -0.05),
                                   help="当日跌幅超过此阈值时触发风险提醒。例如-5%表示当日下跌5%以上时提示止损。")
sim_num = st.sidebar.number_input("模拟次数", value=config.get("default_sim_count", 3000),
                                   help="蒙特卡洛模拟中随机生成的投资组合数量。模拟次数越多，结果越精确，但计算时间越长。")

st.sidebar.markdown("---")

# 设置管理按钮
col_save, col_load = st.sidebar.columns(2)
with col_save:
    if st.button("💾 保存当前设置", use_container_width=True):
        # 收集当前选择
        selected_codes = []
        for cat, stocks in st.session_state.master_pool.items():
            for c, n in stocks.items():
                if st.session_state.get(f"sel_{c}", True):
                    selected_codes.append(c)
        
        # 收集持仓（从session_state获取）
        holdings_save = {}
        selected_names = [st.session_state.master_pool[cat][c] for cat in st.session_state.master_pool for c in selected_codes if c in st.session_state.master_pool[cat]]
        for name in selected_names:
            holdings_save[name] = st.session_state.get(f"hold_{name}", 0)
        
        # 保存到配置
        st.session_state.user_settings = {
            'selected_codes': selected_codes,
            'holdings': holdings_save,
            'stop_loss': stop_loss_val,
            'sim_num': sim_num
        }
        
        # 更新全局配置并保存到文件
        config['user_settings'] = st.session_state.user_settings
        save_config(config)
        st.sidebar.success("设置已保存！")

with col_load:
    if st.button("📂 加载保存的设置", use_container_width=True):
        if st.session_state.user_settings:
            # 恢复选择状态
            saved_codes = st.session_state.user_settings.get('selected_codes', [])
            for cat, stocks in st.session_state.master_pool.items():
                for c, n in stocks.items():
                    st.session_state[f"sel_{c}"] = c in saved_codes
            
            # 恢复持仓
            saved_holdings = st.session_state.user_settings.get('holdings', {})
            for name, qty in saved_holdings.items():
                st.session_state[f"hold_{name}"] = qty
            
            st.sidebar.success("设置已加载！")
        else:
            st.sidebar.warning("没有找到保存的设置")

st.sidebar.markdown("---")

# 标的选择
final_sel = {}
for cat, stocks in st.session_state.master_pool.items():
    st.sidebar.write(f"**{cat}**")
    for c, n in stocks.items():
        # 从session_state获取选择状态，默认为True
        checkbox_state = st.session_state.get(f"sel_{c}", True)
        if st.sidebar.checkbox(f"{n} ({c})", value=checkbox_state, key=f"sel_{c}_ui"):
            final_sel[c] = n
            # 同步状态到session_state
            st.session_state[f"sel_{c}"] = True
        else:
            st.session_state[f"sel_{c}"] = False

st.sidebar.markdown("---")
# 开始统计按钮
if st.sidebar.button("🚀 开始统计", type="primary", use_container_width=True):
    st.session_state.run_analysis = True
    st.session_state.selected_stocks = final_sel.copy()
    st.sidebar.success("开始分析选中的标的...")
    # 注意：这里不能直接rerun，因为需要先保存session_state
    # 我们将在主界面中根据run_analysis状态来触发分析

# 如果之前有分析运行状态，但用户取消了所有选择，则清除分析状态
if not final_sel and st.session_state.get('run_analysis', False):
    st.session_state.run_analysis = False
    st.session_state.selected_stocks = {}

# --- 主界面 ---
st.title("🏥 医疗量化研报系统 (模块化版)")
st.caption(f"科研工作者专属调仓决策工具 | 当前配置生效日期: {datetime.now().strftime('%Y-%m-%d')}")

# 检查是否应该运行分析
should_run_analysis = st.session_state.get('run_analysis', False) and st.session_state.get('selected_stocks', {})
analysis_stocks = st.session_state.get('selected_stocks', {})

if should_run_analysis and analysis_stocks:
    # 获取数据
    data = fetch_stock_data(list(analysis_stocks.keys()), list(analysis_stocks.values()))
    if not data.empty:
        # 计算收益率
        returns = data.pct_change().dropna()
        # 投资组合统计
        mean_ret, cov_mat = compute_portfolio_stats(returns, config.get("risk_free_rate", 0.0188))
        # 蒙特卡洛模拟
        sim_res = monte_carlo_simulation(mean_ret, cov_mat, sim_num, config.get("risk_free_rate", 0.0188))
        best_p = sim_res.iloc[sim_res['Sharpe'].idxmax()]

        # 功能标签页
        t1, t2, t3, t4, t5 = st.tabs(["💡 AI 资讯深度研判", "🚦 买卖与风险预警", "🕸️ 板块相关性分析", "📊 权重优化实验", "📈 模拟交易"])

        with t1:
            st.subheader("DeepSeek 医疗专业资讯评分")
            st.info("AI资讯分析需要调用外部API，耗时较长，请手动点击按钮获取。")
            
            # 初始化session_state存储新闻和分析结果
            if 'news_results' not in st.session_state:
                st.session_state.news_results = {}
            
            # 按钮触发获取新闻
            if st.button("🔍 获取最新新闻并分析", type="primary"):
                with st.spinner("正在获取新闻并分析..."):
                    for c, n in analysis_stocks.items():
                        try:
                            import akshare as ak
                            news = ak.stock_news_em(symbol=c).head(2)
                            news_list = []
                            for _, row in news.iterrows():
                                stars, nature, color, reason = deepseek_analyze(ds_key, row['新闻标题'], n)
                                news_list.append({
                                    'title': row['新闻标题'],
                                    'stars': stars,
                                    'nature': nature,
                                    'color': color,
                                    'reason': reason,
                                    'url': row.get('文章链接', '')
                                })
                            st.session_state.news_results[c] = {
                                'name': n,
                                'news': news_list
                            }
                        except Exception as e:
                            st.error(f"获取 {n} 的新闻失败：{e}")
            
            # 显示已存储的结果
            for c, data in st.session_state.news_results.items():
                if data['name'] in analysis_stocks.values():  # 只显示当前选中的标的
                    with st.expander(f"📌 {data['name']} ({c}) - 资讯洞察"):
                        for item in data['news']:
                            c_a, c_b = st.columns([4, 2])
                            c_a.write(f"**{item['title']}**")
                            c_b.markdown(f":{item['color']}[{item['stars']}] *{item['reason']}*")
                            if item['url']:
                                c_a.caption(f"[原文链接]({item['url']})")

        with t2:
            st.subheader("量化决策与风险管理")
            st.caption("""
            **列名解释**：
            - **标的**：股票名称
            - **价格**：当前收盘价（元）
            - **当日涨跌**：今日相对于昨日的涨跌幅度
            - **RSI**：相对强弱指标，>70表示超买，<30表示超卖
            - **建议**：基于RSI和均线的买卖建议
            - **风险**：风险提醒，‼️表示触及止损阈值
            - **最优配比**：AI推荐的最优投资组合权重
            """)
            sig_df = get_signals(data, best_p[list(analysis_stocks.values())].to_dict(), stop_loss_val)
            # 对触及止损的行进行高亮
            def highlight_risk(val):
                color = 'red' if '‼️' in str(val) else 'black'
                return f'color: {color}'
            st.table(sig_df.style.map(highlight_risk, subset=['风险']))
            
            # 添加下拉选择器用于突出显示曲线
            stock_names = list(analysis_stocks.values())
            if 'highlighted_stock' not in st.session_state:
                st.session_state.highlighted_stock = stock_names[0] if stock_names else None
            
            # 使用下拉选择器更新选中的股票
            highlighted = st.selectbox(
                "选择要突出显示的股票（或点击图例隐藏/显示）",
                options=stock_names,
                index=stock_names.index(st.session_state.highlighted_stock) if st.session_state.highlighted_stock in stock_names else 0,
                key='highlight_select'
            )
            st.session_state.highlighted_stock = highlighted
            
            # 创建Plotly图表
            cum_returns = (1 + returns).cumprod()
            fig = go.Figure()
            for name in cum_returns.columns:
                line_width = 3 if name == highlighted else 1
                line_opacity = 1.0 if name == highlighted else 0.5
                fig.add_trace(go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns[name],
                    name=name,
                    line=dict(width=line_width),
                    opacity=line_opacity,
                ))
            fig.update_layout(
                title="累积收益率曲线",
                xaxis_title="日期",
                yaxis_title="累积收益率",
                hovermode="x unified",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示最近10个交易日价格数据
            st.subheader("📅 最近10个交易日价格")
            recent_data = get_recent_10_days(data)
            st.dataframe(recent_data.style.format("{:.2f}"))

        with t3:
            st.subheader("资产相关性矩阵 (防范共振风险)")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(returns.corr(), annot=True, cmap='RdYlGn', center=0, ax=ax_corr)
            st.pyplot(fig_corr)

        with t4:
            col_l, col_r = st.columns(2)
            with col_l:
                st.metric("预期年化收益", f"{best_p['Ret']:.2%}")
                st.caption("📈 **预期年化收益**：基于历史数据计算的该投资组合在未来一年的预期收益率，已考虑复利效应。")
                fig_pie, ax_pie = plt.subplots()
                best_p[list(analysis_stocks.values())].plot.pie(autopct='%1.1f%%', ax=ax_pie, cmap='Pastel1', title="AI 推荐配比")
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
                    display_df = top10[['排名', 'Ret', 'Vol', 'Sharpe'] + list(analysis_stocks.values())].copy()
                    # 将收益和波动率转换为百分比字符串
                    display_df['Ret'] = display_df['Ret'].apply(lambda x: f"{x:.2%}")
                    display_df['Vol'] = display_df['Vol'].apply(lambda x: f"{x:.2%}")
                    display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.2f}")
                    # 将各股票权重转换为百分比字符串
                    for name in analysis_stocks.values():
                        display_df[name] = display_df[name].apply(lambda x: f"{x:.1%}")
                    
                    st.dataframe(display_df, use_container_width=True)
                    st.caption(f"共模拟 {sim_num} 次，展示了夏普比率最高的10个组合")
                
                fig_ef, ax_ef = plt.subplots()
                ax_ef.scatter(sim_res.Vol, sim_res.Ret, c=sim_res.Sharpe, cmap='viridis', alpha=0.3)
                ax_ef.scatter(best_p['Vol'], best_p['Ret'], color='red', marker='*', s=200)
                st.pyplot(fig_ef)
                st.caption("📊 **有效前沿图**：每个点代表一个随机投资组合，红色星号标记夏普比率最高的组合。")

        with t5:
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
                    # 调用分析函数
                    best_weights_dict = best_p[selected_names].to_dict()
                    analysis_df, total_value = analyze_portfolio(holdings, data, best_weights_dict, stop_loss_val)
                    
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

    else:
        st.error("网络连接异常，无法获取行情。")
else:
    st.warning("👈 请在左侧勾选标的。")