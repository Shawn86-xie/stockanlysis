"""
FILE: app.py
ROLE: Streamlit 应用入口，提供 Web 界面，展示数据分析结果、图表和交互控件。
LOGIC: 
    1. 加载配置和用户设置，初始化标的库和持仓数据。
    2. 提供侧边栏交互控件，包括标的库管理、交易录入、风险参数设置。
    3. 实现数据缓存机制，使用 session_state 避免重复 I/O，提升性能。
    4. 集成多页面导航，提供量化决策、风险管理、技术分析等功能。
DEPENDENCIES: streamlit, pandas, numpy, matplotlib, seaborn, plotly, config, data_fetcher, services.market_data_service, ai_analyzer, backtester, math_engine, font_utils
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# 导入自定义模块
from config import load_config, save_config, save_trade, delete_trade, clear_portfolio
from data_fetcher import get_signals, compute_portfolio_stats, monte_carlo_simulation, analyze_portfolio, calculate_deviation, fetch_stock_data
from services.market_data_service import safe_fetch_latest_prices, safe_search_stock_info, safe_fetch_candle_data, get_portfolio_summary, get_recent_10_days
from services.market_stock_list import get_market_stock_list_info, refresh_market_stock_list, init_market_stock_list_if_needed
from ai_analyzer import deepseek_analyze
from backtester import generate_signal_series, run_vectorized_backtest, plot_backtest_results, plot_backtest_with_divergence, rsi_grid_search, grid_search_with_backtest, detect_rsi_divergence, plot_divergence_chart, calculate_rsi
from math_engine import fit_trend_line, detect_peaks_valleys
from font_utils import setup_chinese_font
from auth import check_authentication, show_logout_button

# --- 页面配置（必须在其他st命令之前）---
st.set_page_config(page_title="主控台", page_icon="🏠", layout="wide")

# --- 环境与中文字体配置（使用缓存，整个容器生命周期只执行一次）---
@st.cache_resource
def setup_environment():
    """
    初始化环境配置（字体、matplotlib设置等）
    使用 @st.cache_resource 确保整个容器生命周期内只执行一次
    """
    font_name, font_path = setup_chinese_font()
    if font_name:
        print(f"[初始化] 已配置中文字体: {font_name}")
    else:
        print("[初始化] 使用默认字体配置")
    plt.rcParams['axes.unicode_minus'] = False
    return {"font_name": font_name, "font_path": font_path, "initialized": True}

# 调用初始化函数（缓存后只执行一次）
_env_config = setup_environment()

# --- 初始化锁：确保首次初始化逻辑只运行一次 ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state._config_pending_save = False  # 标记配置是否需要保存
    print("[初始化] 系统首次初始化完成，此后不再重复执行")

# --- 用户认证 ---
authenticated, user_name, username = check_authentication()
if not authenticated:
    st.stop()

# --- 配置管理（使用 session_state 缓存，避免重复 I/O）---
def get_cached_config():
    """
    获取缓存的配置，避免每次 rerun 都读取磁盘
    只在首次加载或显式刷新时读取文件
    """
    if '_config_cache' not in st.session_state:
        st.session_state._config_cache = load_config()
        st.session_state._config_dirty = False
    return st.session_state._config_cache

def safe_save_config(config_data, force=False):
    """
    安全保存配置（带限流和变更检测）

    Args:
        config_data: 要保存的配置数据
        force: 是否强制保存（忽略限流）

    Returns:
        bool: 是否实际执行了保存
    """
    import time as _time

    # 初始化保存时间记录
    if '_last_save_time' not in st.session_state:
        st.session_state._last_save_time = 0

    current_time = _time.time()
    min_interval = 5  # 最小保存间隔：5秒

    # 限流检查
    if not force:
        elapsed = current_time - st.session_state._last_save_time
        if elapsed < min_interval:
            return False

    # 执行保存
    save_config(config_data)
    st.session_state._last_save_time = current_time
    st.session_state._config_cache = config_data
    st.session_state._config_dirty = False
    return True

def mark_config_dirty():
    """标记配置已修改，需要保存"""
    st.session_state._config_dirty = True

# 获取缓存的配置
config = get_cached_config()

# --- 初始化session_state缓存 ---
# 使用session_state缓存历史数据，避免重复加载
# 为什么要使用session_state：
# 1. Streamlit每次交互都会rerun整个脚本
# 2. 如果不缓存，每次rerun都会重新读取数据（即使数据没有变化）
# 3. session_state可以跨rerun保持数据，避免重复I/O
# 4. 结合Parquet本地存储，实现"内存缓存+磁盘缓存"双层加速
if 'hist_data_cache' not in st.session_state:
    st.session_state.hist_data_cache = {}
    
if 'current_stocks_key' not in st.session_state:
    st.session_state.current_stocks_key = None

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
    
# 自动加载保存的标的勾选状态
if st.session_state.user_settings:
    saved_codes = st.session_state.user_settings.get('selected_codes', [])
    # 恢复选择状态
    for cat, stocks in st.session_state.master_pool.items():
        for c, n in stocks.items():
            st.session_state[f"sel_{c}"] = c in saved_codes

# --- Streamlit 侧边栏 (精简版) ---
st.sidebar.title("系统设置")

# 显示当前用户和登出按钮
st.sidebar.markdown(f"**当前用户:** {user_name}")
show_logout_button()

# API 密钥设置
ds_key = st.sidebar.text_input("DeepSeek API 密钥", value=config.get("deepseek_api_key", ""), type="password",
                               help="用于访问DeepSeek API的密钥，获取AI分析的新闻资讯。")

st.sidebar.markdown("---")

# 风险参数设置
st.sidebar.subheader("分析参数")
stop_loss_val = st.sidebar.slider("风险提醒阈值 (止损)", -0.10, -0.01, config.get("stop_loss_threshold", -0.05),
                                   help="当日跌幅超过此阈值时触发风险提醒。")
sim_num = st.sidebar.number_input("蒙特卡洛模拟次数", value=config.get("default_sim_count", 3000),
                                   help="模拟次数越多，结果越精确，但计算时间越长。")
news_count = st.sidebar.number_input("新闻获取数量", min_value=1, max_value=20,
                                     value=config.get("news_count", 5),
                                     help="每个标的获取的最新新闻数量。",
                                     key="news_count_input")

st.sidebar.markdown("---")

# 设置保存/加载 - 显示待保存提示
if st.session_state.get('_config_pending_save', False):
    st.sidebar.warning("⚠️ 有未保存的配置变更")

if st.sidebar.button("💾 保存设置", use_container_width=True):
    selected_codes = []
    for cat, stocks in st.session_state.master_pool.items():
        for c, n in stocks.items():
            if st.session_state.get(f"sel_{c}", True):
                selected_codes.append(c)

    holdings_save = {}
    selected_names = [st.session_state.master_pool[cat][c] for cat in st.session_state.master_pool for c in selected_codes if c in st.session_state.master_pool[cat]]
    for name in selected_names:
        holdings_save[name] = st.session_state.get(f"hold_{name}", 0)

    st.session_state.user_settings = {
        'selected_codes': selected_codes,
        'holdings': holdings_save,
        'stop_loss': stop_loss_val,
        'sim_num': sim_num,
        'news_count': news_count
    }

    if ds_key:
        config['deepseek_api_key'] = ds_key
    config['user_settings'] = st.session_state.user_settings
    config['news_count'] = news_count
    safe_save_config(config, force=True)  # 用户显式保存，强制执行
    st.session_state._config_pending_save = False  # 清除待保存标志
    st.sidebar.success("设置已保存！")

if st.sidebar.button("📂 加载设置", use_container_width=True):
    if st.session_state.user_settings:
        saved_codes = st.session_state.user_settings.get('selected_codes', [])
        for cat, stocks in st.session_state.master_pool.items():
            for c, n in stocks.items():
                st.session_state[f"sel_{c}"] = c in saved_codes

        saved_holdings = st.session_state.user_settings.get('holdings', {})
        for name, qty in saved_holdings.items():
            st.session_state[f"hold_{name}"] = qty

        saved_news_count = st.session_state.user_settings.get('news_count', 5)
        st.session_state["news_count_input"] = saved_news_count
        st.sidebar.success("设置已加载！")
        st.rerun()
    else:
        st.sidebar.warning("没有保存的设置")

# 刷新配置按钮（从磁盘重新加载 config.json）
if st.sidebar.button("🔄 刷新配置", use_container_width=True, help="重新从 config.json 加载配置（修改配置文件后使用）"):
    # 清除配置缓存
    if '_config_cache' in st.session_state:
        del st.session_state._config_cache
    # 重新加载配置
    fresh_config = load_config()
    st.session_state._config_cache = fresh_config
    st.session_state._config_dirty = False

    # 更新标的库
    config_master_pool = fresh_config.get('master_pool')
    if config_master_pool:
        st.session_state.master_pool = config_master_pool

    # 更新用户设置
    st.session_state.user_settings = fresh_config.get('user_settings', {})

    st.sidebar.success("✅ 配置已从文件重新加载！")
    st.rerun()

# 初始化选择状态存储（使用 _sel_ 前缀，与 checkbox key 分开）
if '_stock_selections' not in st.session_state:
    st.session_state._stock_selections = {}
    for cat, stocks in st.session_state.master_pool.items():
        for c in stocks.keys():
            st.session_state._stock_selections[c] = True

# 处理全选/取消全选
if st.session_state.get('_toggle_select_all', False):
    st.session_state['_toggle_select_all'] = False
    current_all_selected = all(st.session_state._stock_selections.get(c, True)
                               for cat in st.session_state.master_pool.values()
                               for c in cat.keys())
    new_state = not current_all_selected
    for cat, stocks in st.session_state.master_pool.items():
        for c in stocks.keys():
            st.session_state._stock_selections[c] = new_state
            # 同时更新 checkbox 的 key（在 widget 创建前可以修改）
            st.session_state[f"cb_{c}"] = new_state
    st.toast(f"已{'取消全选' if not new_state else '全选'}")

# 初始化 final_sel（标的选择将在主界面完成）
final_sel = {}
for cat, stocks in st.session_state.master_pool.items():
    for c, n in stocks.items():
        if st.session_state._stock_selections.get(c, True):
            final_sel[c] = n

# --- 主界面 ---
st.title("量化研报系统 V1.2")
st.caption(f"科研工作者专属调仓决策工具 | {datetime.now().strftime('%Y-%m-%d')}")

# ==================== 主界面标签页 ====================
main_tab1, main_tab2, main_tab3 = st.tabs(["标的选择与分析", "标的库管理", "交易录入"])

# ==================== TAB 1: 标的选择与分析 ====================
with main_tab1:
    # 分类管理控件
    with st.expander("📁 分类管理", expanded=False):
        col_add_cat, col_rename_cat, col_del_cat = st.columns(3)

        with col_add_cat:
            st.markdown("**新增分类**")
            new_cat_name = st.text_input("分类名称", placeholder="如 '观察池'", key="new_category_name")
            if st.button("➕ 添加分类", use_container_width=True, key="btn_add_category"):
                if new_cat_name and new_cat_name.strip():
                    cat_name = new_cat_name.strip()
                    if cat_name in st.session_state.master_pool:
                        st.warning(f"分类 [{cat_name}] 已存在")
                    else:
                        st.session_state.master_pool[cat_name] = {}
                        # 保存到配置文件
                        current_config = get_cached_config().copy()
                        current_config['master_pool'] = st.session_state.master_pool
                        safe_save_config(current_config, force=True)
                        st.success(f"已添加分类 [{cat_name}]")
                        st.rerun()
                else:
                    st.warning("请输入分类名称")

        with col_rename_cat:
            st.markdown("**修改分类名称**")
            existing_cats_for_rename = list(st.session_state.master_pool.keys())
            if existing_cats_for_rename:
                rename_cat = st.selectbox("选择分类", options=existing_cats_for_rename, key="rename_category_select")
                new_cat_name_input = st.text_input("新名称", placeholder="输入新名称", key="rename_category_name")
                if st.button("✏️ 修改名称", use_container_width=True, key="btn_rename_category"):
                    if new_cat_name_input and new_cat_name_input.strip():
                        new_name = new_cat_name_input.strip()
                        if new_name == rename_cat:
                            st.warning("新名称与原名称相同")
                        elif new_name in st.session_state.master_pool:
                            st.warning(f"分类 [{new_name}] 已存在")
                        else:
                            # 保留原分类的股票，用新名称创建
                            st.session_state.master_pool[new_name] = st.session_state.master_pool.pop(rename_cat)
                            # 保存到配置文件
                            current_config = get_cached_config().copy()
                            current_config['master_pool'] = st.session_state.master_pool
                            safe_save_config(current_config, force=True)
                            st.success(f"已将分类 [{rename_cat}] 修改为 [{new_name}]")
                            st.rerun()
                    else:
                        st.warning("请输入新名称")

        with col_del_cat:
            st.markdown("**删除分类**")
            existing_cats = list(st.session_state.master_pool.keys())
            if len(existing_cats) > 1:
                del_cat = st.selectbox("选择要删除的分类", options=existing_cats, key="del_category_select")

                # 检查该分类是否有股票
                stocks_in_cat = st.session_state.master_pool.get(del_cat, {})
                if stocks_in_cat:
                    # 有股票，需要选择目标分类
                    other_cats = [c for c in existing_cats if c != del_cat]
                    target_cat = st.selectbox(
                        f"将 {len(stocks_in_cat)} 只股票移动到",
                        options=other_cats,
                        key="target_category_select"
                    )
                    if st.button("🗑️ 删除并移动股票", use_container_width=True, key="btn_del_category"):
                        # 移动股票到目标分类
                        for code, name in stocks_in_cat.items():
                            if target_cat not in st.session_state.master_pool:
                                st.session_state.master_pool[target_cat] = {}
                            st.session_state.master_pool[target_cat][code] = name
                        # 删除原分类
                        del st.session_state.master_pool[del_cat]
                        # 保存到配置文件
                        current_config = get_cached_config().copy()
                        current_config['master_pool'] = st.session_state.master_pool
                        safe_save_config(current_config, force=True)
                        st.success(f"已删除分类 [{del_cat}]，{len(stocks_in_cat)} 只股票已移动到 [{target_cat}]")
                        st.rerun()
                else:
                    # 空分类，直接删除
                    if st.button("🗑️ 删除空分类", use_container_width=True, key="btn_del_empty_category"):
                        del st.session_state.master_pool[del_cat]
                        # 保存到配置文件
                        current_config = get_cached_config().copy()
                        current_config['master_pool'] = st.session_state.master_pool
                        safe_save_config(current_config, force=True)
                        st.success(f"已删除空分类 [{del_cat}]")
                        st.rerun()
            else:
                st.info("至少需要保留一个分类")

    st.markdown("---")

    # 标的选择区域 - 横向紧凑布局
    all_categories = list(st.session_state.master_pool.keys())

    # 分类全选/取消全选的回调函数
    def toggle_category_selection(category: str):
        stocks = st.session_state.master_pool[category]
        # 检查当前分类是否全选
        all_selected = all(
            st.session_state.get(f"cb_{c}", st.session_state._stock_selections.get(c, True))
            for c in stocks.keys()
        )
        new_state = not all_selected
        for c in stocks.keys():
            st.session_state._stock_selections[c] = new_state
            st.session_state[f"cb_{c}"] = new_state

    if all_categories:
        for cat in all_categories:
            stocks = st.session_state.master_pool[cat]
            # 分类标题和全选按钮紧贴放在一行
            cat_col1, cat_col2, _ = st.columns([2, 1, 7])
            with cat_col1:
                st.markdown(f"**{cat}**")
            with cat_col2:
                # 检查该分类是否全选，显示对应按钮文字
                cat_all_selected = all(
                    st.session_state.get(f"cb_{c}", st.session_state._stock_selections.get(c, True))
                    for c in stocks.keys()
                )
                btn_label = "取消全选" if cat_all_selected else "全选"
                st.button(
                    btn_label,
                    key=f"btn_cat_{cat}",
                    on_click=toggle_category_selection,
                    args=(cat,)
                )

            # 每行放置4-5个标的，根据标的数量动态调整
            stock_items = list(stocks.items())
            cols_per_row = 5  # 每行5个

            for row_start in range(0, len(stock_items), cols_per_row):
                row_items = stock_items[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)

                for col_idx, (c, n) in enumerate(row_items):
                    with cols[col_idx]:
                        cb_key = f"cb_{c}"
                        # 仅通过 session_state 管理 checkbox 状态（避免 value 参数与 session_state 冲突）
                        if cb_key not in st.session_state:
                            st.session_state[cb_key] = st.session_state._stock_selections.get(c, True)
                        checked = st.checkbox(f"{n}", key=cb_key, help=c)
                        # 同步更新状态字典
                        st.session_state._stock_selections[c] = checked
                        if checked:
                            final_sel[c] = n
                        else:
                            if c in final_sel:
                                del final_sel[c]

            # 分类之间添加分隔线
            st.divider()
    else:
        st.warning("标的库为空，请先在「标的库管理」中添加标的。")

    st.markdown("---")

    # 操作按钮区域
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        start_analysis = st.button("开始统计分析", type="primary", use_container_width=True)

    with col_btn2:
        refresh_data = st.button("刷新历史数据", type="secondary", use_container_width=True)

    with col_btn3:
        # 点击时设置标志，页面开头会处理实际逻辑
        st.button("全选/取消全选", use_container_width=True,
                  on_click=lambda: st.session_state.update({'_toggle_select_all': True}))

    # 刷新历史数据
    if refresh_data:
        selected_codes = [c for c, n in final_sel.items()]
        selected_names = [n for c, n in final_sel.items()]

        if not selected_codes:
            st.warning("请先勾选要更新的股票！")
        else:
            from market_store import get_market_store
            store = get_market_store()

            with st.spinner(f"正在增量更新 {len(selected_codes)} 只股票的历史数据..."):
                results = store.batch_update_recent_data(
                    codes=selected_codes,
                    names=selected_names,
                    recent_days=30,
                    max_workers=5
                )

            success_count = sum(1 for v in results.values() if v)
            failed_count = len(results) - success_count

            # 清除缓存
            keys_to_delete = [key for key in st.session_state.keys()
                             if key.startswith('_latest_prices_cache') or key.startswith('analysis_data_')]
            for key in keys_to_delete:
                del st.session_state[key]
            st.session_state.hist_data_cache = {}
            st.session_state.current_stocks_key = None
            for key in ['analysis_data', 'returns', 'mean_ret', 'cov_mat', 'sim_res',
                        'best_p', 'latest_prices', 'signals', 'stock_codes', 'stock_names']:
                if key in st.session_state:
                    del st.session_state[key]

            if failed_count == 0:
                st.success(f"增量更新完成！成功更新 {success_count} 只股票")
            else:
                st.warning(f"增量更新完成：成功 {success_count} 只，失败 {failed_count} 只")

    # 开始统计分析
    if start_analysis:
        selected_codes = [c for c in final_sel.keys()]
        holdings_save = {}
        selected_names = list(final_sel.values())
        for name in selected_names:
            holdings_save[name] = st.session_state.get(f"hold_{name}", 0)

        new_user_settings = {
            'selected_codes': selected_codes,
            'holdings': holdings_save,
            'stop_loss': stop_loss_val,
            'sim_num': sim_num,
            'news_count': news_count
        }

        # ✅ 优化：将配置变更暂存到 session_state，不在统计时触发文件写入
        # 这避免了 NAS 环境下因频繁 I/O 导致的页面跳动
        old_settings = st.session_state.get('user_settings', {})
        config_changed = (new_user_settings != old_settings or
                         (ds_key and config.get('deepseek_api_key') != ds_key))

        st.session_state.user_settings = new_user_settings

        # ❌ 不再在统计时保存配置文件
        # ✅ 改为标记配置待保存，用户可通过"保存设置"按钮手动保存
        if config_changed:
            st.session_state._config_pending_save = True
            # 仅更新内存中的配置缓存，不写入磁盘
            if ds_key:
                config['deepseek_api_key'] = ds_key
            config['user_settings'] = st.session_state.user_settings
            config['news_count'] = news_count
            st.session_state._config_cache = config  # 更新内存缓存

        # ✅ 标记统计完成，设置分析标志
        st.session_state.run_analysis = True
        st.session_state.selected_stocks = final_sel.copy()
        st.session_state.calc_done = True  # 标记统计已完成
        st.rerun()

# ==================== TAB 2: 标的库管理 ====================
with main_tab2:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("添加新标的")

        # 市场标的库状态
        market_info = get_market_stock_list_info()
        if market_info['exists']:
            st.success(f"本地市场库: {market_info['count']} 只股票 (更新: {market_info['update_time']})")
        else:
            st.warning("本地市场库未初始化")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("初始化市场库", use_container_width=True):
                with st.spinner("正在初始化..."):
                    result = init_market_stock_list_if_needed()
                    if result['success']:
                        st.success(f"成功！共 {result['count']} 只股票")
                        st.rerun()
                    else:
                        st.error(f"失败: {result['message']}")
        with col_m2:
            if st.button("刷新市场库", use_container_width=True):
                with st.spinner("正在刷新..."):
                    result = refresh_market_stock_list()
                    if result['success']:
                        st.success(f"成功！共 {result['count']} 只股票")
                        st.rerun()
                    else:
                        st.error(f"失败: {result['message']}")

        st.markdown("---")

        # 搜索添加
        search_query = st.text_input("搜索股票代码或名称", placeholder="如 '300347' 或 '泰格医药'", key="stock_search_main")

        if search_query and len(search_query.strip()) > 0:
            search_results = safe_search_stock_info(search_query.strip())
            if search_results:
                st.info(f"找到 {len(search_results)} 个匹配")
                stock_options = [f"{item['code']} - {item['name']}" for item in search_results]
                selected_option = st.selectbox("选择股票", options=stock_options, key="stock_select_main")

                if selected_option:
                    selected_code, selected_name = selected_option.split(" - ", 1)
                    col_cat, col_add = st.columns([2, 1])
                    with col_cat:
                        quick_category = st.selectbox("分类", ["核心组合", "科研/肾科", "其他"], key="quick_cat_main")
                    with col_add:
                        if st.button("添加", type="primary", use_container_width=True):
                            if quick_category not in st.session_state.master_pool:
                                st.session_state.master_pool[quick_category] = {}
                            existing = next((cat for cat, stocks in st.session_state.master_pool.items() if selected_code in stocks), None)
                            if existing:
                                st.warning(f"已在 [{existing}] 中")
                            else:
                                st.session_state.master_pool[quick_category][selected_code] = selected_name
                                # 自动保存到配置文件
                                current_config = get_cached_config().copy()
                                current_config['master_pool'] = st.session_state.master_pool
                                safe_save_config(current_config, force=True)
                                st.success(f"已添加 {selected_name} 并保存到配置")
                                st.rerun()
            else:
                st.warning("未找到匹配的股票")

        # 手动输入
        st.markdown("**手动输入**")
        col_c, col_n, col_t = st.columns(3)
        with col_c:
            new_code = st.text_input("代码", placeholder="000001", key="new_code_main")
        with col_n:
            new_name = st.text_input("名称", placeholder="平安银行", key="new_name_main")
        with col_t:
            new_category = st.selectbox("分类", ["核心组合", "科研/肾科", "其他"], key="new_cat_main")

        if st.button("手动添加标的", use_container_width=True):
            if new_code and new_name:
                if new_category not in st.session_state.master_pool:
                    st.session_state.master_pool[new_category] = {}
                existing = next((cat for cat, stocks in st.session_state.master_pool.items() if new_code in stocks), None)
                if existing:
                    st.warning(f"代码 {new_code} 已在 [{existing}] 中")
                else:
                    st.session_state.master_pool[new_category][new_code] = new_name
                    # 自动保存到配置文件
                    current_config = get_cached_config().copy()
                    current_config['master_pool'] = st.session_state.master_pool
                    safe_save_config(current_config, force=True)
                    st.success(f"已添加 {new_name}({new_code}) 并保存到配置")
                    st.rerun()
            else:
                st.warning("请填写代码和名称")

    with col_right:
        st.subheader("当前标的库")

        # 初始化移动状态
        if '_moving_stock' not in st.session_state:
            st.session_state._moving_stock = None  # 格式: {'code': ..., 'name': ..., 'from_cat': ...}

        for cat, stocks in list(st.session_state.master_pool.items()):
            with st.expander(f"{cat} ({len(stocks)} 只)", expanded=True):
                for c, n in list(stocks.items()):
                    # 检查当前标的是否正在移动
                    is_moving = (st.session_state._moving_stock is not None and
                                 st.session_state._moving_stock.get('code') == c and
                                 st.session_state._moving_stock.get('from_cat') == cat)

                    if is_moving:
                        # 移动模式：显示目标分类选择
                        st.markdown(f"**移动 {n}**")
                        other_cats = [c2 for c2 in st.session_state.master_pool.keys() if c2 != cat]
                        if other_cats:
                            col_target, col_confirm, col_cancel = st.columns([2, 1, 1])
                            with col_target:
                                target_cat = st.selectbox(
                                    "移动到",
                                    options=other_cats,
                                    key=f"move_target_{cat}_{c}",
                                    label_visibility="collapsed"
                                )
                            with col_confirm:
                                if st.button("✓", key=f"confirm_move_{cat}_{c}", help="确认移动"):
                                    # 执行移动
                                    if target_cat not in st.session_state.master_pool:
                                        st.session_state.master_pool[target_cat] = {}
                                    st.session_state.master_pool[target_cat][c] = n
                                    del st.session_state.master_pool[cat][c]
                                    if not st.session_state.master_pool[cat]:
                                        del st.session_state.master_pool[cat]
                                    # 保存配置
                                    current_config = get_cached_config().copy()
                                    current_config['master_pool'] = st.session_state.master_pool
                                    safe_save_config(current_config, force=True)
                                    st.session_state._moving_stock = None
                                    st.rerun()
                            with col_cancel:
                                if st.button("✗", key=f"cancel_move_{cat}_{c}", help="取消"):
                                    st.session_state._moving_stock = None
                                    st.rerun()
                        else:
                            st.info("没有其他分类可移动")
                            if st.button("取消", key=f"cancel_move_no_target_{cat}_{c}"):
                                st.session_state._moving_stock = None
                                st.rerun()
                    else:
                        # 正常模式：显示标的信息和操作按钮
                        col_info, col_move, col_del = st.columns([3, 1, 1])
                        with col_info:
                            st.text(f"{n} ({c})")
                        with col_move:
                            if st.button("↔", key=f"move_{cat}_{c}_main", help="移动到其他分类"):
                                st.session_state._moving_stock = {'code': c, 'name': n, 'from_cat': cat}
                                st.rerun()
                        with col_del:
                            if st.button("🗑", key=f"del_{cat}_{c}_main", help="删除"):
                                del st.session_state.master_pool[cat][c]
                                if not st.session_state.master_pool[cat]:
                                    del st.session_state.master_pool[cat]
                                # 自动保存到配置文件
                                current_config = get_cached_config().copy()
                                current_config['master_pool'] = st.session_state.master_pool
                                safe_save_config(current_config, force=True)
                                st.rerun()

        st.markdown("---")
        if st.button("保存标的库到配置", type="primary", use_container_width=True):
            # 使用缓存的配置，避免重复读取
            current_config = get_cached_config().copy()
            current_config['master_pool'] = st.session_state.master_pool
            safe_save_config(current_config, force=True)
            st.success("标的库已保存！")

# ==================== TAB 3: 交易录入 ====================
with main_tab3:
    col_input, col_list = st.columns([1, 1])

    with col_input:
        st.subheader("录入新交易")

        all_stocks = []
        for cat, stocks in st.session_state.master_pool.items():
            for code, name in stocks.items():
                all_stocks.append({"code": code, "name": name, "category": cat})

        if all_stocks:
            stock_options = [f"{item['code']} - {item['name']}" for item in all_stocks]
            selected_trade = st.selectbox("选择标的", options=stock_options, key="trade_select_main")

            if selected_trade:
                parts = selected_trade.split(" - ")
                trade_code = parts[0]
                trade_name = parts[1]

                col_p, col_q = st.columns(2)
                with col_p:
                    buy_price = st.number_input("买入均价 (元)", min_value=0.01, value=100.0, step=0.01, key="price_main")
                with col_q:
                    quantity = st.number_input("买入数量 (股)", min_value=1, value=100, step=100, key="qty_main")

                buy_date = st.date_input("买入日期", value=datetime.now(), key="date_main")

                if st.button("保存交易记录", type="primary", use_container_width=True):
                    if buy_price > 0 and quantity > 0:
                        trade_data = {
                            "code": trade_code,
                            "name": trade_name,
                            "buy_price": float(buy_price),
                            "quantity": int(quantity),
                            "buy_date": buy_date.strftime("%Y-%m-%d"),
                            "timestamp": int(datetime.now().timestamp())
                        }
                        config = get_cached_config().copy()
                        config = save_trade(config, trade_data)
                        # 更新缓存
                        st.session_state._config_cache = config
                        st.success("交易记录已保存！")
                        st.rerun()
                    else:
                        st.error("价格和数量必须大于0")
        else:
            st.info("请先在标的库中添加股票")

    with col_list:
        st.subheader("当前持仓记录")
        config = get_cached_config()
        portfolio = config.get('portfolio', [])

        # 初始化编辑状态
        if 'editing_trade_idx' not in st.session_state:
            st.session_state.editing_trade_idx = None

        if portfolio:
            for idx, trade in enumerate(portfolio):
                # 检查是否正在编辑此条记录
                is_editing = st.session_state.editing_trade_idx == idx

                if is_editing:
                    # 编辑模式
                    st.markdown(f"**编辑: {trade['name']}** ({trade['code']})")
                    col_ep, col_eq = st.columns(2)
                    with col_ep:
                        edit_price = st.number_input(
                            "买入均价",
                            min_value=0.01,
                            value=float(trade['buy_price']),
                            step=0.01,
                            key=f"edit_price_{idx}"
                        )
                    with col_eq:
                        edit_qty = st.number_input(
                            "买入数量",
                            min_value=1,
                            value=int(trade['quantity']),
                            step=100,
                            key=f"edit_qty_{idx}"
                        )

                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("保存", key=f"save_edit_{idx}", type="primary", use_container_width=True):
                            # 更新持仓记录
                            config['portfolio'][idx]['buy_price'] = float(edit_price)
                            config['portfolio'][idx]['quantity'] = int(edit_qty)
                            safe_save_config(config, force=True)
                            st.session_state.editing_trade_idx = None
                            st.rerun()
                    with col_cancel:
                        if st.button("取消", key=f"cancel_edit_{idx}", use_container_width=True):
                            st.session_state.editing_trade_idx = None
                            st.rerun()
                else:
                    # 显示模式
                    col_t1, col_t2, col_t3, col_t4 = st.columns([3, 2, 1, 1])
                    with col_t1:
                        st.markdown(f"**{trade['name']}** ({trade['code']})")
                        st.caption(f"{trade['quantity']}股 @ {trade['buy_price']}元")
                    with col_t2:
                        st.caption(trade['buy_date'])
                    with col_t3:
                        if st.button("修改", key=f"edit_trade_{idx}_main"):
                            st.session_state.editing_trade_idx = idx
                            st.rerun()
                    with col_t4:
                        if st.button("删除", key=f"del_trade_{idx}_main"):
                            config = delete_trade(config, idx)
                            st.rerun()

            st.markdown("---")
            if st.button("清空所有持仓", type="secondary", use_container_width=True):
                config = clear_portfolio(config)
                st.rerun()
        else:
            st.info("暂无持仓记录")

st.markdown("---")

# ==================== 分析结果显示区域 ====================
# 检查是否应该运行分析
should_run_analysis = st.session_state.get('run_analysis', False) and st.session_state.get('selected_stocks', {})
analysis_stocks = st.session_state.get('selected_stocks', {})

if should_run_analysis and analysis_stocks:
    # 生成缓存键：基于股票代码和名称的组合
    codes = list(analysis_stocks.keys())
    names = list(analysis_stocks.values())
    cache_key = f"{sorted(codes)}:{sorted(names)}"
    
    # 检查缓存是否有效：如果缓存键变化，则重新获取数据
    # 注意：增量更新已在"刷新历史数据"按钮处理时立即执行，这里只需从本地读取
    if (st.session_state.current_stocks_key != cache_key or
        cache_key not in st.session_state.hist_data_cache):

        # 使用fetch_stock_data函数获取数据（从本地Parquet读取，不触发增量更新）
        with st.spinner("正在加载历史行情数据..."):
            data = fetch_stock_data(codes, names, incremental_update=False)
        
        # 缓存数据
        if not data.empty:
            st.session_state.hist_data_cache[cache_key] = data
            st.session_state.current_stocks_key = cache_key
        else:
            # 如果数据为空，尝试使用兜底函数（这里我们直接使用缓存中的旧数据，如果有的话）
            if cache_key in st.session_state.hist_data_cache:
                st.warning("获取最新数据失败，使用缓存的历史数据。")
                data = st.session_state.hist_data_cache[cache_key]
            else:
                st.error("无法获取股票数据，请检查网络连接或股票代码。")
                st.stop()
    else:
        # 使用缓存的数据
        data = st.session_state.hist_data_cache[cache_key]
        st.success(f"使用缓存的历史数据（共 {len(data)} 个交易日）")
    
    if not data.empty:
        # 检查实际获取到的股票数据，更新analysis_stocks
        actual_stocks = {}
        for code, name in analysis_stocks.items():
            if name in data.columns:
                actual_stocks[code] = name
            else:
                st.warning(f"股票 {name}({code}) 数据获取失败，已从分析中排除。可能是网络问题或该股票代码暂时无法获取数据，请检查网络连接或稍后重试。")
        # 如果没有成功获取到任何股票，则报错并停止
        if len(actual_stocks) == 0:
            st.error("未能获取到任何股票数据，请检查网络连接或股票代码。")
            st.stop()
        # 更新analysis_stocks为实际获取到的股票
        analysis_stocks = actual_stocks
        # 确保data只包含actual_stocks中的股票（实际上已经如此，但为了安全）
        data = data[list(actual_stocks.values())]
        # 计算收益率
        returns = data.pct_change().dropna()
        # 投资组合统计
        mean_ret, cov_mat = compute_portfolio_stats(returns, config.get("risk_free_rate", 0.0188))
        # 蒙特卡洛模拟
        sim_res = monte_carlo_simulation(mean_ret, cov_mat, sim_num, config.get("risk_free_rate", 0.0188))
        best_p = sim_res.iloc[sim_res['Sharpe'].idxmax()]

        # 获取最新价格和交易信号（供子页面使用）
        stock_codes = list(analysis_stocks.keys())
        stock_names = list(analysis_stocks.values())
        latest_prices = safe_fetch_latest_prices(stock_codes, stock_names)
        signals = get_signals(data, best_p.to_dict(), stop_loss_val, latest_prices)

        # ✅ 保存完整分析结果到session_state，供其他页面使用（数据接力模式）
        st.session_state.analysis_data = data
        st.session_state.returns = returns
        st.session_state.mean_ret = mean_ret
        st.session_state.cov_mat = cov_mat
        st.session_state.sim_res = sim_res
        st.session_state.best_p = best_p
        st.session_state.latest_prices = latest_prices
        st.session_state.signals = signals
        st.session_state.stock_codes = stock_codes
        st.session_state.stock_names = stock_names

        # 分析结果概要
        st.subheader("分析结果概要")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("分析标的数", f"{len(analysis_stocks)}")
        with col2:
            st.metric("预期年化收益", f"{best_p['Ret']:.2%}")
        with col3:
            st.metric("夏普比率", f"{best_p['Sharpe']:.2f}")
        with col4:
            st.metric("组合波动率", f"{best_p['Vol']:.2%}")

        st.markdown("---")

        # 快速导航 - 精简版
        st.subheader("功能模块导航")

        pages_map = [
            {"name": "AI资讯研判", "icon": "💡", "path": "pages/01_ai_news.py"},
            {"name": "买卖风险", "icon": "🚦", "path": "pages/02_signals_risk.py"},
            {"name": "相关性分析", "icon": "🕸️", "path": "pages/03_corr.py"},
            {"name": "权重优化", "icon": "📊", "path": "pages/04_portfolio_opt.py"},
            {"name": "模拟交易", "icon": "📈", "path": "pages/05_sim_trade.py"},
            {"name": "实盘监测", "icon": "💼", "path": "pages/06_live_portfolio.py"},
            {"name": "K线分析", "icon": "📉", "path": "pages/07_kline_lab.py"},
            {"name": "优先级雷达", "icon": "🎯", "path": "pages/08_priority_radar.py"},
            {"name": "宏观审计", "icon": "🌍", "path": "pages/09_macro_audit.py"},
            {"name": "数据检查", "icon": "🔍", "path": "pages/10_data_integrity.py"}
        ]

        # 两行五列布局
        cols1 = st.columns(5)
        cols2 = st.columns(5)

        for i, page in enumerate(pages_map):
            col = cols1[i] if i < 5 else cols2[i-5]
            with col:
                if st.button(f"{page['icon']} {page['name']}", key=f"nav_{i}", use_container_width=True):
                    st.switch_page(page['path'])

    else:
        st.error("网络连接异常，无法获取行情。")
else:
    st.info("请在上方「标的选择与分析」标签页中勾选标的，然后点击「开始统计分析」按钮。")
