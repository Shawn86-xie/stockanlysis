import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from data_fetcher import rank_master_pool
from config import load_config
from ai_analyzer import deepseek_analyze
from services.data_service import DataServiceError

def weights_equal(w1, w2, tolerance=0.001):
    """
    比较两个权重字典是否相等（在容差范围内）
    """
    if w1 is None or w2 is None:
        return False
    for key in set(w1.keys()) | set(w2.keys()):
        if abs(w1.get(key, 0) - w2.get(key, 0)) > tolerance:
            return False
    return True

# AI分析记录持久化存储路径
AI_ANALYSIS_HISTORY_FILE = Path("data/ai_analysis_history.json")

def load_ai_analysis_history():
    """
    从本地文件加载AI分析历史记录

    数据结构（v2）：
    {
        "股票代码": {
            "name": "股票名称",
            "records": [
                {"analysis": "分析内容", "timestamp": "2026-01-18 10:30:00", "id": "unique_id"},
                ...
            ]
        }
    }
    """
    if AI_ANALYSIS_HISTORY_FILE.exists():
        try:
            with open(AI_ANALYSIS_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 兼容旧版数据格式（v1 -> v2 迁移）
                return migrate_history_format(data)
        except Exception as e:
            print(f"加载AI分析历史记录失败: {e}")
            return {}
    return {}


def migrate_history_format(data):
    """
    将旧版数据格式迁移到新版

    旧版格式：{code: {name, analysis, timestamp}}
    新版格式：{code: {name, records: [{analysis, timestamp, id}, ...]}}
    """
    if not data:
        return {}

    migrated = {}
    for code, value in data.items():
        if isinstance(value, dict):
            # 检查是否已经是新格式
            if 'records' in value:
                migrated[code] = value
            else:
                # 旧格式，需要迁移
                import uuid
                migrated[code] = {
                    'name': value.get('name', code),
                    'records': [{
                        'analysis': value.get('analysis', ''),
                        'timestamp': value.get('timestamp', ''),
                        'id': str(uuid.uuid4())[:8]  # 生成唯一ID
                    }]
                }
    return migrated


def save_ai_analysis_history(analysis_results):
    """保存AI分析历史记录到本地文件"""
    try:
        # 确保data目录存在
        AI_ANALYSIS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(AI_ANALYSIS_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存AI分析历史记录失败: {e}")
        return False


def add_analysis_record(code, name, analysis):
    """
    添加一条分析记录（不覆盖历史记录）

    Args:
        code: 股票代码
        name: 股票名称
        analysis: 分析内容

    Returns:
        str: 新记录的ID
    """
    import uuid

    if 'ai_analysis_results' not in st.session_state:
        st.session_state.ai_analysis_results = {}

    record_id = str(uuid.uuid4())[:8]
    new_record = {
        'analysis': analysis,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'id': record_id
    }

    if code in st.session_state.ai_analysis_results:
        # 已有该股票的记录，追加新记录
        st.session_state.ai_analysis_results[code]['records'].append(new_record)
    else:
        # 新股票，创建记录结构
        st.session_state.ai_analysis_results[code] = {
            'name': name,
            'records': [new_record]
        }

    # 持久化保存
    save_ai_analysis_history(st.session_state.ai_analysis_results)
    return record_id


def delete_analysis_record(code, record_id):
    """
    删除一条分析记录

    Args:
        code: 股票代码
        record_id: 记录ID

    Returns:
        bool: 是否删除成功
    """
    if 'ai_analysis_results' not in st.session_state:
        return False

    if code not in st.session_state.ai_analysis_results:
        return False

    records = st.session_state.ai_analysis_results[code]['records']
    # 过滤掉要删除的记录
    new_records = [r for r in records if r.get('id') != record_id]

    if len(new_records) == len(records):
        return False  # 没有找到匹配的记录

    if new_records:
        st.session_state.ai_analysis_results[code]['records'] = new_records
    else:
        # 如果该股票没有记录了，删除整个条目
        del st.session_state.ai_analysis_results[code]

    # 持久化保存
    save_ai_analysis_history(st.session_state.ai_analysis_results)
    return True


def get_total_record_count():
    """获取总记录数"""
    if 'ai_analysis_results' not in st.session_state:
        return 0
    total = 0
    for code, data in st.session_state.ai_analysis_results.items():
        if 'records' in data:
            total += len(data['records'])
    return total

# 设置页面配置
st.set_page_config(page_title="标的优先度雷达", layout="wide")

# 初始化AI分析历史记录（从文件加载，支持持久化）
if 'ai_analysis_results' not in st.session_state:
    st.session_state.ai_analysis_results = load_ai_analysis_history()

# 检查是否有master_pool
if 'master_pool' not in st.session_state:
    st.warning("请先返回主页面配置标的库并开始分析。")
    st.stop()

# 加载配置
config = load_config()

# 页面标题
st.title("🏆 标的优先度雷达")
st.caption("基于线性回归拟合及动能算法，对标的库中的股票进行综合评分排序")

# 数据审计状态栏
with st.expander("🔍 数据审计状态", expanded=False):
    from storage_manager import show_data_audit_ui
    show_data_audit_ui()

# ==================== 策略实验室：多因子敏感度调优 ====================
st.subheader("🔬 策略实验室：多因子敏感度调优")
st.caption("选择预设模型或自定义权重，调整不同因子的重要性，观察排名变化")

# 预设权重模型
PRESET_MODELS = {
    "均衡稳健型": {"k": 0.35, "r2": 0.35, "dist": 0.20, "vol": 0.10},
    "动能黑马型": {"k": 0.60, "r2": 0.10, "dist": 0.10, "vol": 0.20},
    "趋势长牛型": {"k": 0.20, "r2": 0.60, "dist": 0.10, "vol": 0.10},
    "突破突击型": {"k": 0.20, "r2": 0.10, "dist": 0.50, "vol": 0.20},
    "自定义": None
}

# 初始化session_state中的权重
if 'current_weights' not in st.session_state:
    st.session_state.current_weights = PRESET_MODELS["均衡稳健型"]
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "均衡稳健型"

# 模型选择器
col1, col2 = st.columns([2, 1])
with col1:
    selected_model = st.radio(
        "选择权重模型",
        options=list(PRESET_MODELS.keys()),
        index=list(PRESET_MODELS.keys()).index(st.session_state.selected_model),
        horizontal=True,
        key="model_selector"
    )

# 如果模型改变，更新权重
if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    if selected_model != "自定义":
        st.session_state.current_weights = PRESET_MODELS[selected_model]
    st.rerun()

# 自定义权重滑块
if selected_model == "自定义":
    st.markdown("#### 🎛️ 自定义因子权重（总和自动调整至100%）")
    
    # 初始化自定义权重
    if 'custom_weights' not in st.session_state:
        st.session_state.custom_weights = {"k": 35, "r2": 35, "dist": 20, "vol": 10}
    
    # 创建四个滑块
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        k_weight = st.slider("相对斜率权重 (%)", 0, 100, st.session_state.custom_weights["k"], 5, 
                           help="趋势强度权重", key="k_slider")
    with col2:
        r2_weight = st.slider("趋势稳定性权重 (%)", 0, 100, st.session_state.custom_weights["r2"], 5,
                            help="拟合优度权重", key="r2_slider")
    with col3:
        dist_weight = st.slider("距上轨距离权重 (%)", 0, 100, st.session_state.custom_weights["dist"], 5,
                              help="突破概率权重", key="dist_slider")
    with col4:
        vol_weight = st.slider("量能强度权重 (%)", 0, 100, st.session_state.custom_weights["vol"], 5,
                             help="成交量权重", key="vol_slider")
    
    # 计算总权重
    total = k_weight + r2_weight + dist_weight + vol_weight
    
    # 如果总权重不为100，自动调整
    if total != 100:
        # 按比例调整
        if total > 0:
            k_weight = int(k_weight * 100 / total)
            r2_weight = int(r2_weight * 100 / total)
            dist_weight = int(dist_weight * 100 / total)
            vol_weight = 100 - k_weight - r2_weight - dist_weight
        else:
            # 如果全为0，设置为默认值
            k_weight, r2_weight, dist_weight, vol_weight = 35, 35, 20, 10
        
        # 更新session_state
        st.session_state.custom_weights = {
            "k": k_weight, 
            "r2": r2_weight, 
            "dist": dist_weight, 
            "vol": vol_weight
        }
        st.rerun()
    
    # 保存自定义权重
    st.session_state.custom_weights = {
        "k": k_weight, 
        "r2": r2_weight, 
        "dist": dist_weight, 
        "vol": vol_weight
    }
    
    # 转换为小数格式
    st.session_state.current_weights = {
        "k": k_weight / 100,
        "r2": r2_weight / 100,
        "dist": dist_weight / 100,
        "vol": vol_weight / 100
    }
    
    # 显示当前权重
    st.success(f"✅ 当前权重: 相对斜率 {k_weight}% | 趋势稳定性 {r2_weight}% | 距上轨距离 {dist_weight}% | 量能强度 {vol_weight}%")
else:
    # 显示预设模型的权重
    weights = st.session_state.current_weights
    k_pct = int(weights["k"] * 100)
    r2_pct = int(weights["r2"] * 100)
    dist_pct = int(weights["dist"] * 100)
    vol_pct = int(weights["vol"] * 100)
    st.info(f"📊 **{selected_model}** 权重分配: 相对斜率 {k_pct}% | 趋势稳定性 {r2_pct}% | 距上轨距离 {dist_pct}% | 量能强度 {vol_pct}%")

st.markdown("---")

# 获取master_pool
master_pool = st.session_state.master_pool

# 调用rank_master_pool函数获取排名数据（使用当前权重）
with st.spinner(f"正在使用「{selected_model}」模型计算标的综合评分..."):
    # 将master_pool转换为rank_master_pool所需的格式
    pool_for_ranking = {}
    for category, stocks in master_pool.items():
        stock_list = []
        for code, name in stocks.items():
            stock_list.append({'code': code, 'name': name})
        pool_for_ranking[category] = stock_list
    
    # 获取排名数据，传递权重参数
    ranking_df = rank_master_pool(pool_for_ranking, weights=st.session_state.current_weights)

# 检查是否成功获取数据
if ranking_df.empty:
    st.error("未能获取标的排名数据，请检查数据源。")
    st.stop()

# 1. 顶部统计卡片
st.subheader("📊 统计概览")

# 计算上升标的占比 (k>0)
rising_count = (ranking_df['斜率(k)'] > 0).sum()
total_count = len(ranking_df)
rising_ratio = rising_count / total_count if total_count > 0 else 0

# 计算高稳定性标的数量 (R² > 0.8)
stable_count = (ranking_df['R²'] > 0.8).sum()

# 创建两列显示统计卡片
col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="库内上升标的占比",
        value=f"{rising_ratio:.1%}",
        delta=f"{rising_count}/{total_count} 只"
    )
with col2:
    st.metric(
        label="高稳定性标的数量",
        value=f"{stable_count} 只",
        delta=f"R² > 0.8"
    )

st.markdown("---")

# 2. 核心排名表
st.subheader("📈 综合诊断排名")

# 对排名表进行视觉强化
# 首先复制一份数据用于显示
display_df = ranking_df.copy()

# 添加🔥前缀给综合得分前3名
top3_indices = display_df['综合评分'].nlargest(3).index
for idx in top3_indices:
    display_df.loc[idx, '名称'] = f"🔥 {display_df.loc[idx, '名称']}"

# 重新排列列顺序以符合要求
# 原列：['分类', '代码', '名称', '斜率(k)', '归一化斜率(k_rel)', 'R²', '距上轨距离', '距离权重', '量能强度', '量能权重', '综合评分']
# 需要展示的列：标的名称、相对斜率、趋势稳定性(R²)、距上轨(%)、量能强度、综合诊断得分
# 我们选择：名称、归一化斜率(k_rel)、R²、距上轨距离、量能强度、综合评分
selected_columns = ['名称', '归一化斜率(k_rel)', 'R²', '距上轨距离', '量能强度', '综合评分']
renamed_columns = {
    '名称': '标的名称',
    '归一化斜率(k_rel)': '相对斜率',
    'R²': '趋势稳定性',
    '距上轨距离': '距上轨(%)',
    '量能强度': '量能强度',
    '综合评分': '综合诊断得分'
}

# 创建显示用的DataFrame
show_df = display_df[selected_columns].copy()
show_df = show_df.rename(columns=renamed_columns)

# 格式化数值显示
def format_values(val):
    if isinstance(val, (int, float)):
        if abs(val) < 0.001:
            return f"{val:.2e}"
        elif abs(val) < 0.01:
            return f"{val:.4f}"
        else:
            return f"{val:.3f}"
    return val

# 应用格式化
for col in ['相对斜率', '趋势稳定性', '距上轨(%)', '量能强度', '综合诊断得分']:
    show_df[col] = show_df[col].apply(format_values)

# 定义样式函数
def highlight_r2(val):
    """对R²>0.75的单元格加粗显示"""
    try:
        if float(val) > 0.75:
            return 'font-weight: bold'
    except:
        pass
    return ''

def highlight_distance(val):
    """对距上轨距离在[0%, +2%]之间的单元格标记为浅橙色，文字颜色为深色以提高可读性"""
    try:
        v = float(val)
        if 0 <= v <= 0.02:
            # 浅橙色背景，深色文字确保可读性
            return 'background-color: #fff3cd; color: #333333; font-weight: bold;'
    except:
        pass
    return ''

# 应用样式
styled_df = show_df.style.applymap(highlight_r2, subset=['趋势稳定性'])
styled_df = styled_df.applymap(highlight_distance, subset=['距上轨(%)'])

# 显示表格
st.dataframe(styled_df, width='stretch', height=400)

st.markdown("---")

# 3. 联动功能
st.subheader("🔗 快速跳转分析")

# 创建选择框，列出所有标的
stock_options = ranking_df['名称'].tolist()
selected_stock = st.selectbox(
    "选择标的进行深度分析",
    options=stock_options,
    index=0,
    help="选择后点击下方按钮跳转到对应分析页面"
)

# 获取选中标的的代码
selected_row = ranking_df[ranking_df['名称'] == selected_stock.replace('🔥 ', '')]
if not selected_row.empty:
    selected_code = selected_row.iloc[0]['代码']
    selected_name = selected_row.iloc[0]['名称']
    
    col1, col2 = st.columns(2)
    with col1:
        # 跳转到独立K线分析页面（使用按钮+switch_page实现参数传递）
        if st.button(f"📈 独立K线分析 - {selected_name}", key="goto_kline", use_container_width=True):
            # 保存选中的标的到 session_state，供目标页面读取
            st.session_state.kline_jump_target = {
                'code': selected_code,
                'name': selected_name
            }
            st.switch_page("pages/07_kline_lab.py")
    with col2:
        # 跳转到AI资讯分析页面
        if st.button(f"💡 AI资讯分析 - {selected_name}", key="goto_ai_news", use_container_width=True):
            st.session_state.ai_news_jump_target = {
                'code': selected_code,
                'name': selected_name
            }
            st.switch_page("pages/01_ai_news.py")
    
    # 显示选中标的的详细指标
    with st.expander(f"📊 {selected_name} 详细指标", expanded=False):
        detail_cols = st.columns(3)
        with detail_cols[0]:
            st.metric("相对斜率", f"{selected_row.iloc[0]['归一化斜率(k_rel)']:.4f}")
            st.metric("斜率(k)", f"{selected_row.iloc[0]['斜率(k)']:.4f}")
        with detail_cols[1]:
            st.metric("趋势稳定性 (R²)", f"{selected_row.iloc[0]['R²']:.4f}")
            st.metric("距上轨距离", f"{selected_row.iloc[0]['距上轨距离']:.2%}")
        with detail_cols[2]:
            st.metric("量能强度", f"{selected_row.iloc[0]['量能强度']:.2f}")
            st.metric("综合评分", f"{selected_row.iloc[0]['综合评分']:.4f}")
        
        # AI综合解读与分析
        st.markdown("---")
        st.subheader("🤖 AI综合解读与分析")
        
        # 准备技术指标数据
        tech_data = {
            '股票名称': selected_name,
            '股票代码': selected_code,
            '相对斜率': selected_row.iloc[0]['归一化斜率(k_rel)'],
            '斜率(k)': selected_row.iloc[0]['斜率(k)'],
            '趋势稳定性_R2': selected_row.iloc[0]['R²'],
            '距上轨距离': selected_row.iloc[0]['距上轨距离'],
            '量能强度': selected_row.iloc[0]['量能强度'],
            '综合评分': selected_row.iloc[0]['综合评分']
        }
        
        # 构建分析提示词
        analysis_prompt = f"""
作为资深金融分析师，请对以下股票的技术指标进行综合解读：

股票名称：{tech_data['股票名称']} ({tech_data['股票代码']})
技术指标：
1. 相对斜率：{tech_data['相对斜率']:.4f} (归一化后的趋势强度)
2. 斜率(k)：{tech_data['斜率(k)']:.4f} (阻力线斜率，正值为上升趋势)
3. 趋势稳定性(R²)：{tech_data['趋势稳定性_R2']:.4f} (拟合优度，0-1之间，越高越稳定)
4. 距上轨距离：{tech_data['距上轨距离']:.2%} (距离阻力线的百分比，负值表示已突破，0-2%为突破临界区)
5. 量能强度：{tech_data['量能强度']:.2f} (当日成交量/5日均量，>1表示放量)
6. 综合评分：{tech_data['综合评分']:.4f} (综合评分，越高越好)

请从以下角度提供专业分析（每点不超过50字）：
1. 趋势判断：基于斜率和R²，当前处于什么趋势阶段？
2. 突破概率：基于距上轨距离和量能强度，近期突破可能性如何？
3. 风险提示：主要风险点有哪些？
4. 操作建议：给出具体的操作建议（买入/持有/卖出/观望）。

请用简洁专业的中文回答，避免使用markdown格式，直接使用纯文本。
"""
        
        # 获取API密钥
        api_key = config.get("deepseek_api_key", "")
        
        if not api_key:
            st.warning("⚠️ 未配置DeepSeek API密钥，无法进行AI分析。请在主页面侧边栏配置API密钥。")
        else:
            if st.button("🚀 启动AI技术分析", type="primary", key=f"ai_analysis_{selected_code}"):
                with st.spinner("AI正在分析技术指标，请稍候..."):
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                        
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
                        st.markdown("#### 📋 AI分析结果：")
                        st.info(ai_analysis)

                        # ✅ 使用新的追加式保存（不覆盖历史记录）
                        record_id = add_analysis_record(selected_code, selected_name, ai_analysis)
                        st.caption(f"✅ 分析记录已保存 (ID: {record_id})")

                    except Exception as e:
                        st.error(f"AI分析失败：{e}")
            
            # 显示该标的的历史分析结果（如果有）
            if 'ai_analysis_results' in st.session_state and selected_code in st.session_state.ai_analysis_results:
                stock_history = st.session_state.ai_analysis_results[selected_code]
                records = stock_history.get('records', [])
                if records:
                    st.markdown(f"#### 📜 历史分析记录 ({len(records)}条)")
                    # 按时间倒序显示
                    for record in reversed(records):
                        with st.container():
                            col_time, col_del = st.columns([4, 1])
                            with col_time:
                                st.caption(f"🕒 {record.get('timestamp', '未知时间')} | ID: {record.get('id', 'N/A')}")
                            with col_del:
                                if st.button("🗑️", key=f"del_{selected_code}_{record.get('id')}", help="删除此条记录"):
                                    if delete_analysis_record(selected_code, record.get('id')):
                                        st.success("已删除")
                                        st.rerun()
                            st.info(record.get('analysis', ''))

st.markdown("---")

# 4. 历史记录管理
st.subheader("📚 历史分析记录管理")
col_info1, col_info2 = st.columns([3, 1])
with col_info1:
    st.caption(f"历史分析记录已自动保存到本地文件：`{AI_ANALYSIS_HISTORY_FILE}`，重启应用后可自动恢复。")
with col_info2:
    if st.button("🔄 从文件重新加载", type="secondary", key="reload_history"):
        st.session_state.ai_analysis_results = load_ai_analysis_history()
        st.success("已从本地文件重新加载历史记录")
        st.rerun()

if 'ai_analysis_results' in st.session_state and st.session_state.ai_analysis_results:
    # 统计信息
    total_stocks = len(st.session_state.ai_analysis_results)
    total_records = get_total_record_count()
    st.info(f"📊 共 **{total_stocks}** 个标的，**{total_records}** 条分析记录")

    # 构建汇总表格数据
    summary_data = []
    for code, data in st.session_state.ai_analysis_results.items():
        records = data.get('records', [])
        if records:
            # 获取最新和最早的时间
            timestamps = [r.get('timestamp', '') for r in records]
            latest_time = max(timestamps) if timestamps else ''
            summary_data.append({
                '股票代码': code,
                '股票名称': data.get('name', code),
                '记录数': len(records),
                '最新分析时间': latest_time
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # 按最新分析时间倒序排列
        summary_df = summary_df.sort_values('最新分析时间', ascending=False)
        st.dataframe(summary_df, width='stretch', height=200)

    # 按标的汇总展示详细记录
    st.markdown("#### 📖 按标的查看详细记录")

    for code, data in st.session_state.ai_analysis_results.items():
        stock_name = data.get('name', code)
        records = data.get('records', [])
        if records:
            with st.expander(f"📌 {stock_name} ({code}) - {len(records)}条记录", expanded=False):
                # 按时间倒序显示每条记录
                for record in reversed(records):
                    record_id = record.get('id', 'N/A')
                    timestamp = record.get('timestamp', '未知时间')
                    analysis = record.get('analysis', '')

                    # 每条记录的头部：时间和删除按钮
                    col_time, col_del = st.columns([5, 1])
                    with col_time:
                        st.caption(f"🕒 {timestamp} | ID: {record_id}")
                    with col_del:
                        if st.button("🗑️ 删除", key=f"mgmt_del_{code}_{record_id}", help="删除此条记录"):
                            if delete_analysis_record(code, record_id):
                                st.success("已删除")
                                st.rerun()

                    # 分析内容
                    st.info(analysis)
                    st.markdown("---")

    # 管理功能
    st.markdown("#### ⚙️ 批量管理")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ 清空所有历史记录", type="secondary"):
            st.session_state.ai_analysis_results = {}
            # 同步删除本地文件
            if save_ai_analysis_history({}):
                st.success("已清空所有历史记录（包括本地文件）")
            else:
                st.warning("已清空会话记录，但本地文件删除失败")
            st.rerun()
    with col2:
        # 导出为CSV（展开所有记录）
        export_data = []
        for code, data in st.session_state.ai_analysis_results.items():
            stock_name = data.get('name', code)
            for record in data.get('records', []):
                export_data.append({
                    '股票代码': code,
                    '股票名称': stock_name,
                    '分析时间': record.get('timestamp', ''),
                    '记录ID': record.get('id', ''),
                    '分析内容': record.get('analysis', '')
                })
        if export_data:
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 导出CSV",
                data=csv,
                file_name="ai_analysis_history.csv",
                mime="text/csv",
                key="download_csv"
            )
    with col3:
        # 导出为JSON
        json_str = json.dumps(st.session_state.ai_analysis_results, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 导出JSON",
            data=json_str,
            file_name="ai_analysis_history.json",
            mime="application/json",
            key="download_json"
        )
else:
    st.info("暂无历史分析记录。请先对某些标的进行AI分析。")

st.markdown("---")

# 5. 提示语
st.caption("📝 **基于线性回归拟合及动能算法，仅供科研决策参考**")
st.caption("""
- **相对斜率**：归一化后的斜率，用于公平比较不同价位标的趋势强度
- **趋势稳定性 (R²)**：拟合优度，越高表示趋势越稳定
- **距上轨(%)**：最新收盘价距离阻力线的百分比，越小表示越接近突破
- **量能强度**：当日成交量与过去5日平均成交量的比值
- **综合诊断得分**：综合各项指标计算的最终评分，用于标的排序
""")
