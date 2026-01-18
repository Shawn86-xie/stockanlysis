"""
数据完整性检查与修复工具
专门用于检测和修复历史数据缺失问题，特别是昨天数据缺失的情况
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import time
import os
from pathlib import Path

# 导入本地行情仓库
from market_store import get_market_store


def get_last_trading_day(reference_date=None):
    """
    获取最近的交易日（排除周末和已知节假日）

    Args:
        reference_date: 参考日期，默认为今天

    Returns:
        date: 最近的交易日日期
    """
    if reference_date is None:
        reference_date = datetime.now().date()
    elif isinstance(reference_date, datetime):
        reference_date = reference_date.date()

    # 已知的A股节假日（2025-2026年主要节假日）
    # 注意：这里只列举主要节假日，实际生产中建议使用专门的交易日历API
    known_holidays = {
        # 2025年节假日
        datetime(2025, 1, 1).date(),   # 元旦
        datetime(2025, 1, 28).date(),  # 春节
        datetime(2025, 1, 29).date(),
        datetime(2025, 1, 30).date(),
        datetime(2025, 1, 31).date(),
        datetime(2025, 2, 1).date(),
        datetime(2025, 2, 2).date(),
        datetime(2025, 2, 3).date(),
        datetime(2025, 2, 4).date(),
        datetime(2025, 4, 4).date(),   # 清明节
        datetime(2025, 5, 1).date(),   # 劳动节
        datetime(2025, 5, 2).date(),
        datetime(2025, 5, 5).date(),
        datetime(2025, 5, 31).date(),  # 端午节
        datetime(2025, 6, 2).date(),
        datetime(2025, 10, 1).date(),  # 国庆节
        datetime(2025, 10, 2).date(),
        datetime(2025, 10, 3).date(),
        datetime(2025, 10, 6).date(),
        datetime(2025, 10, 7).date(),
        # 2026年节假日
        datetime(2026, 1, 1).date(),   # 元旦
        datetime(2026, 1, 2).date(),
        datetime(2026, 2, 16).date(),  # 春节（预估）
        datetime(2026, 2, 17).date(),
        datetime(2026, 2, 18).date(),
        datetime(2026, 2, 19).date(),
        datetime(2026, 2, 20).date(),
        datetime(2026, 2, 21).date(),
        datetime(2026, 2, 22).date(),
        datetime(2026, 2, 23).date(),
        datetime(2026, 4, 5).date(),   # 清明节（预估）
        datetime(2026, 4, 6).date(),
        datetime(2026, 5, 1).date(),   # 劳动节
        datetime(2026, 5, 4).date(),
        datetime(2026, 5, 5).date(),
        datetime(2026, 6, 19).date(),  # 端午节（预估）
        datetime(2026, 6, 20).date(),
        datetime(2026, 10, 1).date(),  # 国庆节
        datetime(2026, 10, 2).date(),
        datetime(2026, 10, 5).date(),
        datetime(2026, 10, 6).date(),
        datetime(2026, 10, 7).date(),
        datetime(2026, 10, 8).date(),
    }

    check_date = reference_date
    # 最多回退30天寻找交易日
    for _ in range(30):
        # 检查是否为周末
        if check_date.weekday() >= 5:  # 周六=5, 周日=6
            check_date -= timedelta(days=1)
            continue
        # 检查是否为已知节假日
        if check_date in known_holidays:
            check_date -= timedelta(days=1)
            continue
        # 找到交易日
        return check_date

    # 如果30天内都没找到，返回参考日期（理论上不会发生）
    return reference_date


def is_trading_day(check_date):
    """
    判断指定日期是否为交易日

    Args:
        check_date: 要检查的日期

    Returns:
        bool: True表示是交易日，False表示非交易日
    """
    if isinstance(check_date, datetime):
        check_date = check_date.date()

    # 周末不是交易日
    if check_date.weekday() >= 5:
        return False

    # 检查是否为节假日
    last_trading = get_last_trading_day(check_date)
    return last_trading == check_date

def check_data_integrity(codes, names):
    """
    检查数据完整性，特别关注最近日期的数据
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
    
    Returns:
        dict: 检查结果，包含缺失情况、最后日期等信息
    """
    store = get_market_store()
    results = {
        'total_stocks': len(codes),
        'checked_stocks': 0,
        'missing_yesterday': [],
        'missing_recent': [],
        'details': {}
    }
    
    today = datetime.now().date()
    # ✅ 使用交易日判断：获取最近的交易日，而非简单的昨天
    last_trading_day = get_last_trading_day(today - timedelta(days=1))
    # 检查最近5个交易日（排除周末和节假日）
    recent_days = 5

    for code, name in zip(codes, names):
        try:
            # 读取数据
            df = store.read_stock_data(code, update_if_needed=False)

            if df.empty:
                results['details'][code] = {
                    'name': name,
                    'status': 'empty',
                    'record_count': 0,
                    'last_date': None,
                    'missing_yesterday': True,
                    'missing_recent': list(range(recent_days))
                }
                results['missing_yesterday'].append((code, name))
                results['missing_recent'].append((code, name))
                continue

            # 获取日期列表
            dates = [d.date() for d in df.index]
            last_date = df.index.max().date()

            # ✅ 检查最近交易日数据（而非简单的昨天）
            has_last_trading_day = last_trading_day in dates

            # ✅ 检查最近N个交易日数据（排除周末和节假日）
            missing_recent_days = []
            check_date = today - timedelta(days=1)
            trading_days_checked = 0
            max_lookback = 15  # 最多回溯15天

            while trading_days_checked < recent_days and (today - check_date).days <= max_lookback:
                # 只检查交易日
                if is_trading_day(check_date):
                    if check_date not in dates:
                        missing_recent_days.append(check_date)
                    trading_days_checked += 1
                check_date -= timedelta(days=1)

            results['details'][code] = {
                'name': name,
                'status': 'ok' if not missing_recent_days else 'missing',
                'record_count': len(df),
                'last_date': last_date,
                'has_yesterday': has_last_trading_day,  # 实际是"最近交易日"
                'last_trading_day': last_trading_day,   # 记录实际检查的交易日
                'missing_recent_days': missing_recent_days,
                'date_range': f"{df.index.min().date()} 到 {last_date}"
            }

            if not has_last_trading_day:
                results['missing_yesterday'].append((code, name))
            
            if missing_recent_days:
                results['missing_recent'].append((code, name))
            
            results['checked_stocks'] += 1
            
        except Exception as e:
            results['details'][code] = {
                'name': name,
                'status': 'error',
                'error': str(e)
            }
    
    return results

def repair_missing_data(codes, names, force_update=False, max_retry=3):
    """
    修复缺失数据，特别是昨天数据
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
        force_update: 是否强制重新下载数据
        max_retry: 最大重试次数
    
    Returns:
        dict: 修复结果
    """
    store = get_market_store()
    repair_results = {
        'total_stocks': len(codes),
        'repaired_stocks': 0,
        'failed_stocks': [],
        'details': {}
    }
    
    # ✅ 批量并行修复数据（性能优化）
    operation = "强制重新初始化" if force_update else "增量更新"

    # 记录修复前的数据状态
    data_before = {}
    for code, name in zip(codes, names):
        try:
            df_before = store.read_stock_data(code, update_if_needed=False)
            data_before[code] = {
                'name': name,
                'record_count': len(df_before) if not df_before.empty else 0,
                'df': df_before
            }
        except Exception as e:
            data_before[code] = {
                'name': name,
                'record_count': 0,
                'df': None,
                'error': str(e)
            }

    # 批量并行修复
    if force_update:
        # 强制重新初始化所有数据
        results = store.batch_init_stock_data(
            codes=codes,
            names=names,
            days=365,
            force_update=True,
            max_workers=5  # 5线程并行，平衡性能和API限流风险
        )
    else:
        # 智能增量更新
        results = store.batch_update_recent_data(
            codes=codes,
            names=names,
            recent_days=30,
            max_workers=5
        )

    # 收集修复结果
    yesterday = (datetime.now() - timedelta(days=1)).date()

    for code, name in zip(codes, names):
        success = results.get(code, False)
        before_info = data_before.get(code, {})

        try:
            if success:
                # 检查修复后的数据
                df_after = store.read_stock_data(code, update_if_needed=False)
                record_count_after = len(df_after) if not df_after.empty else 0
                record_count_before = before_info.get('record_count', 0)
                added_records = record_count_after - record_count_before

                # 检查昨天数据是否已修复
                has_yesterday_after = yesterday in [d.date() for d in df_after.index] if not df_after.empty else False

                repair_results['details'][code] = {
                    'name': name,
                    'status': 'success',
                    'operation': operation,
                    'added_records': added_records,
                    'record_count_before': record_count_before,
                    'record_count_after': record_count_after,
                    'has_yesterday': has_yesterday_after,
                    'last_date': df_after.index.max().date() if not df_after.empty else None
                }

                if added_records > 0:
                    repair_results['repaired_stocks'] += 1
                else:
                    repair_results['details'][code]['status'] = 'no_update'
            else:
                repair_results['details'][code] = {
                    'name': name,
                    'status': 'failed',
                    'operation': operation,
                    'error': '更新操作失败'
                }
                repair_results['failed_stocks'].append((code, name))

        except Exception as e:
            repair_results['details'][code] = {
                'name': name,
                'status': 'error',
                'error': str(e)
            }
            repair_results['failed_stocks'].append((code, name))
    
    return repair_results

def get_data_statistics(codes, names):
    """
    获取数据统计信息
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
    
    Returns:
        DataFrame: 统计信息
    """
    store = get_market_store()
    stats = []
    
    for code, name in zip(codes, names):
        try:
            df = store.read_stock_data(code, update_if_needed=False)
            
            if df.empty:
                stats.append({
                    '代码': code,
                    '名称': name,
                    '记录数': 0,
                    '开始日期': None,
                    '最后日期': None,
                    '数据状态': '空数据'
                })
                continue
            
            last_date = df.index.max().date()
            today = datetime.now().date()
            # ✅ 使用交易日判断：获取最近的交易日，而非简单的昨天
            last_trading_day = get_last_trading_day(today - timedelta(days=1))

            # 检查数据完整性
            dates = [d.date() for d in df.index]
            has_last_trading_day = last_trading_day in dates

            # ✅ 计算缺失的最近交易日（排除周末和节假日）
            missing_recent = []
            check_date = today - timedelta(days=1)
            trading_days_checked = 0
            max_lookback = 15  # 最多回溯15天

            while trading_days_checked < 5 and (today - check_date).days <= max_lookback:
                # 只检查交易日
                if is_trading_day(check_date):
                    if check_date not in dates:
                        missing_recent.append(check_date)
                    trading_days_checked += 1
                check_date -= timedelta(days=1)

            data_status = "完整"
            if not has_last_trading_day:
                data_status = f"缺失最近交易日({last_trading_day})"
            elif missing_recent:
                data_status = f"缺失{len(missing_recent)}个最近交易日"
            
            stats.append({
                '代码': code,
                '名称': name,
                '记录数': len(df),
                '开始日期': df.index.min().date(),
                '最后日期': last_date,
                '数据状态': data_status,
                '距今天数': (today - last_date).days if last_date else None
            })
            
        except Exception as e:
            stats.append({
                '代码': code,
                '名称': name,
                '记录数': 0,
                '开始日期': None,
                '最后日期': None,
                '数据状态': f'错误: {str(e)}'
            })
    
    return pd.DataFrame(stats)

def create_data_integrity_report(codes, names):
    """
    创建完整的数据完整性报告
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
    
    Returns:
        str: HTML格式的报告
    """
    # 检查数据完整性
    integrity_results = check_data_integrity(codes, names)
    
    # 生成报告
    report_lines = [
        "# 数据完整性检查报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"检查股票数: {integrity_results['total_stocks']}",
        f"成功检查: {integrity_results['checked_stocks']}",
        "",
        "## 摘要",
        f"- 缺失昨天数据的股票: {len(integrity_results['missing_yesterday'])}只",
        f"- 缺失最近交易日数据的股票: {len(integrity_results['missing_recent'])}只",
        ""
    ]
    
    if integrity_results['missing_yesterday']:
        report_lines.append("## 缺失昨天数据的股票")
        for code, name in integrity_results['missing_yesterday']:
            detail = integrity_results['details'][code]
            report_lines.append(f"- {name}({code}): 最后日期 {detail.get('last_date', 'N/A')}")
    
    if integrity_results['missing_recent']:
        report_lines.append("## 缺失最近交易日数据的股票")
        for code, name in integrity_results['missing_recent']:
            detail = integrity_results['details'][code]
            missing_days = detail.get('missing_recent_days', [])
            if missing_days:
                days_str = ', '.join([d.strftime('%Y-%m-%d') for d in missing_days])
                report_lines.append(f"- {name}({code}): 缺失 {days_str}")
    
    report_lines.append("## 详细数据统计")
    stats_df = get_data_statistics(codes, names)
    report_lines.append(stats_df.to_string())
    
    return "\n".join(report_lines)

def manual_data_fix(codes, names, target_date=None):
    """
    手动修复特定日期的数据
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
        target_date: 目标日期（datetime.date对象），如果为None则修复昨天数据
    
    Returns:
        dict: 修复结果
    """
    if target_date is None:
        target_date = datetime.now().date() - timedelta(days=1)
    
    from data_fetcher import _fallback_fetch_from_akshare
    import akshare as ak
    
    store = get_market_store()
    results = {
        'target_date': target_date,
        'total_stocks': len(codes),
        'fixed_stocks': 0,
        'failed_stocks': [],
        'details': {}
    }
    
    for code, name in zip(codes, names):
        try:
            # 检查是否已经有目标日期数据
            df = store.read_stock_data(code, update_if_needed=False)
            if not df.empty:
                dates = [d.date() for d in df.index]
                if target_date in dates:
                    results['details'][code] = {
                        'name': name,
                        'status': 'already_exists',
                        'date': target_date
                    }
                    continue
            
            # 获取目标日期数据
            start_date = target_date.strftime("%Y%m%d")
            end_date = target_date.strftime("%Y%m%d")
            
            # 使用AkShare获取单日数据
            df_single = ak.stock_zh_a_hist(
                symbol=code, 
                period="daily", 
                start_date=start_date, 
                end_date=end_date, 
                adjust="qfq"
            )
            
            if not isinstance(df_single, pd.DataFrame) or df_single.empty:
                results['details'][code] = {
                    'name': name,
                    'status': 'no_data_available',
                    'date': target_date,
                    'error': 'API返回空数据（可能是节假日）'
                }
                continue
            
            # 处理数据
            df_single = df_single[['日期', '收盘']].rename(columns={'日期': 'date', '收盘': 'close'})
            df_single['date'] = pd.to_datetime(df_single['date'])
            df_single['close'] = pd.to_numeric(df_single['close'], errors='coerce')
            df_single['code'] = code
            df_single = df_single.set_index('date')
            
            # 合并到现有数据
            if df.empty:
                combined_df = df_single
            else:
                combined_df = pd.concat([df, df_single])
                # 去重并排序
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
            
            # 保存数据
            filepath = store._get_filepath(code)
            combined_df.to_parquet(filepath)
            
            results['details'][code] = {
                'name': name,
                'status': 'fixed',
                'date': target_date,
                'price': float(df_single['close'].iloc[0]) if not df_single.empty else None
            }
            results['fixed_stocks'] += 1
            
        except Exception as e:
            results['details'][code] = {
                'name': name,
                'status': 'error',
                'date': target_date,
                'error': str(e)
            }
            results['failed_stocks'].append((code, name))
    
    return results

# Streamlit界面函数
def show_data_integrity_ui():
    """显示数据完整性检查的Streamlit界面"""
    st.title("🔍 数据完整性检查与修复")
    st.markdown("""
    此工具用于检查和修复历史数据缺失问题，特别是昨天数据缺失的情况。
    
    ### 常见问题原因：
    1. **网络连接问题** - 数据获取API暂时不可用
    2. **节假日停盘** - 股市休市，没有新数据
    3. **系统调度延迟** - 自动更新任务未及时执行
    4. **数据源限制** - 某些股票数据可能有限制
    """)
    
    # 从配置加载股票列表
    from config import load_config
    config = load_config()
    
    # 获取master_pool中的股票
    master_pool = config.get('master_pool', {})
    all_stocks = []
    for category, stocks in master_pool.items():
        for code, name in stocks.items():
            all_stocks.append({'code': code, 'name': name, 'category': category})
    
    if not all_stocks:
        st.warning("配置文件中没有找到股票数据")
        return
    
    # 股票选择
    st.subheader("选择要检查的股票")

    # 快速操作按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        select_all = st.button("✅ 全选", use_container_width=True)
    with col_btn2:
        clear_all = st.button("❌ 清空", use_container_width=True)

    # 初始化multiselect的默认值
    if 'stock_multiselect' not in st.session_state:
        st.session_state.stock_multiselect = [f"{s['code']} - {s['name']} ({s['category']})" for s in all_stocks[:5]]

    # 处理全选和清空按钮 - 直接更新multiselect组件的session_state
    if select_all:
        st.session_state.stock_multiselect = [f"{s['code']} - {s['name']} ({s['category']})" for s in all_stocks]
        st.rerun()

    if clear_all:
        st.session_state.stock_multiselect = []
        st.rerun()

    # 股票多选框
    selected_indices = st.multiselect(
        "选择股票（可多选）",
        options=[f"{s['code']} - {s['name']} ({s['category']})" for s in all_stocks],
        key='stock_multiselect'
    )

    # 显示选中数量
    if selected_indices:
        st.info(f"已选择 {len(selected_indices)} 只股票 / 总共 {len(all_stocks)} 只")
    else:
        st.warning("⚠️ 请选择至少一只股票")
        return
    
    # 解析选择的股票
    selected_stocks = []
    for item in selected_indices:
        parts = item.split(" - ")
        code = parts[0]
        name = parts[1].split(" (")[0]
        selected_stocks.append({'code': code, 'name': name})
    
    codes = [s['code'] for s in selected_stocks]
    names = [s['name'] for s in selected_stocks]
    
    # 操作选项
    st.subheader("检查与修复选项")
    col1, col2 = st.columns(2)
    
    with col1:
        check_button = st.button("🔍 检查数据完整性", type="primary", use_container_width=True)
    
    with col2:
        repair_button = st.button("🛠️ 修复缺失数据", type="secondary", use_container_width=True)
    
    # 强制更新选项
    force_update = st.checkbox("强制重新下载数据（如果常规修复无效）")
    
    if check_button:
        with st.spinner("正在检查数据完整性..."):
            # 获取统计信息
            stats_df = get_data_statistics(codes, names)
            
            # 显示统计表格
            st.subheader("数据统计")
            st.dataframe(stats_df, use_container_width=True)
            
            # 完整性检查
            integrity_results = check_data_integrity(codes, names)
            
            # 显示摘要
            st.subheader("完整性摘要")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("检查股票数", integrity_results['total_stocks'])
            
            with col2:
                st.metric("缺失昨天数据", len(integrity_results['missing_yesterday']))
            
            with col3:
                st.metric("缺失最近交易日", len(integrity_results['missing_recent']))
            
            # 显示详细信息
            if integrity_results['missing_yesterday']:
                st.warning(f"发现 {len(integrity_results['missing_yesterday'])} 只股票缺少昨天数据")
                missing_list = "\n".join([f"- {name}({code})" for code, name in integrity_results['missing_yesterday']])
                st.text(missing_list)
            
            # 生成报告
            report = create_data_integrity_report(codes, names)
            with st.expander("查看完整报告"):
                st.code(report)
    
    if repair_button:
        with st.spinner("正在修复缺失数据..."):
            # 执行修复
            repair_results = repair_missing_data(codes, names, force_update=force_update)
            
            # 显示修复结果
            st.subheader("修复结果")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("处理股票数", repair_results['total_stocks'])
            
            with col2:
                st.metric("修复成功", repair_results['repaired_stocks'])
            
            with col3:
                st.metric("修复失败", len(repair_results['failed_stocks']))
            
            # 显示修复详情
            if repair_results['repaired_stocks'] > 0:
                st.success(f"成功修复 {repair_results['repaired_stocks']} 只股票的数据")
                
                # 显示新增记录数
                added_records = []
                for code, detail in repair_results['details'].items():
                    if detail.get('status') == 'success' and detail.get('added_records', 0) > 0:
                        added_records.append((detail['name'], detail['added_records']))
                
                if added_records:
                    st.write("新增记录数：")
                    for name, count in added_records:
                        st.write(f"- {name}: +{count} 条记录")
            
            if repair_results['failed_stocks']:
                st.error(f"{len(repair_results['failed_stocks'])} 只股票修复失败")
                for code, name in repair_results['failed_stocks']:
                    detail = repair_results['details'][code]
                    st.write(f"- {name}({code}): {detail.get('error', '未知错误')}")
            
            # 建议
            if len(repair_results['failed_stocks']) > 0 and not force_update:
                st.info("💡 提示：部分股票修复失败，可以尝试勾选'强制重新下载数据'选项后重试。")
    
    # 手动修复特定日期
    st.subheader("手动修复特定日期")
    st.markdown("如果自动修复无效，可以手动指定日期进行修复。")
    
    target_date = st.date_input("选择要修复的日期", value=datetime.now().date() - timedelta(days=1))
    
    if st.button("🔧 手动修复指定日期", type="secondary"):
        with st.spinner(f"正在手动修复 {target_date} 的数据..."):
            manual_results = manual_data_fix(codes, names, target_date)
            
            st.subheader("手动修复结果")
            st.write(f"目标日期: {target_date}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("修复成功", manual_results['fixed_stocks'])
            with col2:
                st.metric("修复失败", len(manual_results['failed_stocks']))
            
            # 显示详情
            if manual_results['fixed_stocks'] > 0:
                fixed_details = []
                for code, detail in manual_results['details'].items():
                    if detail.get('status') == 'fixed':
                        fixed_details.append(f"- {detail['name']}({code}): {detail.get('price', 'N/A')}元")
                
                if fixed_details:
                    st.success("修复成功的股票：")
                    st.write("\n".join(fixed_details))
    
    # 预防措施建议
    st.subheader("预防措施")
    st.markdown("""
    ### 如何避免数据缺失问题：
    
    1. **定期检查** - 每天开盘前运行数据完整性检查
    2. **设置自动更新** - 配置定时任务自动更新数据
    3. **监控日志** - 关注数据获取的日志和错误信息
    4. **备用数据源** - 考虑使用多个数据源作为备份
    
    ### 自动更新脚本示例：
    ```python
    # 每天收盘后自动更新数据
    python -c "from market_store import update_recent_data_for_all; update_recent_data_for_all(['000001', '000002'])"
    ```
    """)

if __name__ == "__main__":
    # 命令行接口
    import argparse
    
    parser = argparse.ArgumentParser(description="数据完整性检查与修复工具")
    parser.add_argument("--check", action="store_true", help="检查数据完整性")
    parser.add_argument("--repair", action="store_true", help="修复缺失数据")
    parser.add_argument("--force", action="store_true", help="强制重新下载数据")
    parser.add_argument("--codes", nargs="+", help="股票代码列表")
    parser.add_argument("--names", nargs="+", help="股票名称列表")
    
    args = parser.parse_args()
    
    if args.codes and args.names and len(args.codes) == len(args.names):
        codes = args.codes
        names = args.names
        
        if args.check:
            print("检查数据完整性...")
            results = check_data_integrity(codes, names)
            print(f"检查完成: {results}")
            
        elif args.repair:
            print("修复缺失数据...")
            results = repair_missing_data(codes, names, force_update=args.force)
            print(f"修复完成: {results}")
    else:
        print("请提供股票代码和名称列表，或使用Streamlit界面")
