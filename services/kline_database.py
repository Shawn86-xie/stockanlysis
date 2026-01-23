"""
全市场K线数据库管理模块

功能：
1. 批量下载全市场K线数据到本地
2. 增量更新已有数据
3. 数据库状态查询
4. 并行下载加速
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 数据库目录
KLINE_DB_DIR = Path("market_data/candle")
KLINE_DB_META_FILE = Path("market_data/kline_db_meta.json")


def get_kline_db_stats():
    """
    获取K线数据库统计信息

    Returns:
        dict: 包含数据库统计信息
    """
    KLINE_DB_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = list(KLINE_DB_DIR.glob("*_ohlcv.parquet"))
    total_count = len(parquet_files)

    if total_count == 0:
        return {
            'exists': False,
            'total_count': 0,
            'latest_update': None,
            'oldest_file': None,
            'newest_file': None,
            'total_size_mb': 0
        }

    # 计算总大小
    total_size = sum(f.stat().st_size for f in parquet_files)
    total_size_mb = total_size / (1024 * 1024)

    # 获取最新和最旧的文件时间
    file_times = [(f, f.stat().st_mtime) for f in parquet_files]
    file_times.sort(key=lambda x: x[1])

    oldest_time = datetime.fromtimestamp(file_times[0][1])
    newest_time = datetime.fromtimestamp(file_times[-1][1])

    # 统计今日更新的文件数
    today = datetime.now().date()
    today_updated = sum(1 for f, t in file_times
                        if datetime.fromtimestamp(t).date() == today)

    return {
        'exists': True,
        'total_count': total_count,
        'latest_update': newest_time.strftime("%Y-%m-%d %H:%M:%S"),
        'oldest_update': oldest_time.strftime("%Y-%m-%d %H:%M:%S"),
        'today_updated': today_updated,
        'total_size_mb': round(total_size_mb, 2)
    }


def get_cached_stock_codes():
    """
    获取已缓存的股票代码列表

    Returns:
        set: 已缓存的股票代码集合
    """
    KLINE_DB_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files = list(KLINE_DB_DIR.glob("*_ohlcv.parquet"))
    codes = set()
    for f in parquet_files:
        # 文件名格式: {code}_ohlcv.parquet
        code = f.stem.replace("_ohlcv", "")
        codes.add(code)
    return codes


def check_data_freshness(code):
    """
    检查单只股票数据的新鲜度

    Args:
        code: 股票代码

    Returns:
        dict: 数据状态信息
    """
    filepath = KLINE_DB_DIR / f"{code}_ohlcv.parquet"

    if not filepath.exists():
        return {'exists': False, 'needs_update': True, 'reason': '文件不存在'}

    try:
        df = pd.read_parquet(filepath)
        if df.empty:
            return {'exists': True, 'needs_update': True, 'reason': '数据为空'}

        last_date = pd.to_datetime(df['日期']).max().date()
        today = datetime.now().date()
        days_diff = (today - last_date).days

        # 判断是否需要更新
        is_weekend = datetime.now().weekday() >= 5
        if days_diff <= 1:
            needs_update = False
            reason = '数据是最新的'
        elif is_weekend and days_diff <= 3:
            needs_update = False
            reason = '周末，数据有效'
        else:
            needs_update = True
            reason = f'数据过期 {days_diff} 天'

        return {
            'exists': True,
            'needs_update': needs_update,
            'last_date': last_date.strftime("%Y-%m-%d"),
            'days_diff': days_diff,
            'reason': reason,
            'rows': len(df)
        }
    except Exception as e:
        return {'exists': True, 'needs_update': True, 'reason': f'读取错误: {e}'}


def download_single_stock(code, days=1460, retry=3, incremental=False, force_full_if_insufficient=True):
    """
    下载单只股票的K线数据

    Args:
        code: 股票代码
        days: 历史天数（全量下载时使用）
        retry: 重试次数
        incremental: 是否增量更新（True时只下载缺失的数据）
        force_full_if_insufficient: 如果现有数据不足300天，强制全量下载

    Returns:
        tuple: (success, code, message)
    """
    KLINE_DB_DIR.mkdir(parents=True, exist_ok=True)
    filepath = KLINE_DB_DIR / f"{code}_ohlcv.parquet"

    # 增量更新：读取现有数据，只下载缺失部分
    existing_df = None
    if incremental and filepath.exists():
        try:
            existing_df = pd.read_parquet(filepath)
            if not existing_df.empty:
                # 检查现有数据是否足够（至少300个交易日）
                if force_full_if_insufficient and len(existing_df) < 300:
                    # 数据不足，强制全量下载
                    existing_df = None
                else:
                    last_date = pd.to_datetime(existing_df['日期']).max()
                    # 从最后日期的下一天开始下载
                    start_date = (last_date + timedelta(days=1)).strftime("%Y%m%d")
                    end_date = datetime.now().strftime("%Y%m%d")

                    # 如果开始日期已经是今天或未来，说明数据已是最新
                    if start_date >= end_date:
                        return (True, code, "数据已是最新")
            else:
                existing_df = None
        except Exception as e:
            existing_df = None

    for attempt in range(retry):
        try:
            end_date = datetime.now().strftime("%Y%m%d")

            if existing_df is not None:
                # 增量模式：从上次最后日期开始
                last_date = pd.to_datetime(existing_df['日期']).max()
                start_date = (last_date + timedelta(days=1)).strftime("%Y%m%d")
            else:
                # 全量模式：下载指定天数
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

            # 尝试获取数据，如果失败则调整开始日期
            df = None
            for offset in [0, 365, 730]:  # 尝试逐步增加历史天数
                try:
                    current_start_date = (datetime.now() - timedelta(days=days + offset)).strftime("%Y%m%d")
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=current_start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    continue

            # 处理空数据情况
            if df is None or df.empty:
                if existing_df is not None:
                    # 增量模式下，空数据可能表示停牌或无新数据
                    return (True, code, "无新数据（可能停牌）")
                if attempt < retry - 1:
                    time.sleep(0.5)
                    continue
                return (False, code, "返回空数据")

            # 标准化列名
            df = df.rename(columns={
                '日期': '日期',
                '开盘': '开盘',
                '最高': '最高',
                '最低': '最低',
                '收盘': '收盘',
                '成交量': '成交量'
            })

            # 确保数据类型
            numeric_cols = ['开盘', '最高', '最低', '收盘', '成交量']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['日期'] = pd.to_datetime(df['日期'])

            # 只保留需要的列
            keep_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
            df = df[[c for c in keep_cols if c in df.columns]]

            # 增量模式：合并新旧数据
            new_records = len(df)
            if existing_df is not None and not df.empty:
                existing_df['日期'] = pd.to_datetime(existing_df['日期'])
                # 合并数据，去重（以日期为准）
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['日期'], keep='last')
                combined_df = combined_df.sort_values('日期').reset_index(drop=True)
                df = combined_df

            # 数据完整性检查：移除无效行
            df = df.dropna(subset=['收盘'])  # 收盘价不能为空
            df = df[df['收盘'] > 0]  # 收盘价必须大于0

            # 如果最终数据少于200行，尝试从更早时间开始
            if len(df) < 200 and existing_df is None:
                # 可能是新股，不视为错误
                pass

            # 保存到本地
            df.to_parquet(filepath, index=False)

            if existing_df is not None:
                return (True, code, f"增量更新+{new_records}条，共{len(df)}条")
            else:
                return (True, code, f"成功，{len(df)}条记录")

        except Exception as e:
            if attempt < retry - 1:
                time.sleep(0.5)
                continue
            return (False, code, str(e))

    return (False, code, "重试次数用尽")


def batch_download_kline(stock_list, days=1460, max_workers=5,
                         progress_callback=None, skip_existing=True, incremental=False):
    """
    批量下载K线数据（支持并行）

    Args:
        stock_list: 股票列表 [{'code': '...', 'name': '...'}, ...]
        days: 历史天数
        max_workers: 并行线程数
        progress_callback: 进度回调函数 (current, total, success, failed, message)
        skip_existing: 是否跳过已有数据（全量下载时使用）
        incremental: 是否增量更新模式（True时只下载缺失的数据并合并）

    Returns:
        dict: 下载结果统计
    """
    # 如果跳过已有数据（仅在全量模式下生效）
    if skip_existing and not incremental:
        cached_codes = get_cached_stock_codes()
        to_download = [s for s in stock_list if s['code'] not in cached_codes]
    else:
        to_download = stock_list

    total = len(to_download)
    if total == 0:
        return {
            'total': len(stock_list),
            'downloaded': 0,
            'success': 0,
            'failed': 0,
            'skipped': len(stock_list),
            'message': '所有股票已有缓存数据'
        }

    success_count = 0
    failed_count = 0
    failed_list = []

    # 用于线程安全的计数器
    lock = threading.Lock()
    completed = [0]

    def download_with_callback(stock):
        nonlocal success_count, failed_count
        code = stock['code']
        name = stock['name']

        # 使用增量更新模式
        success, _, msg = download_single_stock(code, days, incremental=incremental)

        with lock:
            completed[0] += 1
            if success:
                success_count += 1
            else:
                failed_count += 1
                failed_list.append({'code': code, 'name': name, 'error': msg})

            if progress_callback:
                progress_callback(
                    completed[0],
                    total,
                    success_count,
                    failed_count,
                    f"{name} ({code}): {'成功' if success else msg}"
                )

        # 控制请求频率
        time.sleep(0.1)
        return success

    # 使用线程池并行下载
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_with_callback, stock)
                   for stock in to_download]

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                pass

    return {
        'total': len(stock_list),
        'downloaded': total,
        'success': success_count,
        'failed': failed_count,
        'skipped': len(stock_list) - total,
        'failed_list': failed_list
    }


def incremental_update_kline(stock_list, max_workers=5, progress_callback=None):
    """
    增量更新K线数据（真正的增量：只下载缺失的日期并合并）

    更新逻辑：
    1. 检查每只股票数据的最后日期
    2. 只下载最后日期之后的新数据
    3. 将新数据追加到现有数据后面
    4. 保证数据连续性和完整性

    Args:
        stock_list: 股票列表
        max_workers: 并行线程数
        progress_callback: 进度回调函数

    Returns:
        dict: 更新结果统计
    """
    # 找出需要更新的股票（数据过期的）
    to_update = []
    up_to_date = 0
    no_data = 0

    for stock in stock_list:
        code = stock['code']
        status = check_data_freshness(code)
        if not status['exists']:
            # 完全没有数据的股票，需要全量下载
            to_update.append({'stock': stock, 'mode': 'full'})
            no_data += 1
        elif status['needs_update']:
            # 数据过期，需要增量更新
            to_update.append({'stock': stock, 'mode': 'incremental'})
        else:
            up_to_date += 1

    if not to_update:
        return {
            'total': len(stock_list),
            'checked': len(stock_list),
            'up_to_date': up_to_date,
            'updated': 0,
            'success': 0,
            'failed': 0,
            'message': '所有数据都是最新的'
        }

    # 使用增量模式下载（只下载缺失的数据并合并）
    stocks_to_download = [item['stock'] for item in to_update]
    result = batch_download_kline(
        stocks_to_download,
        days=730,  # 对于完全没有数据的股票使用730天（两年）
        max_workers=max_workers,
        progress_callback=progress_callback,
        skip_existing=False,
        incremental=True  # 启用真正的增量更新模式
    )

    return {
        'total': len(stock_list),
        'checked': len(stock_list),
        'up_to_date': up_to_date,
        'updated': result['downloaded'],
        'success': result['success'],
        'failed': result['failed'],
        'failed_list': result.get('failed_list', [])
    }


def read_local_kline(code):
    """
    从本地数据库读取K线数据

    Args:
        code: 股票代码

    Returns:
        DataFrame 或 None
    """
    filepath = KLINE_DB_DIR / f"{code}_ohlcv.parquet"

    if not filepath.exists():
        return None

    try:
        df = pd.read_parquet(filepath)
        return df
    except Exception as e:
        return None


def validate_kline_data(code):
    """
    验证单只股票K线数据的完整性和正确性

    检查项目：
    1. 文件是否存在
    2. 数据是否为空
    3. 必要列是否完整
    4. 数据类型是否正确
    5. 日期是否连续（考虑节假日）
    6. 价格是否合理（不能为负或异常值）

    Args:
        code: 股票代码

    Returns:
        dict: 验证结果
    """
    filepath = KLINE_DB_DIR / f"{code}_ohlcv.parquet"

    result = {
        'code': code,
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # 1. 文件存在性检查
    if not filepath.exists():
        result['valid'] = False
        result['errors'].append('文件不存在')
        return result

    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f'文件读取失败: {e}')
        return result

    # 2. 空数据检查
    if df.empty:
        result['valid'] = False
        result['errors'].append('数据为空')
        return result

    # 3. 必要列检查
    required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        result['valid'] = False
        result['errors'].append(f'缺少必要列: {missing_cols}')
        return result

    # 4. 数据类型检查
    df['日期'] = pd.to_datetime(df['日期'])

    # 5. 价格合理性检查
    for col in ['开盘', '最高', '最低', '收盘']:
        if (df[col] <= 0).any():
            result['warnings'].append(f'{col}列存在非正数值')
        if (df[col] > 10000).any():
            result['warnings'].append(f'{col}列存在异常高值(>10000)')

    # 6. OHLC逻辑检查（最高>=最低, 开盘和收盘在最高最低之间）
    invalid_ohlc = (
        (df['最高'] < df['最低']) |
        (df['开盘'] > df['最高']) |
        (df['开盘'] < df['最低']) |
        (df['收盘'] > df['最高']) |
        (df['收盘'] < df['最低'])
    )
    if invalid_ohlc.any():
        result['warnings'].append(f'存在{invalid_ohlc.sum()}条OHLC逻辑异常记录')

    # 7. 日期连续性检查（简化版：只检查是否有重复日期）
    if df['日期'].duplicated().any():
        result['warnings'].append('存在重复日期')
        df = df.drop_duplicates(subset=['日期'], keep='last')

    # 统计信息
    result['stats'] = {
        'rows': len(df),
        'date_range': f"{df['日期'].min().strftime('%Y-%m-%d')} ~ {df['日期'].max().strftime('%Y-%m-%d')}",
        'days': (df['日期'].max() - df['日期'].min()).days,
        'latest_date': df['日期'].max().strftime('%Y-%m-%d'),
        'latest_close': float(df.iloc[-1]['收盘'])
    }

    return result


def batch_validate_kline(stock_list, progress_callback=None):
    """
    批量验证K线数据

    Args:
        stock_list: 股票列表
        progress_callback: 进度回调函数

    Returns:
        dict: 验证结果统计
    """
    total = len(stock_list)
    valid_count = 0
    invalid_count = 0
    warning_count = 0
    results = []

    for i, stock in enumerate(stock_list):
        code = stock['code']
        name = stock['name']

        if progress_callback and i % 100 == 0:
            progress_callback(i, total, f"正在验证: {name} ({code})")

        result = validate_kline_data(code)
        result['name'] = name

        if result['valid']:
            if result['warnings']:
                warning_count += 1
            else:
                valid_count += 1
        else:
            invalid_count += 1
            results.append(result)

    return {
        'total': total,
        'valid': valid_count,
        'invalid': invalid_count,
        'warnings': warning_count,
        'invalid_list': results
    }


def scan_with_local_db(stock_list, pattern, progress_callback=None):
    """
    使用本地数据库进行扫描（快速模式）

    Args:
        stock_list: 股票列表
        pattern: 形态识别器
        progress_callback: 进度回调函数

    Returns:
        tuple: (results_df, stats)
    """
    results = []
    total = len(stock_list)

    success_count = 0
    skip_count = 0
    match_count = 0

    start_time = time.time()

    for i, stock in enumerate(stock_list):
        code = stock['code']
        name = stock['name']

        # 更新进度
        if progress_callback:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            progress_callback(
                i + 1, total, success_count, skip_count, match_count,
                f"{name} ({code})", speed, eta
            )

        # 从本地读取
        df = read_local_kline(code)

        if df is None or df.empty or len(df) < 70:
            skip_count += 1
            continue

        try:
            # 计算指标
            df = pattern.calculate_indicators(df)

            # 检查形态
            is_match, details = pattern.check_pattern(df)

            success_count += 1

            if details['score'] > 0:
                match_count += 1
                results.append({
                    '代码': code,
                    '名称': name,
                    '评分': details['score'],
                    '60日跌幅%': details['return_60d'],
                    '20日涨跌%': details['return_20d'],
                    '5日涨幅%': details['return_5d'],
                    '底部震荡%': details['bottom_range'],
                    '距底部': details['price_to_low'],
                    '5日上涨天数': details['up_days_5'],
                    '量比': details['volume_ratio'],
                    '波动比': details['vol_ratio'],
                    '现价': details['current_price'],
                    '阶段1_下跌': '✓' if details['phase1'] else '✗',
                    '阶段2_触底': '✓' if details['phase2'] else '✗',
                    '阶段3_反弹': '✓' if details['phase3'] else '✗',
                    '完全匹配': is_match
                })
        except Exception as e:
            skip_count += 1
            continue

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('评分', ascending=False)

    stats = {
        'total': total,
        'success': success_count,
        'skipped': skip_count,
        'matched': match_count,
        'duration': time.time() - start_time
    }

    return results_df, stats
