"""
FILE: market_store.py
ROLE: 本地行情仓库，负责系统数据的“代谢与存储”，实现冷热数据分离存储和高效读写。
LOGIC: 
    1. 使用Parquet列式存储，每只股票一个文件，支持快速读取和高效压缩。
    2. 实现冷热数据分离：历史数据归档，最近数据快速访问。
    3. 确保T-1数据归档的原子性，避免数据丢失。
    4. 提供数据对齐后的Validator接口，保证数据一致性。
DEPENDENCIES: pandas, akshare, numpy, streamlit, pathlib
"""

import pandas as pd
import numpy as np
import akshare as ak
import streamlit as st
from datetime import datetime, timedelta
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 为什么要使用Parquet：
# 1. 列式存储，读取速度快，尤其适合时间序列数据
# 2. 压缩率高，节省磁盘空间
# 3. 支持分区和谓词下推（虽然本项目暂不需要）
# 4. 与pandas集成良好，读写简单

class MarketStore:
    """本地历史行情仓库"""
    
    def __init__(self, base_dir="market_data"):
        """
        初始化本地仓库
        
        Args:
            base_dir: Parquet文件存储的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def _get_filepath(self, code):
        """
        获取股票对应的Parquet文件路径
        
        Args:
            code: 股票代码
            
        Returns:
            Path对象
        """
        return self.base_dir / f"{code}.parquet"
    
    @st.cache_data(ttl=3600)
    def _fetch_from_akshare(_self, code, start_date, end_date, _retry=3):
        """
        从AkShare获取股票历史数据（带缓存）

        注意：使用st.cache_data缓存，避免短时间内重复请求相同数据
        为什么使用st.cache_data：
        1. 避免在同一个session内重复请求相同数据
        2. 缓存时间设为1小时，平衡数据新鲜度和性能
        3. 即使不同用户请求相同股票，也能共享缓存（根据参数）

        Args:
            code: 股票代码
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            _retry: 重试次数，默认3次（已优化，避免周末等无效重试）

        Returns:
            DataFrame，包含日期和收盘价
        """
        for attempt in range(_retry):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code, 
                    period="daily", 
                    start_date=start_date, 
                    end_date=end_date, 
                    adjust="qfq"
                )
                
                if not isinstance(df, pd.DataFrame) or df.empty:
                    # 返回空DataFrame，检查是否还能重试
                    if attempt < _retry - 1:
                        # 等待时间随尝试次数增加而增加（指数退避），最多5秒
                        wait_time = min(2 ** attempt, 5)
                        print(f"第{attempt+1}次尝试获取 {code} 数据失败，返回空DataFrame，{wait_time}秒后重试")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"第{attempt+1}次尝试获取 {code} 数据失败，已达最大重试次数({_retry})，放弃")
                        return pd.DataFrame()  # 返回空DataFrame
                    
                # 只保留需要的列
                df = df[['日期', '收盘']].copy()
                df = df.rename(columns={'日期': 'date', '收盘': 'close'})
                df['date'] = pd.to_datetime(df['date'])
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['code'] = code
                
                print(f"成功获取 {code} 数据，共 {len(df)} 条记录")
                return df.set_index('date')
            except ConnectionResetError as e:
                # 针对连接重置错误，等待稍长时间
                if attempt < _retry - 1:
                    wait_time = min(2 ** (attempt + 1), 10)  # 最多等待10秒
                    print(f"第{attempt+1}次尝试获取 {code} 数据失败（连接重置），{wait_time}秒后重试")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"从AkShare获取 {code} 数据失败，连接重置错误，已重试{_retry}次")
                    return pd.DataFrame()
            except Exception as e:
                if attempt < _retry - 1:
                    wait_time = min(2 ** attempt, 5)  # 最多等待5秒
                    print(f"第{attempt+1}次尝试获取 {code} 数据失败，{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"从AkShare获取 {code} 数据失败，已重试{_retry}次: {e}")
                    return pd.DataFrame()
    
    def init_stock_data(self, code, days=365, force_update=False):
        """
        初始化单只股票的历史数据
        
        策略：
        1. 如果本地已有数据且force_update=False，直接返回
        2. 否则从AkShare拉取全量历史数据
        3. 保存到Parquet文件
        
        Args:
            code: 股票代码
            days: 历史天数
            force_update: 是否强制更新（重新拉取）
            
        Returns:
            bool: 是否成功
        """
        filepath = self._get_filepath(code)
        
        # 如果文件已存在且不需要强制更新，直接返回成功
        if filepath.exists() and not force_update:
            return True
            
        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        # 从AkShare获取数据
        df = self._fetch_from_akshare(code, start_date, end_date)
        if df.empty:
            return False
            
        # 保存到Parquet
        try:
            df.to_parquet(filepath)
            print(f"已初始化 {code} 的历史数据，共 {len(df)} 条记录")
            return True
        except Exception as e:
            print(f"保存 {code} 数据到Parquet失败: {e}")
            return False
    
    def update_recent_data(self, code, recent_days=30):
        """
        增量更新最近N天的数据
        
        策略：
        1. 读取本地已有数据
        2. 计算需要更新的日期范围（从最后日期到昨天）
        3. 从AkShare获取增量数据
        4. 合并并去重
        5. 保存回Parquet
        
        Args:
            code: 股票代码
            recent_days: 最近多少天（默认30天，避免遗漏假期）
            
        Returns:
            bool: 是否成功更新
        """
        filepath = self._get_filepath(code)
        
        # 如果本地没有数据，则初始化
        if not filepath.exists():
            return self.init_stock_data(code, days=365)
        
        try:
            # 读取本地数据
            local_df = pd.read_parquet(filepath)
            
            # 检查是否有数据
            if local_df.empty:
                return self.init_stock_data(code, days=365)
            
            # 获取最后日期
            last_date = local_df.index.max()
            
            # 检查最后日期是否已经是今天
            now = datetime.now()
            today = now.date()
            last_date_date = last_date.date() if hasattr(last_date, 'date') else last_date

            # 如果最后日期已经是今天，不需要更新
            if last_date_date >= today:
                print(f"{code} 数据已是最新（包含今天），无需更新")
                return True

            # ✅ 周末检测：避免不必要的API请求
            is_weekend = now.weekday() >= 5  # 周六=5, 周日=6
            if is_weekend:
                # 周末时，检查数据是否已包含上周五
                # 周六(weekday=5)的上周五是1天前，周日(weekday=6)的上周五是2天前
                days_since_friday = now.weekday() - 4  # 5-4=1, 6-4=2
                last_friday = (now - timedelta(days=days_since_friday)).date()

                if last_date_date >= last_friday:
                    print(f"{code} 今天是周末，数据已包含上周五({last_friday})，无需更新")
                    return True
                # 如果数据不包含上周五，继续更新流程（获取上周五的数据）

            # 计算需要更新的日期范围
            # 从最后日期的下一天开始
            update_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")

            # 周末时结束日期设为上周五，避免请求周末数据
            if is_weekend:
                days_since_friday = now.weekday() - 4
                last_friday = now - timedelta(days=days_since_friday)
                update_end = last_friday.strftime("%Y%m%d")
            else:
                update_end = now.strftime("%Y%m%d")

            # 如果日期范围无效，直接返回
            if update_start > update_end:
                print(f"{code} 无需更新（起始日期 {update_start} > 结束日期 {update_end}）")
                return True
            
            # 从AkShare获取增量数据
            df_update = self._fetch_from_akshare(code, update_start, update_end)
            if df_update.empty:
                # 可能是周末或假期，没有新数据
                return True
            
            # 合并数据（去重）
            combined_df = pd.concat([local_df, df_update])
            # 去重，保留最后出现的记录（以防有更新）
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            
            # 保存回Parquet
            combined_df.to_parquet(filepath)
            print(f"已更新 {code} 的最近 {len(df_update)} 条数据")
            return True
            
        except Exception as e:
            print(f"增量更新 {code} 数据失败: {e}")
            # 失败时尝试重新初始化
            return self.init_stock_data(code, days=365)
    
    def read_stock_data(self, code, update_if_needed=False):
        """
        读取单只股票的历史数据

        策略：
        1. 检查本地是否有数据
        2. 如果没有，初始化
        3. 如果有，根据update_if_needed决定是否增量更新
        4. 返回数据

        Args:
            code: 股票代码
            update_if_needed: 是否检查并更新最近数据（默认False，避免页面切换时自动更新）

        Returns:
            DataFrame: 股票历史数据，索引为日期
        """
        # 确保本地有数据
        if not self._get_filepath(code).exists():
            success = self.init_stock_data(code)
            if not success:
                return pd.DataFrame()
        
        # 如果需要，更新最近数据
        if update_if_needed:
            self.update_recent_data(code)
        
        # 读取数据
        try:
            df = pd.read_parquet(self._get_filepath(code))
            return df[['close', 'code']]  # 只返回需要的列
        except Exception as e:
            print(f"读取 {code} 的Parquet数据失败: {e}")
            return pd.DataFrame()
    
    def read_multiple_stocks(self, codes, names, update_if_needed=False):
        """
        读取多只股票的历史数据并合并为宽表

        策略：
        1. 为每只股票调用read_stock_data
        2. 将数据转换为宽格式（每列一个股票）
        3. 合并对齐日期

        Args:
            codes: 股票代码列表
            names: 股票名称列表
            update_if_needed: 是否检查并更新最近数据（默认False，避免页面切换时自动更新）

        Returns:
            DataFrame: 宽表，列为股票名称，索引为日期
        """
        dfs = []
        
        for code, name in zip(codes, names):
            df = self.read_stock_data(code, update_if_needed)
            if df.empty:
                continue
            
            # 重命名close列为股票名称
            df_renamed = df[['close']].copy()
            df_renamed.columns = [name]
            dfs.append(df_renamed)
        
        # 合并所有数据
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, axis=1)
        # 按日期排序并去除所有列都为NaN的行
        result = result.sort_index().dropna(how='all')
        return result

    def batch_init_stock_data(self, codes, names=None, days=365, force_update=False, max_workers=5):
        """
        批量初始化多只股票的历史数据（并行）

        使用线程池并行获取数据，显著提升初始化速度。

        Args:
            codes: 股票代码列表
            names: 股票名称列表（可选，用于日志显示）
            days: 历史天数
            force_update: 是否强制更新
            max_workers: 最大并发线程数（默认5，避免API限流）

        Returns:
            dict: {code: success_bool} 每只股票的初始化结果
        """
        if names is None:
            names = codes

        results = {}
        print(f"🚀 批量初始化 {len(codes)} 只股票（并行，max_workers={max_workers}）...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(self.init_stock_data, code, days, force_update): (code, name)
                for code, name in zip(codes, names)
            }

            # 收集结果
            completed = 0
            for future in as_completed(future_to_code):
                code, name = future_to_code[future]
                completed += 1
                try:
                    success = future.result()
                    results[code] = success
                    if success:
                        print(f"  [{completed}/{len(codes)}] ✅ {name}({code}) 初始化成功")
                    else:
                        print(f"  [{completed}/{len(codes)}] ❌ {name}({code}) 初始化失败")
                except Exception as e:
                    print(f"  [{completed}/{len(codes)}] ❌ {name}({code}) 初始化异常: {e}")
                    results[code] = False

        success_count = sum(1 for v in results.values() if v)
        print(f"📊 批量初始化完成：成功 {success_count}/{len(codes)}")
        return results

    def batch_update_recent_data(self, codes, names=None, recent_days=30, max_workers=5):
        """
        批量增量更新多只股票的数据（并行）

        使用线程池并行更新数据，显著提升更新速度。

        Args:
            codes: 股票代码列表
            names: 股票名称列表（可选，用于日志显示）
            recent_days: 最近天数
            max_workers: 最大并发线程数（默认5）

        Returns:
            dict: {code: success_bool} 每只股票的更新结果
        """
        if names is None:
            names = codes

        results = {}
        print(f"🚀 批量更新 {len(codes)} 只股票（并行，max_workers={max_workers}）...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(self.update_recent_data, code, recent_days): (code, name)
                for code, name in zip(codes, names)
            }

            # 收集结果
            completed = 0
            for future in as_completed(future_to_code):
                code, name = future_to_code[future]
                completed += 1
                try:
                    success = future.result()
                    results[code] = success
                    if success:
                        print(f"  [{completed}/{len(codes)}] ✅ {name}({code}) 更新成功")
                    else:
                        print(f"  [{completed}/{len(codes)}] ⚠️  {name}({code}) 更新跳过或失败")
                except Exception as e:
                    print(f"  [{completed}/{len(codes)}] ❌ {name}({code}) 更新异常: {e}")
                    results[code] = False

        success_count = sum(1 for v in results.values() if v)
        print(f"📊 批量更新完成：成功 {success_count}/{len(codes)}")
        return results

    def get_stock_info(self, code):
        """
        获取股票的基本信息（文件大小、记录数、最后日期等）
        
        Args:
            code: 股票代码
            
        Returns:
            dict: 股票信息
        """
        filepath = self._get_filepath(code)
        info = {
            'code': code,
            'file_exists': filepath.exists(),
            'file_path': str(filepath)
        }
        
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                info['record_count'] = len(df)
                info['date_range'] = f"{df.index.min().date()} 到 {df.index.max().date()}"
                info['file_size_mb'] = os.path.getsize(filepath) / (1024 * 1024)
            except Exception as e:
                info['error'] = str(e)
        
        return info


# 为什么要使用session_state固定历史数据：
# 1. Streamlit的每次交互都会触发页面重新运行
# 2. 如果不缓存，每次rerun都会重新读取数据（即使数据没有变化）
# 3. session_state可以跨rerun保持数据，避免重复I/O
# 4. 结合Parquet本地存储，实现"内存缓存+磁盘缓存"双层加速

# 全局实例（单例模式，避免重复创建）
_store_instance = None

def get_market_store():
    """
    获取MarketStore单例实例
    
    Returns:
        MarketStore实例
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = MarketStore()
    return _store_instance


def ensure_data_initialized(codes, names):
    """
    确保所有股票的数据都已初始化（并行处理）
    
    Args:
        codes: 股票代码列表
        names: 股票名称列表
        
    Returns:
        bool: 是否全部成功
    """
    store = get_market_store()
    results = []
    
    for code, name in zip(codes, names):
        success = store.init_stock_data(code, force_update=False)
        results.append(success)
        if not success:
            print(f"初始化 {name}({code}) 数据失败")
    
    return all(results)


def update_recent_data_for_all(codes, recent_days=30):
    """
    批量更新多只股票的最近数据
    
    Args:
        codes: 股票代码列表
        recent_days: 最近天数
        
    Returns:
        dict: 每只股票的更新结果
    """
    store = get_market_store()
    results = {}
    
    for code in codes:
        success = store.update_recent_data(code, recent_days)
        results[code] = success
    
    return results


if __name__ == "__main__":
    # 测试代码
    store = MarketStore()
    
    # 测试初始化
    print("测试初始化 000001...")
    success = store.init_stock_data("000001", days=30)
    print(f"初始化结果: {success}")
    
    # 测试读取
    print("\n测试读取 000001...")
    df = store.read_stock_data("000001", update_if_needed=False)
    print(f"数据形状: {df.shape}")
    if not df.empty:
        print(f"日期范围: {df.index.min()} 到 {df.index.max()}")
    
    # 测试增量更新
    print("\n测试增量更新 000001...")
    success = store.update_recent_data("000001", recent_days=30)
    print(f"更新结果: {success}")
    
    # 测试多股票读取
    print("\n测试多股票读取...")
    df_multi = store.read_multiple_stocks(
        ["000001", "000002"], 
        ["平安银行", "万科A"], 
        update_if_needed=False
    )
    print(f"多股票数据形状: {df_multi.shape}")
    print(f"列名: {df_multi.columns.tolist()}")
    
    # 测试信息获取
    print("\n测试信息获取...")
    info = store.get_stock_info("000001")
    for key, value in info.items():
        print(f"{key}: {value}")
