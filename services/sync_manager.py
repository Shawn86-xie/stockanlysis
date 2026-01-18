"""
同步管理器：负责全市场数据同步、自动补缺和收盘后更新

核心功能：
1. 自动补缺逻辑：检测 market_data 中每只标的的 last_date
2. 收盘后更新：若当前时间 > 15:30，自动抓取今日日线并增量追加到 Parquet 文件中
3. 握手校验：追加前执行 validate_with_prev_close() 确保复权因子未发生突变
4. 全市场同步：一键更新所有 Master Pool 标的
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import streamlit as st

# 导入现有模块
from market_store import MarketStore, get_market_store
from config import load_config


class SyncManagerError(Exception):
    """同步管理器异常"""
    pass


class SyncManager:
    """数据同步管理器"""
    
    def __init__(self, base_dir: str = "market_data"):
        """
        初始化同步管理器
        
        Args:
            base_dir: Parquet文件存储的基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.store = get_market_store()
        
    def get_stock_last_date(self, code: str) -> Optional[datetime]:
        """
        获取单只标的的最后日期
        
        Args:
            code: 股票代码
            
        Returns:
            最后日期，如果文件不存在或为空则返回None
        """
        filepath = self.store._get_filepath(code)
        if not filepath.exists():
            return None
            
        try:
            df = pd.read_parquet(filepath)
            if df.empty:
                return None
            return df.index.max().to_pydatetime()
        except Exception as e:
            print(f"读取 {code} 的最后日期失败: {e}")
            return None
    
    def get_all_stocks_last_dates(self) -> Dict[str, Optional[datetime]]:
        """
        获取所有标的的最后日期
        
        Returns:
            字典：{股票代码: 最后日期}
        """
        results = {}
        
        # 遍历 market_data 目录下的所有 Parquet 文件
        for filepath in self.base_dir.glob("*.parquet"):
            code = filepath.stem  # 去除 .parquet 后缀
            last_date = self.get_stock_last_date(code)
            results[code] = last_date
            
        return results
    
    def validate_with_prev_close(self, code: str, new_data: pd.DataFrame) -> bool:
        """
        握手校验：确保复权因子未发生突变
        
        逻辑：检查新数据的第一条记录的前一天收盘价是否与本地最后一条记录
        的收盘价在合理误差范围内（误差 < 2%）
        
        Args:
            code: 股票代码
            new_data: 新获取的数据（DataFrame，索引为日期，包含close列）
            
        Returns:
            bool: 校验是否通过
        """
        if new_data.empty:
            return False
            
        # 读取本地数据
        filepath = self.store._get_filepath(code)
        if not filepath.exists():
            # 如果没有本地数据，直接通过校验
            return True
            
        try:
            local_df = pd.read_parquet(filepath)
            if local_df.empty:
                return True
                
            # 获取本地最后一条记录的收盘价
            last_local_close = local_df['close'].iloc[-1]
            
            # 获取新数据的第一条记录的收盘价
            first_new_close = new_data['close'].iloc[0]
            
            # 计算相对误差
            if last_local_close == 0:
                return False
                
            error_rate = abs(first_new_close - last_local_close) / last_local_close
            
            # 允许2%的误差（考虑复权因子调整、数据精度等）
            if error_rate > 0.02:
                print(f"警告：{code} 复权因子可能突变，误差率 {error_rate:.2%}")
                print(f"本地最后收盘价: {last_local_close:.4f}, 新数据第一收盘价: {first_new_close:.4f}")
                return False
                
            return True
            
        except Exception as e:
            print(f"握手校验失败 {code}: {e}")
            return False
    
    def sync_single_stock(self, code: str, name: str, force_update: bool = False) -> Dict[str, any]:
        """
        同步单只标的
        
        Args:
            code: 股票代码
            name: 股票名称
            force_update: 是否强制更新
            
        Returns:
            同步结果字典
        """
        result = {
            'code': code,
            'name': name,
            'success': False,
            'message': '',
            'new_records': 0,
            'last_date': None
        }
        
        try:
            # 获取最后日期
            last_date = self.get_stock_last_date(code)
            result['last_date'] = last_date
            
            # 计算是否需要更新
            today = datetime.now().date()
            needs_update = False
            
            if last_date is None:
                # 没有数据，需要初始化
                needs_update = True
                update_type = "初始化"
            elif force_update:
                # 强制更新
                needs_update = True
                update_type = "强制更新"
            else:
                # 检查最后日期是否是今天或昨天
                last_date_date = last_date.date()
                days_diff = (today - last_date_date).days

                # 检查今天是否为交易日（工作日且在交易时间）
                now = datetime.now()
                is_weekend = now.weekday() >= 5  # 周六日
                is_trading_day = not is_weekend  # 简化判断，不考虑节假日

                if days_diff > 1:
                    # 超过1天，需要更新
                    needs_update = True
                    update_type = f"补缺 {days_diff} 天"
                elif days_diff == 1 and is_trading_day:
                    # 差1天且今天是交易日，需要更新
                    needs_update = True
                    update_type = "更新今日数据"
                else:
                    # 已是最新（今天的数据或昨天且今天非交易日）
                    result['success'] = True
                    if is_weekend:
                        result['message'] = f"数据已是最新，最后日期: {last_date.strftime('%Y-%m-%d')} (今天是周末，无需更新)"
                    else:
                        result['message'] = f"数据已是最新，最后日期: {last_date.strftime('%Y-%m-%d')}"
                    return result
            
            if not needs_update:
                return result
            
            # 计算需要更新的日期范围
            # 如果今天是周末，使用上个交易日作为结束日期
            now = datetime.now()
            is_weekend = now.weekday() >= 5

            if is_weekend:
                # 周末：回退到上周五
                # 周六(weekday=5)减1天，周日(weekday=6)减2天
                days_to_subtract = now.weekday() - 4  # 5-4=1, 6-4=2
                last_trading_day = now - timedelta(days=days_to_subtract)
                effective_end_date = last_trading_day.strftime("%Y%m%d")
            else:
                # 工作日：使用今天
                effective_end_date = now.strftime("%Y%m%d")

            if last_date is None:
                # 初始化：获取最近365天数据
                end_date = effective_end_date
                start_date = (now - timedelta(days=365)).strftime("%Y%m%d")
            else:
                # 增量更新：从最后日期的下一天开始
                update_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
                update_end = effective_end_date

                # 如果起始日期大于结束日期，不需要更新
                if update_start > update_end:
                    result['success'] = True
                    result['message'] = f"无需更新（起始日期 {update_start} > 结束日期 {update_end}）"
                    return result

                start_date = update_start
                end_date = update_end
            
            print(f"开始 {update_type} {code}({name}): {start_date} 到 {end_date}")
            
            # 从 AkShare 获取数据
            max_retries = 3
            df_new = pd.DataFrame()

            for attempt in range(max_retries):
                try:
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )

                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # 只保留需要的列
                        df = df[['日期', '收盘']].copy()
                        df = df.rename(columns={'日期': 'date', '收盘': 'close'})
                        df['date'] = pd.to_datetime(df['date'])
                        df['close'] = pd.to_numeric(df['close'], errors='coerce')
                        df['code'] = code
                        df_new = df.set_index('date')
                        break
                    else:
                        # 返回空DataFrame，检查是否还能重试
                        if attempt < max_retries - 1:
                            wait_time = min(2 ** attempt, 5)  # 最多等待5秒，避免过长
                            print(f"第{attempt+1}次尝试获取 {code} 数据失败，返回空DataFrame，{wait_time}秒后重试")
                            time.sleep(wait_time)
                        else:
                            print(f"第{attempt+1}次尝试获取 {code} 数据失败，已达最大重试次数，放弃")
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 5)  # 最多等待5秒
                        print(f"第{attempt+1}次尝试获取 {code} 数据失败: {e}，{wait_time}秒后重试")
                        time.sleep(wait_time)
                    else:
                        print(f"第{attempt+1}次尝试获取 {code} 数据失败: {e}，已达最大重试次数")
                        raise
            
            if df_new.empty:
                result['message'] = f"获取数据失败，返回空DataFrame"
                return result
            
            # 执行握手校验
            if not self.validate_with_prev_close(code, df_new):
                result['message'] = f"握手校验失败，复权因子可能发生突变"
                return result
            
            # 读取本地现有数据（如果有）
            filepath = self.store._get_filepath(code)
            if filepath.exists():
                local_df = pd.read_parquet(filepath)
                if not local_df.empty:
                    # 合并数据（去重）
                    combined_df = pd.concat([local_df, df_new])
                    # 去重，保留最后出现的记录
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df = combined_df.sort_index()
                else:
                    combined_df = df_new
            else:
                combined_df = df_new
            
            # 保存到 Parquet
            combined_df.to_parquet(filepath)
            
            result['success'] = True
            result['new_records'] = len(df_new)
            result['message'] = f"成功{update_type}，新增 {len(df_new)} 条记录"
            
            print(f"完成 {code}({name}) 同步: {result['message']}")
            
        except Exception as e:
            result['message'] = f"同步失败: {str(e)}"
            print(f"同步 {code}({name}) 失败: {e}")
        
        return result
    
    def sync_after_market(self) -> List[Dict[str, any]]:
        """
        收盘后更新：若当前时间 > 15:30，自动抓取今日日线并增量追加
        
        返回所有标的的同步结果
        """
        # 检查当前时间
        current_time = datetime.now().time()
        market_close_time = datetime.strptime("15:30", "%H:%M").time()
        
        if current_time <= market_close_time:
            print(f"当前时间 {current_time.strftime('%H:%M')} 未超过 15:30，不执行收盘后更新")
            return []
        
        print(f"执行收盘后更新（当前时间: {current_time.strftime('%H:%M')})")
        
        # 获取所有 Master Pool 标的
        config = load_config()
        master_pool = config.get('master_pool', {})
        
        all_codes_names = []
        for category, stocks in master_pool.items():
            for code, name in stocks.items():
                all_codes_names.append((code, name))
        
        if not all_codes_names:
            print("Master Pool 为空，无法执行同步")
            return []
        
        # 同步所有标的
        results = []
        for code, name in all_codes_names:
            result = self.sync_single_stock(code, name, force_update=False)
            results.append(result)
            
            # 避免请求过于频繁
            time.sleep(0.5)
        
        # 统计结果
        success_count = sum(1 for r in results if r['success'])
        total_records = sum(r['new_records'] for r in results)
        
        print(f"收盘后更新完成: {success_count}/{len(results)} 成功，新增 {total_records} 条记录")
        
        return results
    
    def sync_all_master_pool(self, force_update: bool = False) -> Dict[str, any]:
        """
        同步所有 Master Pool 标的
        
        Args:
            force_update: 是否强制更新
            
        Returns:
            同步结果汇总
        """
        print(f"开始全市场同步（强制更新: {force_update}）")
        
        # 获取所有 Master Pool 标的
        config = load_config()
        master_pool = config.get('master_pool', {})
        
        if not master_pool:
            return {
                'success': False,
                'message': 'Master Pool 为空，请检查配置',
                'total': 0,
                'success_count': 0,
                'failed_count': 0,
                'total_records': 0,
                'details': []
            }
        
        all_codes_names = []
        for category, stocks in master_pool.items():
            for code, name in stocks.items():
                all_codes_names.append((code, name))
        
        print(f"找到 {len(all_codes_names)} 只标的需要同步")
        
        # 同步所有标的
        results = []
        for i, (code, name) in enumerate(all_codes_names, 1):
            print(f"同步进度: {i}/{len(all_codes_names)} - {code}({name})")
            
            result = self.sync_single_stock(code, name, force_update=force_update)
            results.append(result)
            
            # 避免请求过于频繁
            if i < len(all_codes_names):
                time.sleep(1)
        
        # 汇总结果
        success_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - success_count
        total_records = sum(r['new_records'] for r in results)
        
        # 生成详细报告
        failed_stocks = [(r['code'], r['name'], r['message']) 
                        for r in results if not r['success']]
        
        summary = {
            'success': success_count > 0 or total_records > 0,
            'message': f"同步完成: {success_count} 成功, {failed_count} 失败, 新增 {total_records} 条记录",
            'total': len(results),
            'success_count': success_count,
            'failed_count': failed_count,
            'total_records': total_records,
            'failed_stocks': failed_stocks,
            'details': results
        }
        
        print(f"全市场同步完成: {summary['message']}")
        
        return summary
    
    def get_sync_status_report(self) -> Dict[str, any]:
        """
        获取同步状态报告
        
        Returns:
            状态报告字典
        """
        # 获取所有标的的最后日期
        last_dates = self.get_all_stocks_last_dates()
        
        # 获取 Master Pool 标的
        config = load_config()
        master_pool = config.get('master_pool', {})
        
        master_codes = []
        for category, stocks in master_pool.items():
            master_codes.extend(stocks.keys())
        
        # 计算统计信息
        total_master = len(master_codes)
        missing_master = [code for code in master_codes if code not in last_dates or last_dates[code] is None]
        
        # 计算数据新鲜度
        today = datetime.now().date()
        fresh_counts = {
            'today': 0,
            'yesterday': 0,
            'within_3_days': 0,
            'within_week': 0,
            'older': 0,
            'missing': 0
        }
        
        for code in master_codes:
            if code not in last_dates or last_dates[code] is None:
                fresh_counts['missing'] += 1
                continue
                
            last_date = last_dates[code].date()
            days_diff = (today - last_date).days
            
            if days_diff == 0:
                fresh_counts['today'] += 1
            elif days_diff == 1:
                fresh_counts['yesterday'] += 1
            elif days_diff <= 3:
                fresh_counts['within_3_days'] += 1
            elif days_diff <= 7:
                fresh_counts['within_week'] += 1
            else:
                fresh_counts['older'] += 1
        
        return {
            'total_master': total_master,
            'missing_master': len(missing_master),
            'fresh_counts': fresh_counts,
            'last_dates': last_dates,
            'report_time': datetime.now().isoformat()
        }


# 全局实例（单例模式）
_sync_manager_instance = None

def get_sync_manager() -> SyncManager:
    """获取 SyncManager 单例实例"""
    global _sync_manager_instance
    if _sync_manager_instance is None:
        _sync_manager_instance = SyncManager()
    return _sync_manager_instance


def run_full_market_sync(force_update: bool = False) -> Dict[str, any]:
    """
    运行全市场同步（快捷函数）
    
    Args:
        force_update: 是否强制更新
        
    Returns:
        同步结果汇总
    """
    manager = get_sync_manager()
    return manager.sync_all_master_pool(force_update)


def check_and_run_after_market_sync() -> List[Dict[str, any]]:
    """
    检查并运行收盘后同步（快捷函数）
    
    Returns:
        同步结果列表
    """
    manager = get_sync_manager()
    return manager.sync_after_market()


def get_sync_status() -> Dict[str, any]:
    """
    获取同步状态（快捷函数）
    
    Returns:
        状态报告
    """
    manager = get_sync_manager()
    return manager.get_sync_status_report()


# 测试代码
if __name__ == "__main__":
    print("=== 同步管理器测试 ===")
    
    manager = SyncManager()
    
    # 测试状态报告
    print("\n1. 同步状态报告:")
    status = manager.get_sync_status_report()
    print(f"Master Pool 标的数量: {status['total_master']}")
    print(f"缺失数据标的: {status['missing_master']}")
    print("数据新鲜度:")
    for key, count in status['fresh_counts'].items():
        print(f"  {key}: {count}")
    
    # 测试同步单只股票
    print("\n2. 测试同步单只股票 (000001):")
    result = manager.sync_single_stock("000001", "平安银行", force_update=False)
    print(f"结果: {result}")
    
    # 测试全市场同步（不强制）
    print("\n3. 测试全市场同步（不强制）:")
    summary = manager.sync_all_master_pool(force_update=False)
    print(f"总结: {summary['message']}")
    
    # 测试收盘后更新
    print("\n4. 测试收盘后更新:")
    after_market_results = manager.sync_after_market()
    if after_market_results:
        print(f"收盘后更新完成，处理 {len(after_market_results)} 只标的")
    else:
        print("未到收盘后更新时间或没有需要更新的标的")
    
    print("\n=== 测试完成 ===")
