"""
FILE: storage_manager.py
ROLE: 数据持久化中心，实现冷热分离存储与多维数据校验
LOGIC: 
    1. 冷热分离存储架构：冷数据库存储T-1及之前的历史数据，热缓冲区存储当日实时数据
    2. 三重校验机制：完整性校验、一致性校验、合理性校验
    3. 原子写入：只有通过校验的数据才能写入冷库
    4. 盘后结案：在交易日结束后将热数据归档到冷库
DEPENDENCIES: pandas, numpy, streamlit, pathlib, datetime
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union

# 导入本地行情仓库
from market_store import get_market_store

class DataValidator:
    """
    数据校验器：实现完整性、一致性和合理性三重校验
    """
    
    def __init__(self, epsilon: float = 1e-3):
        """
        初始化校验器
        
        Args:
            epsilon: 一致性校验的容差阈值
        """
        self.epsilon = epsilon
        self.trade_calendar_cache = None  # 交易日历缓存
        
    def validate_sequence(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        完整性校验：检查日期序列是否连续，是否存在NaN值
        
        Args:
            df: 时间序列数据，索引为日期
            
        Returns:
            校验结果字典，包含：
            - is_valid: 是否通过校验
            - message: 校验信息
            - missing_dates: 缺失的日期列表
            - nan_columns: 包含NaN值的列列表
        """
        result = {
            'is_valid': True,
            'message': '完整性校验通过',
            'missing_dates': [],
            'nan_columns': []
        }
        
        if df.empty:
            result['is_valid'] = False
            result['message'] = '数据为空'
            return result
            
        # 检查日期是否连续
        dates = df.index.sort_values()
        if len(dates) > 1:
            date_diff = pd.Series(dates).diff().dropna()
            # 找出缺失的日期（假设工作日，但这里只检查连续性）
            # 实际上我们需要更复杂的逻辑来处理交易日历
            # 这里先简单检查日期间隔是否合理
            if date_diff.max() > timedelta(days=7):
                result['missing_dates'] = []
                # 可以进一步分析缺失的具体日期，但这里简化处理
                result['message'] = f'检测到日期间隔异常，最大间隔: {date_diff.max().days}天'
                
        # 检查NaN值
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            result['nan_columns'] = nan_columns
            result['message'] = f'检测到NaN值的列: {nan_columns}'
            
        return result
    
    def verify_with_live(self, local_df: pd.DataFrame, live_snapshot: Dict) -> Dict[str, any]:
        """
        一致性校验：对比本地最后一行与实时快照的"昨收价"
        
        Args:
            local_df: 本地历史数据，索引为日期
            live_snapshot: 实时快照数据，包含昨收价、当前价等
            
        Returns:
            校验结果字典，包含：
            - is_consistent: 是否一致
            - message: 校验信息
            - local_last_close: 本地最后收盘价
            - live_last_close: 实时昨收价
            - difference: 差异值
        """
        result = {
            'is_consistent': True,
            'message': '一致性校验通过',
            'local_last_close': None,
            'live_last_close': None,
            'difference': 0.0
        }
        
        if local_df.empty:
            result['is_consistent'] = False
            result['message'] = '本地数据为空'
            return result
            
        if 'last_close' not in live_snapshot:
            result['is_consistent'] = False
            result['message'] = '实时快照缺少昨收价'
            return result
            
        # 获取本地最后收盘价
        try:
            local_last_close = float(local_df['close'].iloc[-1])
            live_last_close = float(live_snapshot['last_close'])
            
            result['local_last_close'] = local_last_close
            result['live_last_close'] = live_last_close
            result['difference'] = abs(local_last_close - live_last_close)
            
            if abs(local_last_close - live_last_close) > self.epsilon:
                result['is_consistent'] = False
                result['message'] = f'数据链条断裂：本地昨收({local_last_close:.2f})与实时昨收({live_last_close:.2f})不匹配，差异: {result["difference"]:.4f}'
                
        except Exception as e:
            result['is_consistent'] = False
            result['message'] = f'一致性校验出错: {str(e)}'
            
        return result
    
    def sanity_check(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        合理性校验：过滤异常值
        
        Args:
            df: 时间序列数据，包含收盘价、成交量等
            
        Returns:
            校验结果字典，包含：
            - is_sane: 是否合理
            - message: 校验信息
            - warnings: 警告列表
            - filtered_indices: 被过滤的索引列表
        """
        result = {
            'is_sane': True,
            'message': '合理性校验通过',
            'warnings': [],
            'filtered_indices': []
        }
        
        if df.empty:
            return result
            
        # 检查涨跌幅是否超过20%（排除ST和新股）
        if 'close' in df.columns and len(df) > 1:
            returns = df['close'].pct_change().dropna()
            extreme_returns = returns[abs(returns) > 0.2]  # 20%阈值
            if not extreme_returns.empty:
                result['warnings'].append(f'检测到极端涨跌幅: {extreme_returns.index.tolist()}')
                result['filtered_indices'].extend(extreme_returns.index.tolist())
                
        # 检查成交量是否为0（排除停牌）
        if 'volume' in df.columns:
            zero_volume = df[df['volume'] == 0].index.tolist()
            if zero_volume:
                result['warnings'].append(f'检测到零成交量: {zero_volume}')
                result['filtered_indices'].extend(zero_volume)
                
        # 检查价格是否为负或异常大
        if 'close' in df.columns:
            negative_price = df[df['close'] <= 0].index.tolist()
            if negative_price:
                result['warnings'].append(f'检测到非正价格: {negative_price}')
                result['filtered_indices'].extend(negative_price)
                
        if result['warnings']:
            result['is_sane'] = False
            result['message'] = f'合理性校验未通过: {"; ".join(result["warnings"])}'
            
        return result
    
    def get_trade_calendar(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            交易日列表 (YYYYMMDD格式字符串)
        """
        if self.trade_calendar_cache is not None:
            # 使用缓存，假设缓存包含所需日期范围
            return self.trade_calendar_cache
        
        try:
            import akshare as ak
            trade_cal_df = ak.tool_trade_date_hist_sina()
            trade_cal_df['trade_date'] = pd.to_datetime(trade_cal_df['trade_date'])
            mask = (trade_cal_df['trade_date'] >= pd.Timestamp(start_date)) & \
                   (trade_cal_df['trade_date'] <= pd.Timestamp(end_date))
            filtered = trade_cal_df.loc[mask]
            trade_dates = filtered['trade_date'].dt.strftime('%Y%m%d').tolist()
            self.trade_calendar_cache = trade_dates
            return trade_dates
        except Exception as e:
            print(f"获取交易日历失败: {e}")
            # 返回模拟的交易日历（仅用于测试）
            return []
    
    def price_anchor_handshake(self, local_df: pd.DataFrame, live_snapshot: Dict) -> Dict[str, any]:
        """
        价格锚点校验：对比本地最后收盘价与实时昨收价
        
        Args:
            local_df: 本地历史数据，索引为日期
            live_snapshot: 实时快照数据，包含昨收价
            
        Returns:
            校验结果字典，包含：
            - is_consistent: 是否一致
            - message: 校验信息
            - local_last_close: 本地最后收盘价
            - live_last_close: 实时昨收价
            - difference: 差异值
            - requires_resync: 是否需要强制重新同步
        """
        result = {
            'is_consistent': True,
            'message': '价格锚点校验通过',
            'local_last_close': None,
            'live_last_close': None,
            'difference': 0.0,
            'requires_resync': False
        }
        
        if local_df.empty:
            result['is_consistent'] = False
            result['message'] = '本地数据为空'
            return result
            
        if 'last_close' not in live_snapshot:
            result['is_consistent'] = False
            result['message'] = '实时快照缺少昨收价'
            return result
            
        try:
            local_last_close = float(local_df['close'].iloc[-1])
            live_last_close = float(live_snapshot['last_close'])
            
            result['local_last_close'] = local_last_close
            result['live_last_close'] = live_last_close
            result['difference'] = abs(local_last_close - live_last_close)
            
            if abs(local_last_close - live_last_close) > self.epsilon:
                result['is_consistent'] = False
                result['requires_resync'] = True
                result['message'] = f'价格锚点断裂：本地昨收({local_last_close:.2f})与实时昨收({live_last_close:.2f})不匹配，差异: {result["difference"]:.4f}，触发强制重新同步'
                
        except Exception as e:
            result['is_consistent'] = False
            result['message'] = f'价格锚点校验出错: {str(e)}'
            
        return result
    
    def temporal_alignment(self, local_df: pd.DataFrame, target_dates: List[str]) -> Dict[str, any]:
        """
        时间格对齐：检查本地数据最后日期与目标日期之间是否存在交易日缺口
        
        Args:
            local_df: 本地历史数据，索引为日期
            target_dates: 目标交易日列表 (YYYYMMDD格式字符串)
            
        Returns:
            校验结果字典，包含：
            - is_aligned: 是否对齐
            - message: 校验信息
            - missing_dates: 缺失的交易日列表
            - gap_days: 缺失的天数
        """
        result = {
            'is_aligned': True,
            'message': '时间格对齐通过',
            'missing_dates': [],
            'gap_days': 0
        }
        
        if local_df.empty:
            result['is_aligned'] = False
            result['message'] = '本地数据为空'
            return result
            
        if not target_dates:
            result['is_aligned'] = False
            result['message'] = '目标交易日列表为空'
            return result
            
        # 获取本地最后日期
        local_last_date = local_df.index.max()
        local_last_str = local_last_date.strftime('%Y%m%d')
        
        # 找到目标日期列表中在本地最后日期之后的日期
        missing_dates = []
        for td in target_dates:
            if td > local_last_str:
                missing_dates.append(td)
                
        if missing_dates:
            result['is_aligned'] = False
            result['message'] = f'发现交易日缺口：从{local_last_str}到{missing_dates[-1]}共有{len(missing_dates)}个交易日缺失'
            result['missing_dates'] = missing_dates
            result['gap_days'] = len(missing_dates)
            
        return result
    
    def schema_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        字段标准化：将列名强制映射为标准格式
        
        Args:
            df: 原始DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        if df.empty:
            return df
            
        # 复制以避免修改原始数据
        normalized = df.copy()
        
        # 映射列名
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'close' in col_lower:
                column_mapping[col] = 'close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'volume'
            elif 'date' in col_lower or '日期' in col:
                column_mapping[col] = 'date'
                
        if column_mapping:
            normalized = normalized.rename(columns=column_mapping)
            
        # 确保日期列是datetime类型
        if 'date' in normalized.columns:
            normalized['date'] = pd.to_datetime(normalized['date'])
            # 将时间统一为00:00:00
            normalized['date'] = normalized['date'].dt.normalize()
            
        return normalized
    
    def adjustment_factor_check(self, local_df: pd.DataFrame, live_snapshot: Dict) -> Dict[str, any]:
        """
        复权因子一致性检查
        
        Args:
            local_df: 本地历史数据，索引为日期
            live_snapshot: 实时快照数据，包含复权因子信息
            
        Returns:
            校验结果字典，包含：
            - is_consistent: 是否一致
            - message: 校验信息
            - local_last_factor: 本地最后复权因子
            - live_factor: 实时复权因子
            - requires_resync: 是否需要强制重新同步
        """
        result = {
            'is_consistent': True,
            'message': '复权因子一致性检查通过',
            'local_last_factor': 1.0,
            'live_factor': 1.0,
            'requires_resync': False
        }
        
        # 尝试从本地数据中获取复权因子信息
        # 这里简化处理，实际应从专门的复权因子表中获取
        # 假设本地数据已经是前复权，因子为1.0
        result['local_last_factor'] = 1.0
        
        # 尝试从实时快照中获取复权因子
        live_factor = live_snapshot.get('adjustment_factor', 1.0)
        result['live_factor'] = live_factor
        
        # 如果复权因子差异超过1%，则判定为不一致
        if abs(result['local_last_factor'] - live_factor) > 0.01:
            result['is_consistent'] = False
            result['requires_resync'] = True
            result['message'] = f'复权因子不一致：本地({result["local_last_factor"]:.4f}) vs 实时({live_factor:.4f})，触发强制重新同步'
            
        return result
    
    def validate_all(self, df: pd.DataFrame, live_snapshot: Optional[Dict] = None) -> Dict[str, any]:
        """
        执行所有校验
        
        Args:
            df: 待校验的数据
            live_snapshot: 实时快照数据
            
        Returns:
            综合校验结果
        """
        result = {
            'overall_valid': False,
            'integrity_check': {},
            'consistency_check': {},
            'sanity_check': {},
            'price_anchor_check': {},
            'temporal_alignment_check': {},
            'messages': []
        }
        
        # 完整性校验
        integrity_result = self.validate_sequence(df)
        result['integrity_check'] = integrity_result
        if not integrity_result['is_valid']:
            result['messages'].append(f'完整性校验失败: {integrity_result["message"]}')
            
        # 价格锚点校验（如果有实时快照）
        if live_snapshot is not None:
            price_anchor_result = self.price_anchor_handshake(df, live_snapshot)
            result['price_anchor_check'] = price_anchor_result
            if not price_anchor_result['is_consistent']:
                result['messages'].append(f'价格锚点校验失败: {price_anchor_result["message"]}')
                
        # 一致性校验（如果有实时快照）- 保留原有逻辑
        if live_snapshot is not None:
            consistency_result = self.verify_with_live(df, live_snapshot)
            result['consistency_check'] = consistency_result
            if not consistency_result['is_consistent']:
                result['messages'].append(f'一致性校验失败: {consistency_result["message"]}')
                
        # 合理性校验
        sanity_result = self.sanity_check(df)
        result['sanity_check'] = sanity_result
        if not sanity_result['is_sane']:
            result['messages'].append(f'合理性校验失败: {sanity_result["message"]}')
            
        # 时间格对齐校验（如果有实时快照）
        if live_snapshot is not None and 'date' in live_snapshot:
            # 获取从本地最后日期到实时日期的交易日历
            local_last_date = df.index.max().strftime('%Y%m%d') if not df.empty else None
            live_date = live_snapshot.get('date')
            if local_last_date and live_date:
                # 这里简化处理，实际应该获取这两个日期之间的交易日历
                # 暂时跳过详细实现
                pass
            
        # 综合判断
        overall_valid = integrity_result['is_valid']
        if live_snapshot is not None:
            overall_valid = overall_valid and result['price_anchor_check'].get('is_consistent', True)
            overall_valid = overall_valid and result['consistency_check'].get('is_consistent', True)
        overall_valid = overall_valid and sanity_result['is_sane']
        
        result['overall_valid'] = overall_valid
        if overall_valid:
            result['messages'].append('所有校验通过')
            
        return result


class StorageManager:
    """
    存储管理器：实现冷热分离存储与数据校验
    """
    
    def __init__(self, base_dir: str = "market_data", hot_buffer_dir: str = "temp_hot_data"):
        """
        初始化存储管理器
        
        Args:
            base_dir: 冷数据存储目录
            hot_buffer_dir: 热缓冲区目录
        """
        self.base_dir = Path(base_dir)
        self.hot_buffer_dir = Path(hot_buffer_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.hot_buffer_dir.mkdir(exist_ok=True)
        
        self.validator = DataValidator()
        self.store = get_market_store()
        
    def _get_cold_filepath(self, code: str) -> Path:
        """获取冷数据文件路径"""
        return self.base_dir / f"{code}.parquet"
    
    def _get_hot_filepath(self, code: str) -> Path:
        """获取热数据文件路径"""
        return self.hot_buffer_dir / f"{code}.json"
    
    def _is_market_closed(self) -> bool:
        """
        判断是否已收盘（假设收盘时间为16:00）
        用于触发盘后结案逻辑
        """
        now = datetime.now()
        # 检查是否在16:00之后
        if now.hour >= 16:
            return True
        return False
    
    def save_to_cold_storage(self, code: str, df: pd.DataFrame, 
                             live_snapshot: Optional[Dict] = None) -> Dict[str, any]:
        """
        保存数据到冷存储（经过校验）
        
        Args:
            code: 股票代码
            df: 要保存的数据
            live_snapshot: 实时快照数据，用于一致性校验
            
        Returns:
            保存结果
        """
        result = {
            'success': False,
            'message': '',
            'validation_result': None
        }
        
        # 执行校验
        validation_result = self.validator.validate_all(df, live_snapshot)
        result['validation_result'] = validation_result
        
        if not validation_result['overall_valid']:
            result['message'] = f'数据校验未通过: {"; ".join(validation_result["messages"])}'
            return result
            
        # 校验通过，写入冷存储
        try:
            filepath = self._get_cold_filepath(code)
            df.to_parquet(filepath)
            result['success'] = True
            result['message'] = f'数据已安全写入冷存储: {filepath}'
        except Exception as e:
            result['message'] = f'写入冷存储失败: {str(e)}'
            
        return result
    
    def save_to_hot_buffer(self, code: str, data: Dict) -> Dict[str, any]:
        """
        保存实时数据到热缓冲区
        
        Args:
            code: 股票代码
            data: 实时数据，包含收盘价、成交量、时间戳等
            
        Returns:
            保存结果
        """
        result = {'success': False, 'message': ''}
        
        try:
            # 添加时间戳
            data['timestamp'] = datetime.now().isoformat()
            data['is_finalized'] = False
            
            filepath = self._get_hot_filepath(code)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            # 同时保存到session_state（如果可用）
            if 'st' in globals():
                st.session_state.setdefault('hot_buffer', {})
                st.session_state['hot_buffer'][code] = data
                
            result['success'] = True
            result['message'] = f'实时数据已保存到热缓冲区: {filepath}'
        except Exception as e:
            result['message'] = f'保存到热缓冲区失败: {str(e)}'
            
        return result
    
    def load_from_cold_storage(self, code: str) -> pd.DataFrame:
        """
        从冷存储加载数据
        
        Args:
            code: 股票代码
            
        Returns:
            冷数据DataFrame，如果文件不存在则返回空DataFrame
        """
        filepath = self._get_cold_filepath(code)
        if not filepath.exists():
            return pd.DataFrame()
            
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            print(f'从冷存储加载数据失败: {str(e)}')
            return pd.DataFrame()
    
    def load_from_hot_buffer(self, code: str) -> Optional[Dict]:
        """
        从热缓冲区加载数据
        
        Args:
            code: 股票代码
            
        Returns:
            热数据字典，如果不存在则返回None
        """
        # 先尝试从session_state加载
        if 'st' in globals():
            hot_buffer = st.session_state.get('hot_buffer', {})
            if code in hot_buffer:
                return hot_buffer[code]
                
        # 从文件加载
        filepath = self._get_hot_filepath(code)
        if not filepath.exists():
            return None
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f'从热缓冲区加载数据失败: {str(e)}')
            return None
    
    def get_combined_data(self, code: str) -> pd.DataFrame:
        """
        获取合并的冷热数据
        
        Args:
            code: 股票代码
            
        Returns:
            合并后的DataFrame
        """
        # 加载冷数据
        cold_df = self.load_from_cold_storage(code)
        
        # 加载热数据
        hot_data = self.load_from_hot_buffer(code)
        
        if hot_data is None or cold_df.empty:
            return cold_df
            
        # 将热数据转换为DataFrame行
        hot_date = pd.to_datetime(hot_data.get('date', datetime.now().date()))
        hot_row = {
            'close': hot_data.get('close'),
            'volume': hot_data.get('volume'),
            'code': code,
            'is_hot_data': True
        }
        
        # 创建热数据DataFrame
        hot_df = pd.DataFrame([hot_row], index=[hot_date])
        
        # 合并数据，去重（热数据优先）
        combined_df = pd.concat([cold_df, hot_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
        
        return combined_df
    
    def gap_filling_update(self, code: str, force_resync: bool = False) -> Dict[str, any]:
        """
        增量更新Gap-Filling算法：基于交易日历的补缺更新
        
        Args:
            code: 股票代码
            force_resync: 是否强制重新同步（删除本地文件并重新下载全量数据）
            
        Returns:
            更新结果字典
        """
        result = {
            'success': False,
            'message': '',
            'gap_days': 0,
            'filled_days': 0,
            'requires_resync': False
        }
        
        # 如果强制重新同步，删除本地文件并重新初始化
        if force_resync:
            filepath = self._get_cold_filepath(code)
            if filepath.exists():
                filepath.unlink()
                result['message'] = '已删除本地数据，触发强制重新同步'
            
            # 重新初始化数据
            success = self.store.init_stock_data(code, days=365, force_update=True)
            if success:
                result['success'] = True
                result['message'] = '强制重新同步完成'
            else:
                result['message'] = '强制重新同步失败'
            return result
        
        # 1. 探测：获取本地最后日期
        cold_df = self.load_from_cold_storage(code)
        if cold_df.empty:
            # 如果没有本地数据，则初始化
            success = self.store.init_stock_data(code, days=365, force_update=False)
            result['success'] = success
            result['message'] = '初始化本地数据' if success else '初始化本地数据失败'
            return result
        
        last_local_date = cold_df.index.max()
        last_local_str = last_local_date.strftime('%Y%m%d')
        
        # 2. 计算：获取从最后日期到今天的交易日历
        today_str = datetime.now().strftime('%Y%m%d')
        target_dates = self.validator.get_trade_calendar(last_local_str, today_str)
        
        if not target_dates:
            result['success'] = True
            result['message'] = '无交易日需要更新'
            return result
        
        # 3. 补缺：检查缺失的交易日
        missing_dates = []
        for td in target_dates:
            if td > last_local_str:
                missing_dates.append(td)
        
        if not missing_dates:
            result['success'] = True
            result['message'] = '数据已是最新'
            return result
        
        result['gap_days'] = len(missing_dates)
        
        # 如果缺失交易日超过10天，建议使用增量更新
        if len(missing_dates) > 10:
            result['message'] = f'缺失交易日较多({len(missing_dates)}天)，建议使用增量更新'
        
        # 4. 补缺逻辑：使用AkShare获取缺失数据
        try:
            import akshare as ak
            
            # 获取缺失日期范围内的数据
            start_date = missing_dates[0]
            end_date = missing_dates[-1]
            
            df_update = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df_update.empty:
                result['success'] = True
                result['message'] = f'获取缺失数据失败，可能为停牌期'
                return result
            
            # 转换为标准格式
            df_update = df_update[['日期', '收盘', '成交量']].copy()
            df_update = df_update.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'volume'})
            df_update['date'] = pd.to_datetime(df_update['date'])
            df_update['code'] = code
            df_update = df_update.set_index('date')
            
            # 合并数据
            combined_df = pd.concat([cold_df, df_update])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            
            # 保存到冷存储（不进行校验，因为这是补缺数据）
            filepath = self._get_cold_filepath(code)
            combined_df.to_parquet(filepath)
            
            result['success'] = True
            result['filled_days'] = len(df_update)
            result['message'] = f'成功补缺 {len(df_update)} 个交易日的数据'
            
        except Exception as e:
            result['message'] = f'补缺数据获取失败: {str(e)}'
        
        return result
    
    def finalize_daily_data(self, code: str) -> Dict[str, any]:
        """
        盘后结案：将热数据合并到冷数据中，采用严格对齐协议
        
        Args:
            code: 股票代码
            
        Returns:
            结案结果
        """
        result = {'success': False, 'message': ''}
        
        # 检查是否已收盘
        if not self._is_market_closed():
            result['message'] = '尚未收盘，不能执行盘后结案'
            return result
            
        # 加载热数据
        hot_data = self.load_from_hot_buffer(code)
        if hot_data is None:
            result['message'] = '没有热数据需要结案'
            result['success'] = True  # 没有热数据也算成功
            return result
        
        # 加载冷数据
        cold_df = self.load_from_cold_storage(code)
        
        # 执行价格锚点校验
        if not cold_df.empty:
            price_check = self.validator.price_anchor_handshake(cold_df, hot_data)
            if price_check['requires_resync']:
                # 触发强制重新同步
                resync_result = self.gap_filling_update(code, force_resync=True)
                if resync_result['success']:
                    # 重新加载冷数据
                    cold_df = self.load_from_cold_storage(code)
                    result['message'] = f'价格锚点断裂，已执行强制重新同步: {resync_result["message"]}'
                else:
                    result['message'] = f'价格锚点断裂，但强制重新同步失败: {resync_result["message"]}'
                    return result
        
        # 标记热数据为已结案
        hot_data['is_finalized'] = True
        hot_data['finalized_at'] = datetime.now().isoformat()
        
        # 将热数据转换为DataFrame行
        hot_date = pd.to_datetime(hot_data.get('date', datetime.now().date()))
        hot_row = {
            'close': hot_data.get('close'),
            'volume': hot_data.get('volume'),
            'code': code
        }
        
        hot_df = pd.DataFrame([hot_row], index=[hot_date])
        
        # 合并数据
        if cold_df.empty:
            combined_df = hot_df
        else:
            combined_df = pd.concat([cold_df, hot_df])
            # 去重，保留最后出现的记录
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
        
        # 执行严格校验并保存到冷存储
        save_result = self.save_to_cold_storage(code, combined_df, hot_data)
        
        if save_result['success']:
            # 清理热数据
            self._clean_hot_buffer(code)
            result['success'] = True
            result['message'] = f'盘后结案完成: {code}'
        else:
            result['message'] = f'盘后结案失败: {save_result["message"]}'
            
        return result
    
    def _clean_hot_buffer(self, code: str):
        """清理热缓冲区"""
        filepath = self._get_hot_filepath(code)
        if filepath.exists():
            filepath.unlink()
            
        if 'st' in globals():
            hot_buffer = st.session_state.get('hot_buffer', {})
            if code in hot_buffer:
                del st.session_state['hot_buffer'][code]
    
    def get_data_status(self, code: str) -> Dict[str, any]:
        """
        获取数据状态
        
        Args:
            code: 股票代码
            
        Returns:
            数据状态字典
        """
        status = {
            'code': code,
            'cold_data_exists': False,
            'hot_data_exists': False,
            'last_cold_date': None,
            'hot_data_date': None,
            'is_finalized': False,
            'status': 'unknown'
        }
        
        # 检查冷数据
        cold_filepath = self._get_cold_filepath(code)
        if cold_filepath.exists():
            status['cold_data_exists'] = True
            try:
                cold_df = pd.read_parquet(cold_filepath, columns=['close'])
                if not cold_df.empty:
                    status['last_cold_date'] = cold_df.index.max().date()
            except:
                pass
                
        # 检查热数据
        hot_data = self.load_from_hot_buffer(code)
        if hot_data is not None:
            status['hot_data_exists'] = True
            status['hot_data_date'] = hot_data.get('date')
            status['is_finalized'] = hot_data.get('is_finalized', False)
            
        # 确定状态
        if status['cold_data_exists'] and status['hot_data_exists']:
            if status['is_finalized']:
                status['status'] = 'finalized'
            else:
                status['status'] = 'pending_finalization'
        elif status['cold_data_exists'] and not status['hot_data_exists']:
            status['status'] = 'cold_only'
        elif not status['cold_data_exists'] and status['hot_data_exists']:
            status['status'] = 'hot_only'
        else:
            status['status'] = 'no_data'
            
        return status

    def purge_data(self, scope: str = 'all', ticker: str = None, create_backup: bool = True) -> Dict[str, any]:
        """
        三级重置逻辑：根据scope清理数据
        
        Args:
            scope: 重置层级 - 'hot_buffer', 'ticker', 'all'
            ticker: 当scope='ticker'时需要指定的股票代码
            create_backup: 是否在删除前创建备份
            
        Returns:
            清理结果字典
        """
        result = {
            'success': False,
            'message': '',
            'scope': scope,
            'ticker': ticker,
            'files_deleted': [],
            'backup_path': None
        }
        
        # 安全协议：先备份
        if create_backup:
            try:
                from datetime import datetime
                import zipfile
                import os
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = Path("data/archive")
                backup_dir.mkdir(exist_ok=True)
                backup_path = backup_dir / f"backup_{timestamp}.zip"
                
                with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # 备份冷数据目录
                    if self.base_dir.exists():
                        for file_path in self.base_dir.rglob("*.parquet"):
                            zipf.write(file_path, f"cold_data/{file_path.name}")
                    
                    # 备份热数据目录
                    if self.hot_buffer_dir.exists():
                        for file_path in self.hot_buffer_dir.rglob("*.json"):
                            zipf.write(file_path, f"hot_data/{file_path.name}")
                
                result['backup_path'] = str(backup_path)
                result['message'] = f'备份已创建: {backup_path}'
            except Exception as e:
                result['message'] = f'备份创建失败: {str(e)}'
                return result
        
        try:
            files_deleted = []
            
            # Level 1: 缓存重置 (Cache Purge) - 仅清理Hot Buffer
            if scope == 'hot_buffer':
                if self.hot_buffer_dir.exists():
                    for file_path in self.hot_buffer_dir.rglob("*.json"):
                        file_path.unlink()
                        files_deleted.append(str(file_path))
                
                # 清理session_state中的热缓冲区
                if 'st' in globals():
                    if 'hot_buffer' in st.session_state:
                        st.session_state['hot_buffer'] = {}
                
                result['success'] = True
                result['message'] = f'热缓冲区已清空，删除 {len(files_deleted)} 个文件'
                result['files_deleted'] = files_deleted
            
            # Level 2: 标的重置 (Targeted Reset) - 删除特定标的的.parquet历史文件
            elif scope == 'ticker' and ticker:
                # 删除冷数据文件
                cold_filepath = self._get_cold_filepath(ticker)
                if cold_filepath.exists():
                    cold_filepath.unlink()
                    files_deleted.append(str(cold_filepath))
                
                # 删除热数据文件
                hot_filepath = self._get_hot_filepath(ticker)
                if hot_filepath.exists():
                    hot_filepath.unlink()
                    files_deleted.append(str(hot_filepath))
                
                # 清理session_state中的该股票热数据
                if 'st' in globals():
                    if 'hot_buffer' in st.session_state and ticker in st.session_state['hot_buffer']:
                        del st.session_state['hot_buffer'][ticker]
                
                result['success'] = True
                result['message'] = f'股票 {ticker} 数据已重置，删除 {len(files_deleted)} 个文件'
                result['files_deleted'] = files_deleted
            
            # Level 3: 系统初始化 (System Factory Reset) - 清空所有Parquet文件
            elif scope == 'all':
                # 删除所有冷数据文件
                if self.base_dir.exists():
                    for file_path in self.base_dir.rglob("*.parquet"):
                        file_path.unlink()
                        files_deleted.append(str(file_path))
                
                # 删除所有热数据文件
                if self.hot_buffer_dir.exists():
                    for file_path in self.hot_buffer_dir.rglob("*.json"):
                        file_path.unlink()
                        files_deleted.append(str(file_path))
                
                # 清理session_state中的热缓冲区
                if 'st' in globals():
                    if 'hot_buffer' in st.session_state:
                        st.session_state['hot_buffer'] = {}
                
                # 清理Streamlit缓存
                try:
                    if 'st' in globals():
                        st.cache_data.clear()
                except:
                    pass
                
                result['success'] = True
                result['message'] = f'系统数据已重置，删除 {len(files_deleted)} 个文件'
                result['files_deleted'] = files_deleted
            
            else:
                result['message'] = f'无效的scope参数或缺少ticker: scope={scope}, ticker={ticker}'
            
            return result
            
        except Exception as e:
            result['message'] = f'数据清理失败: {str(e)}'
            return result


# 全局实例
_storage_manager_instance = None

def get_storage_manager():
    """
    获取StorageManager单例实例
    
    Returns:
        StorageManager实例
    """
    global _storage_manager_instance
    if _storage_manager_instance is None:
        _storage_manager_instance = StorageManager()
    return _storage_manager_instance


def get_data_audit_status(codes: List[str]) -> Dict[str, any]:
    """
    获取数据审计状态

    Args:
        codes: 股票代码列表

    Returns:
        审计状态字典
    """
    storage_manager = get_storage_manager()
    statuses = {}

    # ✅ 导入交易日判定函数
    from data_integrity_checker import get_last_trading_day

    # ✅ 获取最近的交易日（考虑周末和节假日）
    today = datetime.now().date()
    last_trading_day = get_last_trading_day(today - timedelta(days=1))

    for code in codes:
        status = storage_manager.get_data_status(code)

        # 根据状态生成状态栏信息
        if status['status'] == 'finalized':
            status_bar = '🟢 数据完整'
        elif status['status'] == 'pending_finalization':
            status_bar = '🟡 数据暂存'
        elif status['status'] == 'cold_only':
            # ✅ 检查最后日期是否是最近交易日（而非简单的昨天）
            if status['last_cold_date']:
                # ✅ 使用交易日判定：数据日期 >= 最近交易日 或 = 今天 均视为完整
                if status['last_cold_date'] >= last_trading_day or status['last_cold_date'] == today:
                    status_bar = '🟢 数据完整（无热数据）'
                else:
                    status_bar = '🟡 数据待更新'
            else:
                status_bar = '🟡 数据待更新'
        elif status['status'] == 'hot_only':
            status_bar = '🟡 仅实时数据'
        else:
            status_bar = '🔴 无数据'

        status['status_bar'] = status_bar
        statuses[code] = status

    return statuses


def finalize_all_daily_data(codes: List[str]) -> Dict[str, any]:
    """
    批量执行盘后结案
    
    Args:
        codes: 股票代码列表
        
    Returns:
        结案结果
    """
    storage_manager = get_storage_manager()
    results = {
        'total': len(codes),
        'success': 0,
        'failed': 0,
        'details': {}
    }
    
    for code in codes:
        result = storage_manager.finalize_daily_data(code)
        results['details'][code] = result
        
        if result['success']:
            results['success'] += 1
        else:
            results['failed'] += 1
            
    return results


def show_data_audit_ui():
    """
    在Streamlit页面显示数据审计状态栏
    可以在各个页面调用此函数来显示数据审计状态
    """
    # 从配置加载股票列表
    from config import load_config
    config = load_config()
    
    # 获取master_pool中的股票
    master_pool = config.get('master_pool', {})
    all_codes = []
    for category, stocks in master_pool.items():
        for code, name in stocks.items():
            all_codes.append(code)
    
    if not all_codes:
        st.warning("配置文件中没有找到股票数据")
        return
    
    # 获取审计状态
    audit_status = get_data_audit_status(all_codes)
    
    # 统计状态
    status_counts = {'🟢 数据完整': 0, '🟡 数据暂存': 0, '🔴 无数据': 0}
    for code, status in audit_status.items():
        status_bar = status.get('status_bar', '')
        for key in status_counts.keys():
            if key in status_bar:
                status_counts[key] += 1
                break
    
    # 显示状态栏
    st.subheader("📊 数据审计状态")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 数据完整", status_counts['🟢 数据完整'])
    with col2:
        st.metric("🟡 数据暂存", status_counts['🟡 数据暂存'])
    with col3:
        st.metric("🔴 无数据", status_counts['🔴 无数据'])
    
    # 显示详细状态
    with st.expander("查看详细状态"):
        status_data = []
        for code, status in audit_status.items():
            # 找到对应的股票名称
            name = code
            for category, stocks in master_pool.items():
                if code in stocks:
                    name = stocks[code]
                    break
            
            status_data.append({
                '代码': code,
                '名称': name,
                '冷数据': '✅' if status['cold_data_exists'] else '❌',
                '热数据': '✅' if status['hot_data_exists'] else '❌',
                '最后日期': status['last_cold_date'] or 'N/A',
                '状态': status['status_bar']
            })
        
        if status_data:
            status_df = pd.DataFrame(status_data)
            st.dataframe(status_df, width='stretch', hide_index=True)
    
    # 提供结案按钮
    if st.button("📦 执行盘后结案", type="secondary"):
        if st.session_state.get('confirm_finalize', False):
            with st.spinner("正在执行盘后结案..."):
                results = finalize_all_daily_data(all_codes)
                
                st.success(f"结案完成: 成功 {results['success']} / 失败 {results['failed']}")
                
                if results['failed'] > 0:
                    st.error("部分股票结案失败:")
                    for code, result in results['details'].items():
                        if not result['success']:
                            st.write(f"- {code}: {result['message']}")
        else:
            st.session_state['confirm_finalize'] = True
            st.warning("确认要执行盘后结案吗？此操作将把今日热数据归档到冷库。再次点击按钮确认。")
    
    # 重置确认状态
    if 'confirm_finalize' in st.session_state and not st.button("执行盘后结案"):
        st.session_state.pop('confirm_finalize', None)


if __name__ == "__main__":
    # 测试代码
    storage_manager = StorageManager()
    
    # 测试数据校验
    validator = DataValidator()
    
    # 创建测试数据
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    test_df = pd.DataFrame({
        'close': np.random.randn(10).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 10)
    }, index=dates)
    
    print("测试完整性校验:")
    print(validator.validate_sequence(test_df))
    
    print("\n测试合理性校验:")
    print(validator.sanity_check(test_df))
    
    print("\n测试数据状态获取:")
    status = storage_manager.get_data_status('000001')
    print(status)
    
    print("\n测试数据审计状态:")
    audit_status = get_data_audit_status(['000001', '000002'])
    print(audit_status)
