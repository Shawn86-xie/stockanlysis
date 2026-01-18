"""
FILE: database_manager.py
ROLE: SQLite数据库管理模块，用于存储宏观指标和AI诊断报告
LOGIC:
    1. 创建med_portfolio.db数据库，包含macro_indicators和ai_reports表
    2. 提供数据持久化函数，支持数据去重和Schema锁定
    3. 实现定期备份功能（每周备份）
DEPENDENCIES: pandas, sqlite3, datetime, pathlib, shutil
"""

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Union
import time

class DatabaseManager:
    """
    SQLite数据库管理器
    """
    
    def __init__(self, db_path: str = "med_portfolio.db", backup_dir: str = "backups"):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
            backup_dir: 备份目录
        """
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建宏观指标表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS macro_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator_name TEXT NOT NULL,
            indicator_value REAL,
            indicator_date DATE NOT NULL,
            percentile REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(indicator_name, indicator_date)
        )
        """)
        
        # 创建AI报告表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT NOT NULL,
            report_date DATE NOT NULL,
            overall_assessment TEXT,
            confidence_score REAL,
            market_timing TEXT,
            technical_analysis TEXT,
            key_insights TEXT,
            investment_recommendations TEXT,
            risk_warnings TEXT,
            raw_response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_type, report_date)
        )
        """)
        
        # 创建股票数据状态表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            name TEXT,
            last_update_date DATE,
            record_count INTEGER,
            data_status TEXT,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(code)
        )
        """)
        
        # 创建索引以提高查询性能
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_indicators(indicator_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_name ON macro_indicators(indicator_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_date ON ai_reports(report_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_code ON stock_data_status(code)")
        
        conn.commit()
        conn.close()
        print(f"数据库初始化完成: {self.db_path}")
        
    def save_to_db(self, df: pd.DataFrame, table_name: str, deduplicate: bool = True, 
                   date_column: str = 'date') -> Dict[str, any]:
        """
        保存数据到数据库，支持数据去重和Schema锁定
        
        Args:
            df: 要保存的DataFrame
            table_name: 表名
            deduplicate: 是否去重
            date_column: 日期列名（用于去重）
            
        Returns:
            保存结果字典
        """
        result = {
            'success': False,
            'message': '',
            'rows_affected': 0
        }
        
        if df.empty:
            result['message'] = '数据为空，不保存'
            return result
            
        try:
            # 确保日期列是datetime64格式
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                # 将时间统一为00:00:00
                df[date_column] = df[date_column].dt.normalize()
            
            # 数据去重（如果需要）
            if deduplicate and date_column in df.columns:
                df = df.drop_duplicates(subset=[date_column], keep='last')
            
            conn = sqlite3.connect(self.db_path)
            
            # 根据表名确定插入策略
            if table_name == 'macro_indicators':
                # 确保列名匹配
                required_cols = ['indicator_name', 'indicator_value', 'indicator_date', 'percentile']
                for col in required_cols:
                    if col not in df.columns:
                        print(f"警告：DataFrame缺少列 {col}")
                        # 尝试从现有列映射
                        if col == 'indicator_date' and 'date' in df.columns:
                            df = df.rename(columns={'date': 'indicator_date'})
                        elif col == 'indicator_name' and 'name' in df.columns:
                            df = df.rename(columns={'name': 'indicator_name'})
                
                # 插入或替换数据
                df.to_sql('macro_indicators', conn, if_exists='append', index=False)
                
            elif table_name == 'ai_reports':
                # 处理AI报告数据
                df.to_sql('ai_reports', conn, if_exists='append', index=False)
                
            elif table_name == 'stock_data_status':
                # 处理股票数据状态
                df.to_sql('stock_data_status', conn, if_exists='replace', index=False)
                
            else:
                # 通用表
                df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            result['rows_affected'] = len(df)
            result['success'] = True
            result['message'] = f'成功保存 {len(df)} 行数据到表 {table_name}'
            
            conn.close()
            
        except Exception as e:
            result['message'] = f'保存数据到数据库失败: {str(e)}'
            print(f"数据库保存错误: {e}")
            
        return result
        
    def load_from_db(self, table_name: str, query: str = None) -> pd.DataFrame:
        """
        从数据库加载数据
        
        Args:
            table_name: 表名
            query: 自定义查询语句（可选）
            
        Returns:
            加载的DataFrame
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            if query:
                df = pd.read_sql_query(query, conn)
            else:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            
            conn.close()
            
            # 确保日期列是datetime格式
            for col in df.columns:
                if 'date' in col.lower() or 'created_at' in col.lower():
                    df[col] = pd.to_datetime(df[col])
                    
            return df
            
        except Exception as e:
            print(f"从数据库加载数据失败: {str(e)}")
            return pd.DataFrame()
    
    def check_macro_data_today(self, indicator_names: List[str] = None) -> Dict[str, bool]:
        """
        检查今天是否已更新宏观数据
        
        Args:
            indicator_names: 指标名称列表（可选）
            
        Returns:
            字典，键为指标名，值为是否已更新
        """
        today = datetime.now().date().isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if indicator_names:
                placeholders = ','.join(['?' for _ in indicator_names])
                query = f"""
                SELECT indicator_name, COUNT(*) as count 
                FROM macro_indicators 
                WHERE indicator_date = ? AND indicator_name IN ({placeholders})
                GROUP BY indicator_name
                """
                params = [today] + indicator_names
            else:
                query = """
                SELECT indicator_name, COUNT(*) as count 
                FROM macro_indicators 
                WHERE indicator_date = ?
                GROUP BY indicator_name
                """
                params = [today]
                
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            conn.close()
            
            # 构建结果字典
            result_dict = {}
            for row in results:
                result_dict[row[0]] = row[1] > 0
                
            return result_dict
            
        except Exception as e:
            print(f"检查宏观数据失败: {str(e)}")
            return {}
    
    def save_macro_indicator(self, name: str, value: float, date: str = None, 
                             percentile: float = None) -> bool:
        """
        保存单个宏观指标
        
        Args:
            name: 指标名称
            value: 指标值
            date: 日期（默认为今天）
            percentile: 百分位数（可选）
            
        Returns:
            是否成功
        """
        if date is None:
            date = datetime.now().date().isoformat()
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 插入或替换数据
            cursor.execute("""
            INSERT OR REPLACE INTO macro_indicators 
            (indicator_name, indicator_value, indicator_date, percentile) 
            VALUES (?, ?, ?, ?)
            """, (name, value, date, percentile))
            
            conn.commit()
            conn.close()
            
            print(f"已保存宏观指标: {name} = {value} ({date})")
            return True
            
        except Exception as e:
            print(f"保存宏观指标失败: {str(e)}")
            return False
    
    def save_ai_report(self, report_data: Dict) -> bool:
        """
        保存AI分析报告
        
        Args:
            report_data: 报告数据字典
            
        Returns:
            是否成功
        """
        try:
            # 准备数据
            report_date = report_data.get('report_date', datetime.now().date().isoformat())
            report_type = report_data.get('report_type', 'macro_analysis')
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT OR REPLACE INTO ai_reports 
            (report_type, report_date, overall_assessment, confidence_score, 
             market_timing, technical_analysis, key_insights, 
             investment_recommendations, risk_warnings, raw_response) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_type,
                report_date,
                report_data.get('overall_assessment'),
                report_data.get('confidence_score'),
                report_data.get('market_timing'),
                report_data.get('technical_analysis'),
                json.dumps(report_data.get('key_insights', []), ensure_ascii=False),
                json.dumps(report_data.get('investment_recommendations', []), ensure_ascii=False),
                json.dumps(report_data.get('risk_warnings', []), ensure_ascii=False),
                json.dumps(report_data.get('raw_response', {}), ensure_ascii=False)
            ))
            
            conn.commit()
            conn.close()
            
            print(f"已保存AI报告: {report_type} ({report_date})")
            return True
            
        except Exception as e:
            print(f"保存AI报告失败: {str(e)}")
            return False
    
    def create_backup(self) -> Dict[str, any]:
        """
        创建数据库备份
        
        Returns:
            备份结果字典
        """
        result = {
            'success': False,
            'message': '',
            'backup_path': None
        }
        
        if not self.db_path.exists():
            result['message'] = '数据库文件不存在'
            return result
            
        try:
            # 生成备份文件名（包含时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"med_portfolio_backup_{timestamp}.db"
            backup_path = self.backup_dir / backup_filename
            
            # 复制数据库文件
            shutil.copy2(self.db_path, backup_path)
            
            result['success'] = True
            result['message'] = f'数据库备份创建成功: {backup_path}'
            result['backup_path'] = str(backup_path)
            
            # 清理旧备份（保留最近7个备份）
            self._cleanup_old_backups()
            
        except Exception as e:
            result['message'] = f'创建备份失败: {str(e)}'
            
        return result
    
    def _cleanup_old_backups(self, keep_count: int = 7):
        """清理旧的备份文件，保留最近指定数量的备份"""
        try:
            backup_files = list(self.backup_dir.glob("med_portfolio_backup_*.db"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if len(backup_files) > keep_count:
                for old_backup in backup_files[keep_count:]:
                    old_backup.unlink()
                    print(f"已清理旧备份: {old_backup}")
                    
        except Exception as e:
            print(f"清理旧备份失败: {str(e)}")
    
    def get_table_info(self, table_name: str) -> Dict[str, any]:
        """
        获取表信息
        
        Args:
            table_name: 表名
            
        Returns:
            表信息字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取表结构
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # 获取行数
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # 获取最近更新时间
            cursor.execute(f"SELECT MAX(created_at) FROM {table_name}")
            last_update = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'table_name': table_name,
                'columns': [col[1] for col in columns],  # 列名
                'row_count': row_count,
                'last_update': last_update,
                'column_details': columns
            }
            
        except Exception as e:
            print(f"获取表信息失败: {str(e)}")
            return {}

    def reset_tables(self, tables: List[str] = ['macro_indicators', 'ai_reports', 'stock_data_status']) -> Dict[str, any]:
        """
        重置数据库表（清空表内容但保留表结构）
        
        Args:
            tables: 要重置的表名列表
            
        Returns:
            重置结果字典
        """
        result = {
            'success': False,
            'message': '',
            'tables_reset': [],
            'rows_deleted': 0
        }
        
        # 安全协议：先备份
        backup_result = self.create_backup()
        if not backup_result['success']:
            result['message'] = f'重置前备份失败: {backup_result["message"]}'
            return result
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            total_rows = 0
            reset_tables = []
            
            for table in tables:
                # 检查表是否存在
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if cursor.fetchone():
                    # 获取当前行数
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    
                    # 清空表内容
                    cursor.execute(f"DELETE FROM {table}")
                    
                    # 重置自增ID（如果表有自增主键）
                    cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
                    
                    total_rows += row_count
                    reset_tables.append(table)
                    print(f"已重置表 {table}，删除 {row_count} 行")
                else:
                    print(f"表 {table} 不存在，跳过")
            
            conn.commit()
            
            # 执行VACUUM压缩数据库
            cursor.execute("VACUUM")
            
            conn.close()
            
            result['success'] = True
            result['message'] = f'成功重置 {len(reset_tables)} 个表，共删除 {total_rows} 行数据'
            result['tables_reset'] = reset_tables
            result['rows_deleted'] = total_rows
            result['backup_path'] = backup_result.get('backup_path')
            
        except Exception as e:
            result['message'] = f'重置表失败: {str(e)}'
            print(f"重置表错误: {e}")
            
        return result


# 全局实例
_database_manager_instance = None

def get_database_manager():
    """
    获取DatabaseManager单例实例
    
    Returns:
        DatabaseManager实例
    """
    global _database_manager_instance
    if _database_manager_instance is None:
        _database_manager_instance = DatabaseManager()
    return _database_manager_instance


def backup_database_weekly():
    """
    每周备份数据库（可在定时任务中调用）
    """
    db_manager = get_database_manager()
    
    # 检查是否需要备份（每周一执行）
    today = datetime.now()
    if today.weekday() == 0:  # 周一
        result = db_manager.create_backup()
        if result['success']:
            print(f"周备份完成: {result['backup_path']}")
        else:
            print(f"周备份失败: {result['message']}")
        return result
    else:
        print("今天不是周一，跳过周备份")
        return {'success': True, 'message': '今天不是周一，跳过周备份'}


if __name__ == "__main__":
    # 测试代码
    db_manager = DatabaseManager()
    
    # 测试保存宏观指标
    test_date = datetime.now().date().isoformat()
    db_manager.save_macro_indicator("EPU", 150.5, test_date, 0.75)
    db_manager.save_macro_indicator("iVIX", 25.3, test_date, 0.60)
    
    # 测试检查数据
    check_result = db_manager.check_macro_data_today(["EPU", "iVIX"])
    print(f"今日数据检查结果: {check_result}")
    
    # 测试加载数据
    df = db_manager.load_from_db("macro_indicators")
    print(f"宏观指标数据: {len(df)} 行")
    
    # 测试表信息
    table_info = db_manager.get_table_info("macro_indicators")
    print(f"表信息: {table_info}")
    
    # 测试备份
    backup_result = db_manager.create_backup()
    print(f"备份结果: {backup_result}")
    
    print("数据库管理器测试完成")
