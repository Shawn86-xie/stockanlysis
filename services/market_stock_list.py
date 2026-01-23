"""
市场标的库管理模块

功能:
1. 从AkShare获取A股全市场股票列表
2. 本地缓存到JSON文件,避免频繁网络请求
3. 支持手动刷新更新市场数据
4. 提供高效的本地搜索功能
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import akshare as ak


# 本地缓存文件路径
MARKET_LIST_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "market_data", "market_stock_list.json")


class MarketStockListError(Exception):
    """市场标的库错误"""
    pass


def ensure_market_data_dir():
    """确保market_data目录存在"""
    dir_path = os.path.dirname(MARKET_LIST_FILE)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def fetch_market_stock_list() -> List[Dict]:
    """
    从AkShare获取A股全市场股票列表

    使用东方财富实时行情接口，比交易所官网接口更稳定

    Returns:
        股票列表，每个元素包含 code 和 name
    """
    try:
        print("正在从网络获取A股市场股票列表...")

        # 使用东方财富实时行情接口（更稳定）
        # 该接口返回全市场A股数据，包含代码和名称
        stock_df = ak.stock_zh_a_spot_em()

        result = []
        for _, row in stock_df.iterrows():
            code = str(row['代码'])
            name = str(row['名称'])
            result.append({
                'code': code,
                'name': name
            })

        print(f"成功获取 {len(result)} 只股票信息")
        return result
    except Exception as e:
        # 如果东方财富接口失败，尝试备用接口
        try:
            print(f"主接口失败({e})，尝试备用接口...")
            stock_list = ak.stock_info_a_code_name()

            result = []
            for _, row in stock_list.iterrows():
                code = str(row['code'])
                name = str(row['name'])
                result.append({
                    'code': code,
                    'name': name
                })

            print(f"备用接口成功获取 {len(result)} 只股票信息")
            return result
        except Exception as e2:
            raise MarketStockListError(f"获取市场股票列表失败: 主接口({e}), 备用接口({e2})")


def save_market_stock_list(stock_list: List[Dict]) -> bool:
    """
    保存市场股票列表到本地JSON文件

    Args:
        stock_list: 股票列表

    Returns:
        是否保存成功
    """
    try:
        ensure_market_data_dir()

        data = {
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'count': len(stock_list),
            'stocks': stock_list
        }

        with open(MARKET_LIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"市场股票列表已保存到: {MARKET_LIST_FILE}")
        return True
    except Exception as e:
        print(f"保存市场股票列表失败: {e}")
        return False


def load_market_stock_list() -> Optional[Dict]:
    """
    从本地JSON文件加载市场股票列表

    Returns:
        包含 update_time, count, stocks 的字典，如果文件不存在返回 None
    """
    try:
        if not os.path.exists(MARKET_LIST_FILE):
            return None

        with open(MARKET_LIST_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data
    except Exception as e:
        print(f"加载市场股票列表失败: {e}")
        return None


def refresh_market_stock_list() -> Dict:
    """
    刷新市场股票列表（从网络获取并保存到本地）

    Returns:
        包含 success, message, count, update_time 的结果字典
    """
    try:
        stock_list = fetch_market_stock_list()

        if save_market_stock_list(stock_list):
            return {
                'success': True,
                'message': f'成功刷新市场股票列表',
                'count': len(stock_list),
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            return {
                'success': False,
                'message': '保存市场股票列表失败',
                'count': 0,
                'update_time': None
            }
    except Exception as e:
        return {
            'success': False,
            'message': str(e),
            'count': 0,
            'update_time': None
        }


def get_market_stock_list_info() -> Dict:
    """
    获取本地市场股票列表的信息

    Returns:
        包含 exists, count, update_time 的信息字典
    """
    data = load_market_stock_list()

    if data is None:
        return {
            'exists': False,
            'count': 0,
            'update_time': None
        }

    return {
        'exists': True,
        'count': data.get('count', 0),
        'update_time': data.get('update_time', '未知')
    }


def search_local_stock_list(query: str, limit: int = 20) -> List[Dict]:
    """
    从本地市场股票列表中搜索股票

    Args:
        query: 搜索关键词（代码或名称）
        limit: 返回结果数量限制

    Returns:
        匹配的股票列表
    """
    if not query or len(query.strip()) == 0:
        return []

    query = query.strip().upper()  # 代码转大写便于匹配
    query_lower = query.lower()

    # 加载本地数据
    data = load_market_stock_list()

    if data is None or 'stocks' not in data:
        return []

    stocks = data['stocks']
    results = []

    # 优先匹配：代码精确匹配
    exact_matches = []
    # 次优匹配：代码前缀匹配
    prefix_matches = []
    # 其他匹配：代码包含或名称包含
    other_matches = []

    for stock in stocks:
        code = str(stock['code'])
        name = str(stock['name'])
        code_upper = code.upper()

        # 精确匹配代码
        if code_upper == query:
            exact_matches.append(stock)
        # 代码前缀匹配
        elif code_upper.startswith(query):
            prefix_matches.append(stock)
        # 代码包含或名称包含
        elif query in code_upper or query_lower in name.lower():
            other_matches.append(stock)

    # 按优先级合并结果
    results = exact_matches + prefix_matches + other_matches

    # 限制返回数量
    return results[:limit]


def init_market_stock_list_if_needed() -> Dict:
    """
    如果本地市场股票列表不存在，则初始化

    Returns:
        操作结果
    """
    info = get_market_stock_list_info()

    if info['exists']:
        return {
            'success': True,
            'message': f"本地市场库已存在，共 {info['count']} 只股票，更新时间: {info['update_time']}",
            'initialized': False
        }

    # 不存在则刷新
    result = refresh_market_stock_list()
    result['initialized'] = result['success']
    return result


# 导出的安全搜索函数（带错误处理）
def safe_search_local_stock(query: str, limit: int = 20) -> List[Dict]:
    """
    安全的本地股票搜索函数（带错误处理）

    如果本地库不存在，返回空列表
    """
    try:
        return search_local_stock_list(query, limit)
    except Exception as e:
        print(f"搜索股票失败: {e}")
        return []
