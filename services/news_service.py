"""
新闻服务模块：负责抓取新闻、调用 DeepSeek 分析、解析评分/性质/摘要
"""
# 在import任何库之前，先清除所有代理环境变量
import os
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('NO_PROXY', None)
os.environ.pop('no_proxy', None)

# 导入并彻底patch requests和urllib3
import requests
from requests.adapters import HTTPAdapter
import urllib3
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry

# 保存原始方法
_original_get = requests.get
_original_post = requests.post
_original_request = requests.request
_original_session_init = requests.Session.__init__
_original_session_request = requests.Session.request
_original_adapter_init = HTTPAdapter.__init__
_original_poolmanager_init = PoolManager.__init__

def _patched_get(url, params=None, **kwargs):
    """移除所有 proxies 参数的 GET 请求"""
    kwargs.pop('proxies', None)
    kwargs.pop('verify', None)  # 也移除 SSL 验证以避免证书问题
    return _original_get(url, params=params, proxies={}, verify=False, **kwargs)

def _patched_post(url, data=None, json=None, **kwargs):
    """移除所有 proxies 参数的 POST 请求"""
    kwargs.pop('proxies', None)
    kwargs.pop('verify', None)
    return _original_post(url, data=data, json=json, proxies={}, verify=False, **kwargs)

def _patched_request(method, url, **kwargs):
    """移除所有 proxies 参数的通用请求"""
    kwargs.pop('proxies', None)
    kwargs.pop('verify', None)
    return _original_request(method, url, proxies={}, verify=False, **kwargs)

def _patched_session_init(self, *args, **kwargs):
    """Session 初始化时移除 proxies"""
    kwargs.pop('proxies', None)
    _original_session_init(self)
    self.proxies = {}
    self.trust_env = False
    self.verify = False

def _patched_session_request(self, method, url, **kwargs):
    """Session 请求时移除 proxies"""
    kwargs.pop('proxies', None)
    kwargs.pop('verify', None)
    return _original_session_request(self, method, url, proxies={}, verify=False, **kwargs)

def _patched_adapter_init(self, *args, **kwargs):
    """HTTPAdapter 初始化时移除 proxies 和其他可能冲突的参数"""
    # 移除所有可能导致问题的参数
    kwargs.pop('proxies', None)
    kwargs.pop('proxy_manager', None)
    kwargs.pop('proxy_headers', None)
    # 调用原始初始化
    try:
        _original_adapter_init(self, *args, **kwargs)
    except TypeError as e:
        # 如果仍然有参数问题，尝试只传递已知的安全参数
        safe_kwargs = {k: v for k, v in kwargs.items() if k in ['pool_connections', 'pool_maxsize', 'max_retries', 'pool_block']}
        _original_adapter_init(self, *args, **safe_kwargs)

def _patched_poolmanager_init(self, *args, **kwargs):
    """PoolManager 初始化时移除所有代理相关参数"""
    # 移除所有代理相关参数
    kwargs.pop('proxies', None)
    kwargs.pop('proxy', None)
    kwargs.pop('proxy_url', None)
    kwargs.pop('proxy_headers', None)
    kwargs.pop('proxy_config', None)

    # 调用原始初始化，捕获可能的参数错误
    try:
        _original_poolmanager_init(self, *args, **kwargs)
    except TypeError as e:
        # 如果参数错误，尝试只传递安全参数
        safe_kwargs = {k: v for k, v in kwargs.items()
                      if k in ['num_pools', 'headers', 'timeout', 'retries', 'block',
                               'cert_reqs', 'ca_certs', 'ssl_version', 'maxsize']}
        _original_poolmanager_init(self, *args, **safe_kwargs)

# 应用所有 patches
requests.get = _patched_get
requests.post = _patched_post
requests.request = _patched_request
requests.Session.__init__ = _patched_session_init
requests.Session.request = _patched_session_request
HTTPAdapter.__init__ = _patched_adapter_init
PoolManager.__init__ = _patched_poolmanager_init

# 禁用 urllib3 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 现在可以安全地import akshare
import akshare as ak
from ai_analyzer import deepseek_analyze


class NewsServiceError(Exception):
    """新闻服务异常"""
    pass


def fetch_and_analyze_news(stock_code: str, stock_name: str, news_count: int, api_key: str):
    """
    抓取新闻并进行分析
    
    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        news_count: 新闻数量
        api_key: DeepSeek API Key
    
    Returns:
        dict: 包含以下键的字典：
            name: 股票名称
            news: 新闻列表，每个元素是一个字典，包含：
                title: 新闻标题
                stars: 评分表情符号
                nature: 性质（利好/利空/中性）
                color: 颜色（red/green/grey）
                reason: 分析理由
                summary: 内容摘要
                date: 发布日期
                url: 文章链接
    
    Raises:
        NewsServiceError: 当抓取或分析过程中出现错误时抛出
    """
    try:
        # 使用 akshare 获取新闻
        news_df = ak.stock_news_em(symbol=stock_code).head(news_count)
        
        news_list = []
        for _, row in news_df.iterrows():
            # 获取发布日期（akshare返回的字段名是'发布时间'）
            publish_date = row.get('发布时间', row.get('新闻发布时间'))

            # 调用 deepseek_analyze 函数进行分析
            analysis_result = deepseek_analyze(api_key, row['新闻标题'], stock_name, publish_date)

            # 构建新闻条目（akshare返回的字段名是'新闻链接'）
            news_item = {
                'title': row['新闻标题'],
                'stars': analysis_result['stars'],
                'nature': analysis_result['nature'],
                'color': analysis_result['color'],
                'reason': analysis_result['reason'],
                'summary': analysis_result['summary'],
                'date': analysis_result['date'],
                'url': row.get('新闻链接', row.get('文章链接', ''))
            }
            news_list.append(news_item)
        
        return {
            'name': stock_name,
            'news': news_list
        }
        
    except Exception as e:
        # 包装异常，提供更有用的错误信息
        raise NewsServiceError(f"获取 {stock_name}({stock_code}) 的新闻失败：{e}")


def batch_fetch_and_analyze(stock_dict: dict, news_count: int, api_key: str):
    """
    批量获取多个股票的新闻并分析

    Args:
        stock_dict: 股票字典，键为股票代码，值为股票名称
        news_count: 新闻数量
        api_key: DeepSeek API Key

    Returns:
        dict: 以股票代码为键，分析结果字典为值的字典

    注意:
        即使部分股票处理失败，也会返回成功的结果
        失败的股票会在日志中记录，不会影响其他股票的处理
    """
    results = {}
    errors = []

    for stock_code, stock_name in stock_dict.items():
        try:
            result = fetch_and_analyze_news(stock_code, stock_name, news_count, api_key)
            results[stock_code] = result
        except Exception as e:
            error_msg = f"{stock_name}({stock_code}): {str(e)}"
            errors.append(error_msg)
            print(f"⚠️ 新闻获取失败 - {error_msg}")

    # 如果所有股票都失败了，才抛出异常
    if errors and not results:
        raise NewsServiceError(f"批量处理全部失败：{'；'.join(errors)}")

    # 如果有部分成功，打印警告但返回成功的结果
    if errors:
        print(f"⚠️ {len(errors)} 个股票处理失败，{len(results)} 个成功")

    return results
