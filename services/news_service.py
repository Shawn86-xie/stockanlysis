"""
新闻服务模块：负责抓取新闻、调用 DeepSeek 分析、解析评分/性质/摘要
"""
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
            # 获取发布日期（假设列名为'新闻发布时间'，如果没有则使用None）
            publish_date = row.get('新闻发布时间')
            
            # 调用 deepseek_analyze 函数进行分析
            analysis_result = deepseek_analyze(api_key, row['新闻标题'], stock_name, publish_date)
            
            # 构建新闻条目
            news_item = {
                'title': row['新闻标题'],
                'stars': analysis_result['stars'],
                'nature': analysis_result['nature'],
                'color': analysis_result['color'],
                'reason': analysis_result['reason'],
                'summary': analysis_result['summary'],
                'date': analysis_result['date'],
                'url': row.get('文章链接', '')
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
        
    Raises:
        NewsServiceError: 当任何股票的处理失败时抛出
    """
    results = {}
    errors = []
    
    for stock_code, stock_name in stock_dict.items():
        try:
            result = fetch_and_analyze_news(stock_code, stock_name, news_count, api_key)
            results[stock_code] = result
        except Exception as e:
            errors.append(str(e))
    
    if errors:
        raise NewsServiceError(f"批量处理失败：{'；'.join(errors)}")
    
    return results
