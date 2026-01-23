"""
服务层模块：提供数据、行情、新闻等核心服务
"""
from .market_data_service import (
    MarketDataServiceError,
    fetch_stock_data,
    fetch_candle_data,
    fetch_latest_prices,
    search_stock_info,
    get_portfolio_summary,
    get_recent_10_days,
    safe_fetch_stock_data,
    safe_fetch_latest_prices,
    safe_fetch_candle_data,
    safe_search_stock_info,
    safe_get_signals
)

from .market_stock_list import (
    MarketStockListError,
    get_market_stock_list_info,
    refresh_market_stock_list,
    init_market_stock_list_if_needed,
    safe_search_local_stock
)

from .data_service import (
    DataServiceError,
    initialize_page_data,
    get_latest_prices,
    get_trading_signals,
    get_candle_data,
    get_recent_10_days
)

from .news_service import (
    NewsServiceError,
    fetch_and_analyze_news,
    batch_fetch_and_analyze
)

from .sync_manager import (
    SyncManagerError,
    get_sync_manager,
    run_full_market_sync,
    check_and_run_after_market_sync,
    get_sync_status
)

__all__ = [
    'MarketDataServiceError',
    'fetch_stock_data',
    'fetch_candle_data',
    'fetch_latest_prices',
    'search_stock_info',
    'get_portfolio_summary',
    'get_recent_10_days',
    'safe_fetch_stock_data',
    'safe_fetch_latest_prices',
    'safe_fetch_candle_data',
    'safe_search_stock_info',
    'safe_get_signals',
    'MarketStockListError',
    'get_market_stock_list_info',
    'refresh_market_stock_list',
    'init_market_stock_list_if_needed',
    'safe_search_local_stock',
    'DataServiceError',
    'initialize_page_data',
    'get_latest_prices',
    'get_trading_signals',
    'get_candle_data',
    'get_recent_10_days',
    'NewsServiceError',
    'fetch_and_analyze_news',
    'batch_fetch_and_analyze',
    'SyncManagerError',
    'get_sync_manager',
    'run_full_market_sync',
    'check_and_run_after_market_sync',
    'get_sync_status'
]
