import json
import os
from datetime import datetime

CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config_data):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)

def save_trade(config_data, trade_data):
    """
    将单笔交易添加到portfolio列表并保存到配置文件
    trade_data格式:
    {
        "code": "688506",
        "name": "百利天恒",
        "buy_price": 185.50,
        "quantity": 1000,
        "buy_date": "2026-01-05",
        "timestamp": 1704612345
    }
    """
    # 确保portfolio列表存在
    if 'portfolio' not in config_data:
        config_data['portfolio'] = []
    
    # 添加新交易
    config_data['portfolio'].append(trade_data)
    
    # 保存到文件
    save_config(config_data)
    
    return config_data

def delete_trade(config_data, index):
    """
    根据索引删除portfolio中的交易记录
    """
    if 'portfolio' in config_data and 0 <= index < len(config_data['portfolio']):
        del config_data['portfolio'][index]
        save_config(config_data)
    return config_data

def clear_portfolio(config_data):
    """
    清空portfolio列表
    """
    if 'portfolio' in config_data:
        config_data['portfolio'] = []
        save_config(config_data)
    return config_data
