import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# 支持 Docker 容器路径 /app 的环境变量配置
CONFIG_PATH = os.environ.get('CONFIG_PATH', 'config.json')
BACKUP_DIR = os.path.join(os.path.dirname(CONFIG_PATH), 'backups')

def load_config():
    """
    加载配置文件，支持环境变量 CONFIG_PATH 或默认相对路径
    优先从.env文件读取DEEPSEEK_API_KEY，其次从config.json读取
    """
    config_data = {}
    
    # 如果配置文件存在，加载它
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            config_data = {}
    
    # 优先从.env文件读取DEEPSEEK_API_KEY
    api_key_from_env = None
    
    # 方法1: 使用python-dotenv（如果安装）
    try:
        from dotenv import load_dotenv
        load_dotenv()  # 加载.env文件
        api_key_from_env = os.environ.get('DEEPSEEK_API_KEY')
        if api_key_from_env:
            print("从.env文件读取DEEPSEEK_API_KEY")
    except ImportError:
        # 如果未安装python-dotenv，尝试手动读取.env文件
        env_file = '.env'
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            os.environ[key] = value
                            if key == 'DEEPSEEK_API_KEY':
                                api_key_from_env = value
                if api_key_from_env:
                    print("从.env文件（手动解析）读取DEEPSEEK_API_KEY")
            except Exception as e:
                print(f"读取.env文件失败: {e}")
    
    # 方法2: 直接从环境变量读取（可能已在容器中设置）
    if not api_key_from_env:
        api_key_from_env = os.environ.get('DEEPSEEK_API_KEY')
        if api_key_from_env:
            print("从系统环境变量读取DEEPSEEK_API_KEY")
    
    # 如果从.env或环境变量中找到了API密钥，更新配置数据
    if api_key_from_env:
        config_data['deepseek_api_key'] = api_key_from_env
        print("已使用.env/environment中的DEEPSEEK_API_KEY覆盖配置")
    
    return config_data

def cleanup_old_backups(backup_dir, days_to_keep=7):
    """
    清理超过指定天数的旧备份文件
    """
    if not os.path.exists(backup_dir):
        return
    
    cutoff_time = datetime.now() - timedelta(days=days_to_keep)
    
    for file_path in Path(backup_dir).glob('config_*.json'):
        try:
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_time:
                os.remove(file_path)
                print(f"清理旧备份: {file_path.name}")
        except Exception as e:
            print(f"清理备份文件失败 {file_path}: {e}")

def create_backup(config_path):
    """
    创建配置文件的备份
    """
    try:
        # 确保备份目录存在
        os.makedirs(BACKUP_DIR, exist_ok=True)
        
        # 生成带时间戳的备份文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"config_{timestamp}.json"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        # 复制配置文件到备份位置
        if os.path.exists(config_path):
            shutil.copy2(config_path, backup_path)
            print(f"已创建备份: {backup_filename}")
            
            # 清理超过7天的旧备份
            cleanup_old_backups(BACKUP_DIR)
            
            return backup_path
        else:
            print("警告: 原始配置文件不存在，跳过备份")
            return None
    except Exception as e:
        print(f"创建备份失败: {e}")
        return None

def save_config(config_data):
    """
    保存配置文件，支持环境变量 CONFIG_PATH 或默认相对路径
    保存前自动创建备份
    """
    try:
        # 创建备份（仅在文件已存在时）
        if os.path.exists(CONFIG_PATH):
            create_backup(CONFIG_PATH)
        
        # 确保配置文件目录存在
        config_dir = os.path.dirname(CONFIG_PATH)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        
        # 保存新配置文件
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        
        print(f"配置文件已保存: {CONFIG_PATH}")
    except Exception as e:
        print(f"保存配置文件失败: {e}")

def list_backups():
    """
    列出所有可用的备份文件
    """
    if not os.path.exists(BACKUP_DIR):
        print("备份目录不存在")
        return []
    
    backups = []
    for file_path in sorted(Path(BACKUP_DIR).glob('config_*.json'), reverse=True):
        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        backups.append({
            'filename': file_path.name,
            'path': str(file_path),
            'timestamp': file_time,
            'size': file_path.stat().st_size
        })
    
    return backups

def restore_backup(backup_filename):
    """
    从备份文件恢复配置
    """
    try:
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        if not os.path.exists(backup_path):
            print(f"备份文件不存在: {backup_filename}")
            return False
        
        # 恢复前创建当前配置的备份（如果有）
        if os.path.exists(CONFIG_PATH):
            emergency_backup = os.path.join(BACKUP_DIR, f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            shutil.copy2(CONFIG_PATH, emergency_backup)
            print(f"已创建紧急备份: {emergency_backup}")
        
        # 恢复备份
        shutil.copy2(backup_path, CONFIG_PATH)
        print(f"已从备份恢复: {backup_filename}")
        return True
    except Exception as e:
        print(f"恢复备份失败: {e}")
        return False

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
    
    # 保存到文件（会自动备份）
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

# 测试代码
if __name__ == "__main__":
    print("=== 配置管理模块测试 ===")
    
    # 加载配置
    config = load_config()
    print(f"配置加载成功，keys: {list(config.keys())}")
    
    # 测试备份功能
    if config:
        print("\n=== 测试备份功能 ===")
        test_config = config.copy()
        test_config['test_key'] = 'test_value'
        save_config(test_config)
        
        # 列出备份
        backups = list_backups()
        print(f"找到 {len(backups)} 个备份:")
        for backup in backups[:3]:  # 显示最近3个备份
            print(f"  - {backup['filename']} ({backup['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})")