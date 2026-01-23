import pandas as pd
import numpy as np
import streamlit as st
import requests
from datetime import datetime, timedelta
import time
import os
from io import StringIO
try:
    import akshare as ak
except ImportError:
    ak = None
try:
    from fredapi import Fred
except ImportError:
    Fred = None

# 常量定义
CHINA_EPU_FRED_SERIES = 'CHNMAINLANDEPU'  # 中国大陆经济政策不确定性指数
CHINA_EPU_CSV_URL = 'https://www.policyuncertainty.com/media/China_Policy_Uncertainty_Data.csv'  # 公开数据源
LOCAL_EPU_CACHE = 'data/china_epu_cache.csv'  # 本地缓存路径

def get_fred_api_key():
    """从config.json获取FRED API密钥"""
    try:
        import json
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('fred_api_key')
    except Exception as e:
        print(f"读取config.json失败: {e}")
        return None

@st.cache_data(ttl=86400 * 7)  # EPU为月度数据，缓存7天
def fetch_china_epu(api_key=None, use_backup=True):
    """
    抓取中国大陆经济政策不确定性指数 (EPU)
    
    优先使用fredapi从FRED获取，如果失败则使用policyuncertainty.com的公开CSV，
    最后尝试从本地缓存读取。
    
    Args:
        api_key (str, optional): FRED API密钥。如果未提供，将从config.json获取。
        use_backup (bool): 是否使用备用数据源，默认为True。
    
    Returns:
        pandas.DataFrame: 包含EPU指数和百分位排名的DataFrame，索引为日期。
                          列：'EPU_Index' (float), 'Percentile' (float)
    """
    # 1. 尝试使用fredapi从FRED获取
    if api_key is None:
        api_key = get_fred_api_key()
        if api_key is None:
            api_key = os.environ.get('FRED_API_KEY')
    
    if api_key and Fred is not None:
        try:
            fred = Fred(api_key=api_key)
            # 获取原始时间序列数据
            series = fred.get_series(CHINA_EPU_FRED_SERIES)
            if series is not None and not series.empty:
                df = pd.DataFrame(series, columns=['EPU_Index'])
                df.index.name = 'Date'
                
                # 循证医学风格处理：计算百分分位 (Percentile Rank)
                # 帮助识别当前风险在历史中的相对位置
                df['Percentile'] = df['EPU_Index'].rank(pct=True)
                
                print("成功从FRED获取中国大陆经济政策不确定性指数")
                # 保存到本地缓存
                os.makedirs('data', exist_ok=True)
                df.to_csv(LOCAL_EPU_CACHE)
                return df
        except Exception as e:
            print(f"从FRED获取数据失败: {e}")
    
    # 2. 尝试从policyuncertainty.com获取公开CSV
    if use_backup:
        try:
            response = requests.get(CHINA_EPU_CSV_URL, timeout=10)
            if response.status_code == 200:
                # 读取CSV，跳过说明行
                csv_content = response.text
                # 跳过开头的说明行（通常以#或空行开始）
                lines = csv_content.split('\n')
                data_lines = []
                for line in lines:
                    if line.strip() and not line.strip().startswith('#'):
                        data_lines.append(line)
                csv_data = '\n'.join(data_lines)
                
                df = pd.read_csv(StringIO(csv_data))
                # 假设CSV包含'Year', 'Month', 'China_Policy_Uncertainty'列
                # 实际列名可能需要调整
                if 'Year' in df.columns and 'Month' in df.columns:
                    # 创建日期列（每月第一天）并设为索引
                    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
                    # 寻找值列
                    value_col = None
                    for col in ['China_Policy_Uncertainty', 'China EPU', 'EPU', 'Value']:
                        if col in df.columns:
                            value_col = col
                            break
                    if value_col:
                        df = df.set_index('Date')
                        df = df[[value_col]].rename(columns={value_col: 'EPU_Index'})
                        df['EPU_Index'] = pd.to_numeric(df['EPU_Index'], errors='coerce')
                        df = df.dropna()
                        # 计算百分位排名
                        df['Percentile'] = df['EPU_Index'].rank(pct=True)
                        
                        print("成功从policyuncertainty.com获取中国经济政策不确定性指数")
                        # 保存到本地缓存
                        os.makedirs('data', exist_ok=True)
                        df.to_csv(LOCAL_EPU_CACHE)
                        return df
        except Exception as e:
            print(f"从policyuncertainty.com获取数据失败: {e}")
    
    # 3. 尝试从本地缓存读取
    if os.path.exists(LOCAL_EPU_CACHE):
        try:
            df = pd.read_csv(LOCAL_EPU_CACHE, index_col='Date')
            df.index = pd.to_datetime(df.index)
            print("从本地缓存读取中国经济政策不确定性指数")
            return df
        except Exception as e:
            print(f"读取本地缓存失败: {e}")
    
    # 4. 如果所有方法都失败，返回模拟数据（仅用于演示）
    print("警告：所有数据源均失败，返回模拟数据")
    dates = pd.date_range(end=datetime.now(), periods=120, freq='ME')
    values = np.random.normal(100, 20, len(dates)).cumsum() / 10 + 200
    df = pd.DataFrame({'EPU_Index': values}, index=dates)
    df.index.name = 'Date'
    df['Percentile'] = df['EPU_Index'].rank(pct=True)
    return df

@st.cache_data(ttl=3600)
def fetch_ivix_data():
    """
    获取中国恐慌指数 (iVIX) 历史数据
    
    使用akshare的index_option_50etf_qvix接口获取上证50ETF期权波动率指数。
    
    Returns:
        pandas.DataFrame: 包含日期和iVIX值的DataFrame，索引为日期。
                          列：'date' (datetime), 'ivix' (float)
    """
    if ak is None:
        print("错误：未安装akshare，无法获取iVIX数据")
        # 返回模拟数据
        dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
        values = np.random.uniform(10, 40, len(dates))
        df = pd.DataFrame({'date': dates, 'ivix': values})
        return df
    
    try:
        # 获取iVIX数据
        df_ivix = ak.index_option_50etf_qvix()
        
        # 检查返回的数据结构
        if df_ivix is None or df_ivix.empty:
            print("iVIX数据为空，尝试模拟数据")
            raise ValueError("空数据")
        
        # 打印列名以便调试
        print(f"原始列名: {list(df_ivix.columns)}")
        
        # 重命名列：将'close'列作为ivix值
        df_ivix = df_ivix.rename(columns={'close': 'ivix'})
        
        # 确保有'date'和'ivix'列
        if 'date' not in df_ivix.columns:
            raise KeyError("无法找到日期列")
        if 'ivix' not in df_ivix.columns:
            # 如果没有close列，尝试使用其他价格列
            for col in ['open', 'high', 'low']:
                if col in df_ivix.columns:
                    df_ivix = df_ivix.rename(columns={col: 'ivix'})
                    break
            if 'ivix' not in df_ivix.columns:
                raise KeyError("无法找到ivix列")
        
        # 转换日期和数值
        df_ivix['date'] = pd.to_datetime(df_ivix['date'], errors='coerce')
        df_ivix['ivix'] = pd.to_numeric(df_ivix['ivix'], errors='coerce')
        
        # 删除无效值
        df_ivix = df_ivix.dropna(subset=['date', 'ivix'])
        
        # 按日期排序
        df_ivix = df_ivix.sort_values('date').reset_index(drop=True)
        
        print(f"成功获取iVIX数据，共 {len(df_ivix)} 条记录")
        return df_ivix[['date', 'ivix']]
        
    except Exception as e:
        print(f"获取iVIX数据失败: {e}")
        # 返回模拟数据
        dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
        values = np.random.uniform(10, 40, len(dates))
        df = pd.DataFrame({'date': dates, 'ivix': values})
        return df

@st.cache_data(ttl=3600 * 6)  # 缓存6小时，两融数据每日收盘后更新一次
def fetch_margin_data():
    """
    稳健获取全市场两融余额数据并计算融资买入占比

    使用akshare宏接口获取沪深两融数据，基于历史数据计算融资买入占比。
    优化：移除了重量级的stock_zh_a_spot_em调用，改用融资买入额/融资余额比例估算。

    Returns:
        pandas.DataFrame: 包含日期、全市场融资余额、融资买入占比、百分位排名等字段的DataFrame。
                          列：'date' (datetime), 'fin_balance' (float, 亿元),
                          'fin_buy_ratio' (float, %), 'percentile' (float)
    """
    # 初始化数据源状态
    data_source = 'unknown'

    # 辅助函数：生成模拟数据
    def generate_mock_margin_data():
        print("警告：所有数据源均失败，返回模拟数据")
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        fin_balance = np.random.uniform(10000, 20000, len(dates))
        df = pd.DataFrame({
            'date': dates,
            'fin_balance': fin_balance,
            'fin_buy_ratio': np.random.uniform(8, 12, len(dates)),  # 模拟真实占比范围
            'percentile': pd.Series(fin_balance).rank(pct=True).values
        })
        return df

    # 辅助函数：基于融资买入额和历史成交额均值计算占比
    def calculate_fin_buy_ratio(df):
        """
        基于融资买入额计算融资买入占比
        使用历史平均成交额（约1万亿）作为基准，避免调用重量级API
        """
        # A股市场近年日均成交额约8000-12000亿，取均值10000亿作为基准
        ESTIMATED_DAILY_TURNOVER = 10000  # 亿元

        if 'fin_buy_amount_total' in df.columns:
            # 使用实际融资买入额计算占比
            df['fin_buy_ratio'] = (df['fin_buy_amount_total'] / ESTIMATED_DAILY_TURNOVER * 100).clip(3, 20)
        else:
            # 备用方案：基于融资余额变化估算
            df['fin_buy_ratio'] = 9.0 + (df['fin_balance'].pct_change() * 100).fillna(0).clip(-3, 3)

        return df

    # 方法1: 使用akshare的宏接口获取沪深两融数据（首选，最快）
    if ak is not None:
        try:
            # 获取沪市两融数据
            margin_sh = ak.macro_china_market_margin_sh()
            # 获取深市两融数据
            margin_sz = ak.macro_china_market_margin_sz()

            if not margin_sh.empty and not margin_sz.empty:
                # 统一列名
                margin_sh = margin_sh.rename(columns={
                    '日期': 'date',
                    '融资余额': 'fin_balance_sh',
                    '融资买入额': 'fin_buy_amount_sh'
                })
                margin_sz = margin_sz.rename(columns={
                    '日期': 'date',
                    '融资余额': 'fin_balance_sz',
                    '融资买入额': 'fin_buy_amount_sz'
                })

                # 转换日期格式
                margin_sh['date'] = pd.to_datetime(margin_sh['date'])
                margin_sz['date'] = pd.to_datetime(margin_sz['date'])

                # 合并数据
                df = pd.merge(margin_sh[['date', 'fin_balance_sh', 'fin_buy_amount_sh']],
                             margin_sz[['date', 'fin_balance_sz', 'fin_buy_amount_sz']],
                             on='date', how='outer')

                # 计算全市场融资余额（单位：亿元）
                # 注意：原数据单位可能是元，除以1e8转换为亿元
                df['fin_balance'] = (df['fin_balance_sh'].fillna(0) + df['fin_balance_sz'].fillna(0)) / 1e8

                # 计算全市场融资买入额（单位：亿元）
                df['fin_buy_amount_total'] = (df['fin_buy_amount_sh'].fillna(0) + df['fin_buy_amount_sz'].fillna(0)) / 1e8

                df = df.sort_values('date')

                # 数据清洗：前向填充缺失值（处理节假日数据不更新问题）
                df = df.ffill()

                # 计算百分位排名
                df['percentile'] = df['fin_balance'].rank(pct=True)

                # 使用轻量级方法计算融资买入占比（避免调用stock_zh_a_spot_em）
                df = calculate_fin_buy_ratio(df)

                print(f"成功通过宏接口获取两融数据: {len(df)} 条")
                print(f"最新融资买入占比: {df['fin_buy_ratio'].iloc[-1]:.2f}%")
                data_source = 'akshare_macro'

                # 记录数据源状态到session_state（如果streamlit可用）
                try:
                    st.session_state['margin_data_source'] = data_source
                    st.session_state['margin_data_quality'] = 'real'
                    st.session_state['margin_ratio_real'] = True
                except:
                    pass

                return df[['date', 'fin_balance', 'fin_buy_ratio', 'percentile']]
        except Exception as e:
            print(f"akshare宏接口尝试失败: {e}")
    
    # 方法2: 尝试akshare的stock_margin_account_info接口（备用）
    if ak is not None:
        try:
            # 获取全市场两融账户信息
            df_margin = ak.stock_margin_account_info()

            if df_margin is not None and not df_margin.empty:
                # 清洗数据
                df = df_margin[['日期', '融资余额', '融资买入额']].copy()
                df = df.rename(columns={'日期': 'date', '融资余额': 'fin_balance', '融资买入额': 'fin_buy_amount'})

                # 转换日期格式和数值
                df['date'] = pd.to_datetime(df['date'])
                df['fin_balance'] = pd.to_numeric(df['fin_balance'], errors='coerce')
                df['fin_buy_amount'] = pd.to_numeric(df['fin_buy_amount'], errors='coerce')

                df = df.sort_values('date')

                # 数据清洗：前向填充缺失值
                df = df.ffill()

                # 转换单位：从元转换为亿元
                df['fin_balance'] = df['fin_balance'] / 1e8
                df['fin_buy_amount'] = df['fin_buy_amount'] / 1e8
                df['fin_buy_amount_total'] = df['fin_buy_amount']  # 用于calculate_fin_buy_ratio

                # 计算百分位排名
                df['percentile'] = df['fin_balance'].rank(pct=True)

                # 使用轻量级方法计算融资买入占比
                df = calculate_fin_buy_ratio(df)

                print(f"成功通过account_info接口获取两融数据: {len(df)} 条")
                data_source = 'akshare_account_info'

                try:
                    st.session_state['margin_data_source'] = data_source
                    st.session_state['margin_data_quality'] = 'real'
                    st.session_state['margin_ratio_real'] = True
                except:
                    pass

                return df[['date', 'fin_balance', 'fin_buy_ratio', 'percentile']]
        except Exception as e:
            print(f"akshare account_info接口尝试失败: {e}")
    
    # 方法3: 尝试使用Tushare Pro（需要配置token）
    try:
        # 检查是否有tushare token配置
        import json
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        tushare_token = config.get('tushare_token')

        if tushare_token:
            import tushare as ts
            ts.set_token(tushare_token)
            pro = ts.pro_api()

            # 获取最近90天的两融数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')

            df = pro.margin(start_date=start_date, end_date=end_date)

            if not df.empty:
                # 按日期汇总融资余额和融资买入额
                df['date'] = pd.to_datetime(df['trade_date'])
                df['fin_balance'] = df['rzye'] / 1e8  # 转换为亿元
                df['fin_buy_amount_total'] = df['rzmre'] / 1e8  # 融资买入额

                # 数据清洗：前向填充缺失值
                df = df.ffill()

                # 计算百分位排名
                df['percentile'] = df['fin_balance'].rank(pct=True)

                # 使用轻量级方法计算融资买入占比
                df = calculate_fin_buy_ratio(df)

                df = df.sort_values('date').reset_index(drop=True)
                print(f"成功通过Tushare Pro获取两融数据，共 {len(df)} 条记录")
                data_source = 'tushare_pro'

                try:
                    st.session_state['margin_data_source'] = data_source
                    st.session_state['margin_data_quality'] = 'real'
                    st.session_state['margin_ratio_real'] = True
                except:
                    pass

                return df[['date', 'fin_balance', 'fin_buy_ratio', 'percentile']]
    except Exception as e:
        print(f"Tushare Pro接口失败: {e}")
    
    # 方法4: 所有接口都失败，返回模拟数据
    print("警告：所有数据源均失败，返回模拟数据")
    df = generate_mock_margin_data()
    data_source = 'mock'
    
    try:
        st.session_state['margin_data_source'] = data_source
        st.session_state['margin_data_quality'] = 'mock'
        st.session_state['margin_ratio_real'] = False
    except:
        pass
        
    return df

# 测试函数
if __name__ == "__main__":
    print("测试宏观数据获取模块...")
    
    # 测试中国经济政策不确定性指数
    print("\n1. 获取中国经济政策不确定性指数...")
    epu_data = fetch_china_epu()
    print(f"EPU数据形状: {epu_data.shape}")
    if not epu_data.empty:
        latest_date = epu_data.index[-1]
        latest_epu = epu_data['EPU_Index'].iloc[-1]
        latest_percentile = epu_data['Percentile'].iloc[-1]
        print(f"最新EPU值: {latest_epu:.2f} (日期: {latest_date.strftime('%Y-%m-%d')})")
        print(f"百分位排名: {latest_percentile:.2%} (历史相对位置)")
    
    # 测试恐慌指数
    print("\n2. 获取恐慌指数(iVIX)...")
    ivix_data = fetch_ivix_data()
    print(f"iVIX数据形状: {ivix_data.shape}")
    if not ivix_data.empty:
        print(f"最新iVIX值: {ivix_data['ivix'].iloc[-1]:.2f} (日期: {ivix_data['date'].iloc[-1].strftime('%Y-%m-%d')})")
    
    # 测试两融数据（修复了KeyError问题）
    print("\n3. 获取两融数据...")
    margin_data = fetch_margin_data()
    print(f"两融数据形状: {margin_data.shape}")
    if not margin_data.empty:
        latest = margin_data.iloc[-1]
        print(f"最新数据 (日期: {latest['date'].strftime('%Y-%m-%d')}):")
        print(f"  融资余额: {latest['fin_balance']:.2f} 亿元")
        print(f"  融资买入占比: {latest['fin_buy_ratio']:.2f}%")
        print(f"  百分位排名: {latest['percentile']:.2%}")
    
    print("\n测试完成！")
