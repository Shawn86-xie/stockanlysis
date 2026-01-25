from typing import Optional, Dict, Any, List, Tuple
import json
import os

# 清除所有代理环境变量（必须在导入 openai 之前）
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)

# Patch httpx 以禁用代理（OpenAI SDK 使用 httpx）
try:
    import httpx

    # 保存原始方法
    _original_httpx_client_init = httpx.Client.__init__
    _original_httpx_async_client_init = httpx.AsyncClient.__init__

    def _patched_httpx_client_init(self, *args, **kwargs):
        """移除 httpx.Client 的 proxies/proxy 参数并禁用环境变量"""
        kwargs.pop('proxies', None)  # 移除可能传入的 proxies
        kwargs.pop('proxy', None)     # 移除可能传入的 proxy
        # 设置 trust_env=False 以禁用从环境变量读取代理
        kwargs['trust_env'] = False
        return _original_httpx_client_init(self, *args, **kwargs)

    def _patched_httpx_async_client_init(self, *args, **kwargs):
        """移除 httpx.AsyncClient 的 proxies/proxy 参数并禁用环境变量"""
        kwargs.pop('proxies', None)
        kwargs.pop('proxy', None)
        kwargs['trust_env'] = False
        return _original_httpx_async_client_init(self, *args, **kwargs)

    # 应用 patches
    httpx.Client.__init__ = _patched_httpx_client_init
    httpx.AsyncClient.__init__ = _patched_httpx_async_client_init
except ImportError:
    pass  # httpx 未安装，跳过

from openai import OpenAI

def deepseek_analyze(api_key: str, news_title: str, stock_name: str, publish_date: Optional[str] = None) -> Dict[str, Any]:
    """
    使用 DeepSeek API 分析新闻并生成摘要
    
    Args:
        api_key: DeepSeek API Key
        news_title: 新闻标题
        stock_name: 股票名称
        publish_date: 发布日期字符串（可选）
    
    Returns:
        dict: 包含以下键的字典：
            stars: emoji 字符串
            nature: "利好"/"利空"/"中性"/"未配置"/"异常"
            color: 颜色字符串 ("red"/"green"/"grey")
            reason: 分析理由或错误信息
            summary: 内容摘要（50字内）或错误提示
            date: 发布日期或"未知日期"
    """
    if not api_key:
        return {
            "stars": "⭐",
            "nature": "未配置",
            "color": "grey",
            "reason": "请在config中填入Key",
            "summary": "无法生成摘要",
            "date": publish_date or "未知日期"
        }
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    # 构建更详细的prompt，要求生成摘要
    date_info = f"，发布日期：{publish_date}" if publish_date else ""
    prompt = f"""您是医学专家兼金融分析师。请分析‘{stock_name}’的新闻：
标题：{news_title}{date_info}

请按以下格式回答：
评分:[1-5]
性质:[利好/利空/中性]
分析:[20字内，说明对股价的潜在影响]
摘要:[50字内，概括新闻核心内容]"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content
        
        # 初始化默认值
        score = 1
        nature = "中性"
        reason = "完成解析"
        summary = "无摘要"
        
        # 更稳健地提取评分 - 处理各种格式：评分:5、评分: 5、评分：5、评分： 5等
        import re
        score_patterns = [
            r'评分[:：]\s*([1-5])',           # 匹配 "评分:5" 或 "评分：5"
            r'评分\s*[:：]\s*([1-5])',       # 匹配 "评分 :5" 或 "评分 ：5"
            r'([1-5])\s*分',                 # 匹配 "5分"
            r'评分\s*([1-5])'                # 匹配 "评分5"
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    score = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # 更稳健地提取性质 - 使用正则表达式匹配
        nature_patterns = [
            r'性质[:：]\s*(利好|利空|中性)',   # 匹配 "性质:利好" 或 "性质：利好"
            r'(利好|利空|中性)\s*性质',       # 匹配 "利好性质"
            r'(利好|利空|中性)'              # 直接匹配关键词
        ]
        
        for pattern in nature_patterns:
            match = re.search(pattern, content)
            if match:
                nature = match.group(1)
                break
        
        # 更稳健地提取分析 - 使用正则表达式
        reason_patterns = [
            r'分析[:：]\s*([^\n]+?)(?=\n*(?:摘要|评分|性质|$))',  # 匹配分析内容直到下一个标签或结尾
            r'分析[:：]\s*([^\n]+)'                              # 匹配分析内容直到换行
        ]
        
        for pattern in reason_patterns:
            match = re.search(pattern, content)
            if match:
                reason = match.group(1).strip()
                break
        
        # 更稳健地提取摘要 - 使用正则表达式
        summary_patterns = [
            r'摘要[:：]\s*([^\n]+?)(?=\n*(?:评分|性质|分析|$))',  # 匹配摘要内容直到下一个标签或结尾
            r'摘要[:：]\s*([^\n]+)'                              # 匹配摘要内容直到换行
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, content)
            if match:
                summary = match.group(1).strip()
                break
        
        color = "red" if nature == "利好" else "green" if nature == "利空" else "grey"
        stars = ("❤️" if nature == "利好" else "💚" if nature == "利空" else "⭐") * score
        
        return {
            "stars": stars,
            "nature": nature,
            "color": color,
            "reason": reason,
            "summary": summary,
            "date": publish_date or "未知日期"
        }
        
    except Exception as e:
        print(f"DeepSeek 分析失败: {e}")
        return {
            "stars": "⚠️",
            "nature": "异常",
            "color": "grey",
            "reason": f"API调用失败: {str(e)}",
            "summary": "无法生成摘要",
            "date": publish_date or "未知日期"
        }

def analyze_macro_environment(api_key: str, macro_indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用 DeepSeek API 分析宏观环境并生成投资建议
    
    Args:
        api_key: DeepSeek API Key
        macro_indicators: 包含宏观指标的字典，包括：
            - epu_current_value: 当前EPU值
            - epu_current_percentile: EPU百分位
            - ivix_current: 当前iVIX值
            - ivix_percentile: iVIX百分位
            - latest_fin_balance: 最新融资余额（亿元）
            - margin_ratio: 融资余额环比变化（%）
            - margin_percentile: 融资余额百分位
            - diagnosis: 当前诊断结论
    
    Returns:
        dict: 包含以下键的字典：
            - overall_assessment: 总体评估 (良好/谨慎/危险/不确定)
            - confidence_score: 置信度分数 (0-100)
            - key_insights: 关键洞察列表
            - investment_recommendations: 投资建议列表
            - risk_warnings: 风险警告列表
            - technical_analysis: 技术分析摘要
            - market_timing: 市场时机建议
    """
    if not api_key:
        return {
            "overall_assessment": "数据不足",
            "confidence_score": 0,
            "key_insights": ["请配置DeepSeek API Key以获取AI分析"],
            "investment_recommendations": ["请先配置API Key"],
            "risk_warnings": ["无AI分析可用"],
            "technical_analysis": "无法生成技术分析",
            "market_timing": "无法评估市场时机"
        }
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    # 构建详细的宏观分析prompt
    prompt = f"""作为顶级宏观经济学家兼量化投资专家，请分析以下中国市场宏观指标：

**经济政策不确定性 (EPU):**
- 当前值: {macro_indicators.get('epu_current_value', 0):.1f}
- 历史百分位: {macro_indicators.get('epu_current_percentile', 0.5):.1%} ({'高位' if macro_indicators.get('epu_current_percentile', 0.5) > 0.7 else '中位' if macro_indicators.get('epu_current_percentile', 0.5) > 0.3 else '低位'})

**恐慌指数 (iVIX):**
- 当前值: {macro_indicators.get('ivix_current', 20):.2f}
- 历史百分位: {macro_indicators.get('ivix_percentile', 0.5):.1%}
- 状态: {'高压' if macro_indicators.get('ivix_current', 20) > 30 else '中压' if macro_indicators.get('ivix_current', 20) > 20 else '低压'}

**两融数据:**
- 融资余额: {macro_indicators.get('latest_fin_balance', 0):,.0f} 亿元
- 环比变化: {macro_indicators.get('margin_ratio', 0):+.2f}%
- 历史百分位: {macro_indicators.get('margin_percentile', 0.5):.1%}

**系统诊断:**
{macro_indicators.get('diagnosis', '无诊断信息')}

请提供全面的宏观分析，包括：

1. **总体市场环境评估** (使用: 良好/谨慎/危险/不确定 四个等级)
2. **置信度分数** (0-100分，基于数据质量和一致性)
3. **3个关键市场洞察** (每条不超过30字)
4. **3条具体的投资建议** (针对当前环境)
5. **2个主要风险警告** (需要警惕的风险)
6. **技术面分析摘要** (100字内，基于指标间的关系)
7. **市场时机建议** (现在应该: 积极加仓/适度建仓/保持观望/减仓避险)

请以JSON格式返回，结构如下：
{{
    "overall_assessment": "评估等级",
    "confidence_score": 85,
    "key_insights": ["洞察1", "洞察2", "洞察3"],
    "investment_recommendations": ["建议1", "建议2", "建议3"],
    "risk_warnings": ["风险1", "风险2"],
    "technical_analysis": "技术分析摘要...",
    "market_timing": "时机建议..."
}}"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严谨的宏观经济学家和量化投资专家。你的分析基于数据和事实，避免主观臆测。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # 解析JSON响应
        try:
            analysis_result = json.loads(content)
            
            # 验证必要字段
            required_fields = ["overall_assessment", "confidence_score", "key_insights", 
                             "investment_recommendations", "risk_warnings", "technical_analysis", "market_timing"]
            
            for field in required_fields:
                if field not in analysis_result:
                    raise KeyError(f"Missing required field: {field}")
            
            return analysis_result
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}, 原始内容: {content}")
            # 返回默认分析
            return {
                "overall_assessment": "分析失败",
                "confidence_score": 0,
                "key_insights": ["AI分析响应格式错误"],
                "investment_recommendations": ["请检查API响应"],
                "risk_warnings": ["无法解析AI分析"],
                "technical_analysis": "解析失败",
                "market_timing": "无法评估"
            }
            
    except Exception as e:
        print(f"宏观环境分析失败: {e}")
        return {
            "overall_assessment": "分析异常",
            "confidence_score": 0,
            "key_insights": [f"API调用失败: {str(e)}"],
            "investment_recommendations": ["请稍后重试"],
            "risk_warnings": ["AI分析不可用"],
            "technical_analysis": "分析失败",
            "market_timing": "无法评估"
        }
