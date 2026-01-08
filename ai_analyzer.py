from openai import OpenAI

def deepseek_analyze(api_key, news_title, stock_name, publish_date=None):
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
            nature: "利好"/"利空"/"中性"
            color: 颜色字符串 ("red"/"green"/"grey")
            reason: 分析理由
            summary: 内容摘要（50字内）
            date: 发布日期（如果提供）
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
        
        # 提取评分
        for s in ["5", "4", "3", "2", "1"]:
            if f"评分:{s}" in content or f"评分: {s}" in content:
                score = int(s)
                break
        
        # 提取性质
        if "利好" in content:
            nature = "利好"
        elif "利空" in content:
            nature = "利空"
        
        # 提取分析
        if "分析:" in content:
            # 找到分析部分，直到下一个标签或结尾
            reason_start = content.find("分析:")
            reason_text = content[reason_start:]
            # 检查是否有"摘要:"标签
            if "摘要:" in reason_text:
                reason = reason_text.split("摘要:")[0].replace("分析:", "").strip()
            else:
                reason = reason_text.replace("分析:", "").strip()
        
        # 提取摘要
        if "摘要:" in content:
            summary_start = content.find("摘要:")
            summary = content[summary_start:].replace("摘要:", "").strip()
            # 清理可能的多余标签
            summary = summary.split("\n")[0].strip()
        
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
            "reason": "连接失败",
            "summary": "无法生成摘要",
            "date": publish_date or "未知日期"
        }
