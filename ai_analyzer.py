from openai import OpenAI

def deepseek_analyze(api_key, news_title, stock_name):
    """
    使用 DeepSeek API 分析新闻
    Args:
        api_key: DeepSeek API Key
        news_title: 新闻标题
        stock_name: 股票名称
    Returns:
        stars: emoji 字符串
        nature: "利好"/"利空"/"中性"
        color: 颜色字符串 ("red"/"green"/"grey")
        reason: 分析理由
    """
    if not api_key:
        return "⭐", "未配置", "grey", "请在config中填入Key"
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = f"您是医学专家兼金融分析师。研判‘{stock_name}’的新闻：‘{news_title}’。按格式回答：评分:[1-5], 性质:[利好/利空/中性], 分析:[20字内]."
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content
        # 提取评分
        score = 1
        for s in ["5", "4", "3", "2", "1"]:
            if f"评分:{s}" in content or f"评分: {s}" in content:
                score = int(s)
                break
        # 提取性质
        nature = "中性"
        if "利好" in content:
            nature = "利好"
        elif "利空" in content:
            nature = "利空"
        # 提取分析
        reason = content.split("分析:")[-1].strip() if "分析:" in content else "完成解析"
        color = "red" if nature == "利好" else "green" if nature == "利空" else "grey"
        stars = ("❤️" if nature == "利好" else "💚" if nature == "利空" else "⭐") * score
        return stars, nature, color, reason
    except Exception as e:
        print(f"DeepSeek 分析失败: {e}")
        return "⚠️", "异常", "grey", "连接失败"