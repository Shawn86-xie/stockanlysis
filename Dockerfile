# 使用官方Python 3.10 slim镜像作为基础镜像
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    CONFIG_PATH=/app/data/config.json \
    # 默认认证配置（建议通过docker-compose覆盖）
    AUTH_USERNAME=admin \
    AUTH_PASSWORD=admin123 \
    AUTH_NAME=管理员 \
    AUTH_COOKIE_KEY=medportfolio_secret_key_2026 \
    AUTH_COOKIE_EXPIRY=7

# 设置工作目录
WORKDIR /app

# 安装系统依赖，包括中文字体库和时区支持
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    # 时区支持
    tzdata \
    # 中文字体支持
    fonts-noto-cjk \
    fonts-wqy-microhei \
    # 清理缓存以减少镜像大小
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 复制requirements.txt并安装依赖（利用Docker缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用核心代码
COPY app.py config.py auth.py ./
COPY ai_analyzer.py backtester.py data_fetcher.py ./
COPY data_integrity_checker.py database_manager.py ./
COPY font_utils.py macro_fetcher.py market_store.py ./
COPY math_engine.py storage_manager.py ./

# 复制pages和services目录
COPY pages/ ./pages/
COPY services/ ./services/

# 创建数据目录用于挂载
RUN mkdir -p /app/data /app/market_data /app/logs

# 暴露Streamlit端口
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 启动Streamlit多页面应用（app.py为主控台入口，pages/为子页面）
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
