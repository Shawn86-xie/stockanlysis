# 使用官方Python 3.10 slim镜像作为基础镜像
FROM python:3.10-slim

# 设置环境变量，确保Python输出实时显示
ENV PYTHONUNBUFFERED=1

# 设置工作目录
WORKDIR /app

# 安装系统依赖，包括中文字体库
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    wget \
    # 中文字体支持
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fonts-wqy-microhei \
    fonts-wqy-zenhei \
    # 清理缓存以减少镜像大小
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用程序代码
COPY . .

# 创建数据目录用于挂载
RUN mkdir -p /app/data

# 设置环境变量，指定配置文件路径（可以在docker-compose中覆盖）
ENV CONFIG_PATH=/app/data/config.json

# 暴露Streamlit端口
EXPOSE 8501

# 健康检查，检查Streamlit服务是否正常
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 启动Streamlit应用
CMD ["streamlit", "run", "med_pro_system.py", "--server.port=8501", "--server.address=0.0.0.0"]