@echo off
echo ================================================================================
echo 启动 MedPortfolio 应用 - 使用虚拟环境
echo ================================================================================
echo.

cd /d "%~dp0"

echo [1/2] 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo 错误: 虚拟环境激活失败
    pause
    exit /b 1
)

echo.
echo [2/2] 启动Streamlit应用...
echo 应用将在浏览器中自动打开: http://localhost:8501
echo 按 Ctrl+C 停止应用
echo.

streamlit run app.py

pause
