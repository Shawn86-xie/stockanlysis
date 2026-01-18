@echo off
echo ================================================================================
echo 系统功能评估 - 使用虚拟环境
echo ================================================================================
echo.

cd /d "%~dp0"

echo [1/3] 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo 错误: 虚拟环境激活失败
    pause
    exit /b 1
)

echo.
echo [2/3] 验证Python环境...
python --version
echo.

echo [3/3] 运行功能评估脚本...
echo.
python system_functionality_assessment.py

echo.
echo ================================================================================
echo 评估完成
echo ================================================================================
pause
