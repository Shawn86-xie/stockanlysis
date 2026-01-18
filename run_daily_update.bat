@echo off
REM ============================================================
REM 医疗量化研报系统 - 市场数据自动更新脚本 (Windows)
REM ============================================================
REM 用途: 自动更新所有股票的市场数据
REM 使用方法: 双击运行，或在Windows任务计划程序中调用
REM ============================================================

echo.
echo ============================================================
echo    医疗量化研报系统 - 市场数据自动更新
echo ============================================================
echo.

REM 设置项目目录（自动获取脚本所在目录）
cd /d "%~dp0"
echo 当前目录: %CD%
echo.

REM 检查虚拟环境是否存在
if not exist "venv\Scripts\python.exe" (
    echo [错误] 虚拟环境不存在！
    echo 请先运行: python -m venv venv
    echo 然后安装依赖: venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [信息] 激活虚拟环境...
call venv\Scripts\activate

echo [信息] 开始更新市场数据...
echo.

REM 执行更新脚本
python daily_market_update.py

REM 捕获退出码
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% equ 0 (
    echo [成功] 市场数据更新完成！
) else if %EXIT_CODE% equ 2 (
    echo [警告] 市场数据更新部分失败，请查看日志文件
) else (
    echo [错误] 市场数据更新失败，请查看日志文件
)

echo.
echo 日志文件位置: logs\market_update_YYYYMMDD.log
echo.

REM 如果是双击运行，暂停以查看结果
if "%1"=="" pause

exit /b %EXIT_CODE%
