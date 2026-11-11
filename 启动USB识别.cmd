@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 请先运行 scripts\setup_windows.ps1 安装识别环境。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "scripts\run_live.py" %*
if errorlevel 1 pause
