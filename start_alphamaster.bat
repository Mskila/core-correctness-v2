@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "ALPHAMASTER_ROOT=%~dp0"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PID_FILE=%~dp0.alphamaster_web.pid"
if defined ALPHAMASTER_PYTHON set "VENV_PY=%ALPHAMASTER_PYTHON%"

echo [AlphaMaster] 项目目录: %~dp0

rem 已有服务直接打开页面，避免重复启动。
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 2; if ($h.status -eq 'ok') { exit 0 } } catch {}; exit 1"
if not errorlevel 1 (
  echo [AlphaMaster] 服务已经运行，正在打开页面...
  if not defined ALPHAMASTER_NO_BROWSER start "" "http://127.0.0.1:8765/"
  call :wait_before_close
  exit /b 0
)

rem 首次复制到新电脑时自动创建本项目专用虚拟环境。
if exist "%VENV_PY%" goto environment_ready

echo [AlphaMaster] 首次运行：正在创建 Python 虚拟环境...
where py >nul 2>&1
if errorlevel 1 goto try_system_python

py -3.12 -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>&1
if not errorlevel 1 (
  py -3.12 -m venv "%~dp0.venv"
  if not errorlevel 1 goto environment_ready
)
py -3.11 -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>&1
if not errorlevel 1 (
  py -3.11 -m venv "%~dp0.venv"
  if not errorlevel 1 goto environment_ready
)
py -3.10 -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>&1
if not errorlevel 1 (
  py -3.10 -m venv "%~dp0.venv"
  if not errorlevel 1 goto environment_ready
)

:try_system_python
for /f "delims=" %%I in ('where python 2^>nul') do if not defined SYSTEM_PY set "SYSTEM_PY=%%I"
if not defined SYSTEM_PY goto no_python
"%SYSTEM_PY%" -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>&1
if errorlevel 1 goto no_python
"%SYSTEM_PY%" -m venv "%~dp0.venv"
if errorlevel 1 goto venv_failed

:environment_ready
if not exist "%VENV_PY%" goto venv_failed

rem 仅在依赖缺失时安装；正常启动不会重复联网或安装。
"%VENV_PY%" -c "import fastapi, multipart, uvicorn, torch, numpy, pandas, pyarrow, matplotlib, loguru" >nul 2>&1
if errorlevel 1 (
  echo [AlphaMaster] 首次运行：正在安装核心依赖，可能需要几分钟...
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 goto dependency_failed
  "%VENV_PY%" -m pip install -r "%~dp0requirements.txt" -c "%~dp0constraints-core.txt"
  if errorlevel 1 goto dependency_failed
)

rem 不占用其他程序的 8765 端口。
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$occupied = $false; foreach ($netLine in @(& netstat.exe -ano -p TCP)) { if ($netLine -match '^\s*TCP\s+\S+:8765\s+\S+\s+LISTENING\s+\d+\s*$') { $occupied = $true; break } }; if ($occupied) { exit 1 }; exit 0"
if errorlevel 1 goto port_in_use

if not exist "%~dp0logs" mkdir "%~dp0logs"
if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1

echo [AlphaMaster] 正在后台启动 Web 服务...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root = [IO.Path]::GetFullPath($env:ALPHAMASTER_ROOT);" ^
  "$python = $env:ALPHAMASTER_PYTHON; if ([string]::IsNullOrWhiteSpace($python)) { $python = Join-Path $root '.venv\Scripts\python.exe' };" ^
  "$script = Join-Path $root 'run_web.py';" ^
  "$pidFile = Join-Path $root '.alphamaster_web.pid';" ^
  "$stdout = Join-Path $root 'logs\web_8765.out.log';" ^
  "$stderr = Join-Path $root 'logs\web_8765.err.log';" ^
  "$p = Start-Process -FilePath $python -ArgumentList @('run_web.py', '--port', '8765') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru;" ^
  "Set-Content -LiteralPath $pidFile -Value @($p.Id, $python) -Encoding Ascii;" ^
  "$ready = $false; for ($i = 0; $i -lt 60; $i++) { try { $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 2; if ($h.status -eq 'ok') { $ready = $true; break } } catch {}; Start-Sleep -Seconds 1 };" ^
  "if (-not $ready) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue; exit 1 };" ^
  "$listenerPid = 0; foreach ($netLine in @(& netstat.exe -ano -p TCP)) { if ($netLine -match '^\s*TCP\s+\S+:8765\s+\S+\s+LISTENING\s+(\d+)\s*$') { $listenerPid = [int]$Matches[1]; break } };" ^
  "Set-Content -LiteralPath $pidFile -Value @($p.Id, $python, $listenerPid) -Encoding Ascii"
if errorlevel 1 goto start_failed

echo [AlphaMaster] 启动成功: http://127.0.0.1:8765/
if not defined ALPHAMASTER_NO_BROWSER start "" "http://127.0.0.1:8765/"
call :wait_before_close
exit /b 0

:no_python
echo [AlphaMaster] 启动失败：未找到 Python 3.10、3.11 或 3.12。
echo 请先从 https://www.python.org/downloads/ 安装 64 位 Python 后重试。
call :wait_before_close
exit /b 10

:venv_failed
echo [AlphaMaster] 启动失败：无法创建 .venv 虚拟环境。
call :wait_before_close
exit /b 11

:dependency_failed
echo [AlphaMaster] 启动失败：依赖安装未完成，请检查网络和 logs 目录。
call :wait_before_close
exit /b 12

:port_in_use
echo [AlphaMaster] 启动失败：端口 8765 已被其他程序占用。
echo 请先关闭占用该端口的程序，再重新运行本文件。
call :wait_before_close
exit /b 13

:start_failed
echo [AlphaMaster] 启动失败：服务未能在 60 秒内就绪。
echo 请查看 logs\web_8765.err.log 和 logs\web_8765.out.log。
if exist "%~dp0logs\web_8765.err.log" (
  echo.
  echo ==================== 错误日志末尾 ====================
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%~dp0logs\web_8765.err.log' -Tail 40"
  echo ======================================================
)
call :wait_before_close
exit /b 14

:wait_before_close
if not defined ALPHAMASTER_NO_PAUSE (
  echo.
  echo 按任意键关闭此窗口...
  pause >nul
)
exit /b 0
