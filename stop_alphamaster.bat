@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "ALPHAMASTER_ROOT=%~dp0"

echo [AlphaMaster] 正在停止当前项目的 Web 服务...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root = [IO.Path]::GetFullPath($env:ALPHAMASTER_ROOT);" ^
  "$pidFile = Join-Path $root '.alphamaster_web.pid';" ^
  "$expectedScript = Join-Path $root 'run_web.py';" ^
  "$targets = [Collections.Generic.List[int]]::new();" ^
  "$expectedPython = Join-Path $root '.venv\Scripts\python.exe';" ^
  "if (Test-Path -LiteralPath $pidFile) { $lines = @(Get-Content -LiteralPath $pidFile); $savedPid = 0; $savedListenerPid = 0; if ($lines.Count -gt 1 -and $lines[1]) { $expectedPython = $lines[1].Trim() }; if ($lines.Count -gt 2) { [void][int]::TryParse($lines[2].Trim(), [ref]$savedListenerPid) }; if ($savedListenerPid -gt 0) { $listenerProc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $savedListenerPid) -ErrorAction SilentlyContinue; if ($listenerProc -and $listenerProc.CommandLine -match '(?i)run_web\.py' -and $listenerProc.CommandLine -match '--port\s+8765') { $targets.Add($savedListenerPid) } }; if ($lines.Count -gt 0 -and [int]::TryParse($lines[0].Trim(), [ref]$savedPid)) { $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $savedPid) -ErrorAction SilentlyContinue; $pythonMatches = $proc -and $proc.ExecutablePath -and $proc.ExecutablePath.Equals($expectedPython, [StringComparison]::OrdinalIgnoreCase); $scriptMatches = $proc -and $proc.CommandLine -and ($proc.CommandLine.IndexOf($expectedScript, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or $proc.CommandLine -match '(?i)run_web\.py'); if ($pythonMatches -and $scriptMatches -and $proc.CommandLine -match '--port\s+8765' -and -not $targets.Contains($savedPid)) { $targets.Add($savedPid) } } };" ^
  "if ($targets.Count -eq 0) { $listenerPids = [Collections.Generic.List[int]]::new(); foreach ($netLine in @(& netstat.exe -ano -p TCP)) { if ($netLine -match '^\s*TCP\s+\S+:8765\s+\S+\s+LISTENING\s+(\d+)\s*$') { $listenerPids.Add([int]$Matches[1]) } }; foreach ($listenerPid in $listenerPids) { $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $listenerPid) -ErrorAction SilentlyContinue; $pythonMatches = $proc -and $proc.ExecutablePath -and $proc.ExecutablePath.Equals($expectedPython, [StringComparison]::OrdinalIgnoreCase); $scriptMatches = $proc -and $proc.CommandLine -and ($proc.CommandLine.IndexOf($expectedScript, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or $proc.CommandLine -match '(?i)run_web\.py'); if ($pythonMatches -and $scriptMatches -and $proc.CommandLine -match '--port\s+8765' -and -not $targets.Contains([int]$listenerPid)) { $targets.Add([int]$listenerPid) } } };" ^
  "foreach ($targetPid in $targets) { Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue };" ^
  "Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue;" ^
  "$deadline = (Get-Date).AddSeconds(15); do { $listening = $false; foreach ($netLine in @(& netstat.exe -ano -p TCP)) { if ($netLine -match '^\s*TCP\s+\S+:8765\s+\S+\s+LISTENING\s+\d+\s*$') { $listening = $true; break } }; if (-not $listening) { break }; Start-Sleep -Milliseconds 250 } while ((Get-Date) -lt $deadline);" ^
  "$remaining = $false; foreach ($netLine in @(& netstat.exe -ano -p TCP)) { if ($netLine -match '^\s*TCP\s+\S+:8765\s+\S+\s+LISTENING\s+\d+\s*$') { $remaining = $true; break } }; if ($remaining -and $targets.Count -gt 0) { exit 2 }; if ($targets.Count -eq 0) { exit 3 }; exit 0"

set "STOP_CODE=%ERRORLEVEL%"
if "%STOP_CODE%"=="0" (
  echo [AlphaMaster] 项目已停止。
  call :wait_before_close
  exit /b 0
)
if "%STOP_CODE%"=="3" (
  echo [AlphaMaster] 未发现由本项目启动的 8765 服务，无需停止。
  call :wait_before_close
  exit /b 0
)
echo [AlphaMaster] 停止失败：项目进程仍可能在运行，请检查任务管理器。
call :wait_before_close
exit /b %STOP_CODE%

:wait_before_close
if not defined ALPHAMASTER_NO_PAUSE (
  echo.
  echo 按任意键关闭此窗口...
  pause >nul
)
exit /b 0
