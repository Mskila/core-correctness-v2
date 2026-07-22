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
  "if (Test-Path -LiteralPath $pidFile) { $lines = @(Get-Content -LiteralPath $pidFile); $savedPid = 0; if ($lines.Count -gt 1 -and $lines[1]) { $expectedPython = $lines[1].Trim() }; if ($lines.Count -gt 0 -and [int]::TryParse($lines[0].Trim(), [ref]$savedPid)) { $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $savedPid) -ErrorAction SilentlyContinue; $pythonMatches = $proc -and $proc.ExecutablePath -and $proc.ExecutablePath.Equals($expectedPython, [StringComparison]::OrdinalIgnoreCase); $scriptMatches = $proc -and $proc.CommandLine -and ($proc.CommandLine.IndexOf($expectedScript, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or $proc.CommandLine -match '(?i)run_web\.py'); if ($pythonMatches -and $scriptMatches -and $proc.CommandLine -match '--port\s+8765') { $targets.Add($savedPid) } } };" ^
  "if ($targets.Count -eq 0) { $listeners = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue; foreach ($listener in $listeners) { $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $listener.OwningProcess) -ErrorAction SilentlyContinue; $pythonMatches = $proc -and $proc.ExecutablePath -and $proc.ExecutablePath.Equals($expectedPython, [StringComparison]::OrdinalIgnoreCase); $scriptMatches = $proc -and $proc.CommandLine -and ($proc.CommandLine.IndexOf($expectedScript, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or $proc.CommandLine -match '(?i)run_web\.py'); if ($pythonMatches -and $scriptMatches -and $proc.CommandLine -match '--port\s+8765') { $targets.Add([int]$listener.OwningProcess) } } };" ^
  "$targets = @($targets | Sort-Object -Unique);" ^
  "foreach ($targetPid in $targets) { & taskkill.exe /PID $targetPid /T /F 2>$null | Out-Null };" ^
  "Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue;" ^
  "$deadline = (Get-Date).AddSeconds(15); while ((Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 250 };" ^
  "$remaining = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue; if ($remaining -and $targets.Count -gt 0) { exit 2 }; if ($targets.Count -eq 0) { exit 3 }; exit 0"

set "STOP_CODE=%ERRORLEVEL%"
if "%STOP_CODE%"=="0" (
  echo [AlphaMaster] 项目已停止。
  exit /b 0
)
if "%STOP_CODE%"=="3" (
  echo [AlphaMaster] 未发现由本项目启动的 8765 服务，无需停止。
  exit /b 0
)
echo [AlphaMaster] 停止失败：项目进程仍可能在运行，请检查任务管理器。
exit /b %STOP_CODE%
