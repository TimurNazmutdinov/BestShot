@echo off
rem BestShots: установка (вызывает install.ps1). Двойной щелчок или: install.bat -DryRun
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 (
  echo.
  echo Установка не завершилась. Смотрите сообщения выше.
  pause
  exit /b 1
)
echo.
pause
