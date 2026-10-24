# BestShots: установка для нового пользователя (Windows).
# Проверяет Python 3.11+, создаёт .venv, ставит зависимости и собирает BestShots.exe (PyInstaller).
# Запуск: install.bat (двойной щелчок) или  powershell -ExecutionPolicy Bypass -File install.ps1
# Параметры: -DryRun (только показать шаги), -SkipDeps (не ставить зависимости), -SkipExe (не собирать exe).
param([switch]$DryRun, [switch]$SkipDeps, [switch]$SkipExe)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Say($m) { Write-Host $m }
function Step($m) { Write-Host ""; Write-Host "== $m ==" -ForegroundColor Cyan }

function Run([string]$exe, [string[]]$arguments) {
    if ($DryRun) { Say "[dry-run] $exe $($arguments -join ' ')"; return }
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Команда завершилась с ошибкой ($LASTEXITCODE): $exe $($arguments -join ' ')" }
}

function Find-Python {
    $candidates = @(
        @{ Exe = "py"; Args = @("-3.13") }, @{ Exe = "py"; Args = @("-3.12") }, @{ Exe = "py"; Args = @("-3.11") },
        @{ Exe = "python"; Args = @() }, @{ Exe = "python3"; Args = @() }
    )
    foreach ($c in $candidates) {
        try {
            $v = & $c.Exe @($c.Args) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $v) {
                $parts = "$v".Trim().Split(".")
                if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 11) {
                    return @{ Exe = $c.Exe; Args = $c.Args; Version = "$v".Trim() }
                }
            }
        } catch { }
    }
    return $null
}

Step "1/4  Проверка Python (нужен 3.11 или новее)"
$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.11+ не найден. Установите его с https://www.python.org/downloads/ (отметьте «Add python.exe to PATH») и запустите install.bat снова." -ForegroundColor Red
    exit 1
}
Say "Найден Python $($py.Version): $($py.Exe) $($py.Args -join ' ')"
if ([int]($py.Version.Split(".")[1]) -ge 14) {
    Write-Host "Предупреждение: для Python $($py.Version) готовые сборки PyTorch могут отсутствовать; рекомендуется Python 3.11–3.13." -ForegroundColor Yellow
}

Step "2/4  Окружение .venv"
if (Test-Path ".venv\Scripts\python.exe") { Say ".venv уже есть — используем его." }
else { Run $py.Exe (@($py.Args) + @("-m", "venv", ".venv")) }
$venvPy = ".venv\Scripts\python.exe"

Step "3/4  Зависимости (requirements.txt; PyTorch и модели — несколько ГБ, понадобится интернет)"
if ($SkipDeps) { Say "Пропущено (-SkipDeps)." }
else {
    Run $venvPy @("-m", "pip", "install", "--upgrade", "pip")
    Run $venvPy @("-m", "pip", "install", "-r", "requirements.txt")
}

Step "4/4  Сборка BestShots.exe (PyInstaller)"
if ($SkipExe) { Say "Пропущено (-SkipExe). Запуск без exe:  .venv\Scripts\python review.py" }
else {
    Run $venvPy @("-m", "pip", "install", "pyinstaller")
    Run $venvPy @("-m", "PyInstaller", "--onefile", "--noconsole", "--name", "BestShots", "--icon", "icon.ico",
                  "--distpath", "dist", "--workpath", "build", "--noconfirm", "launcher.py")
    if (-not $DryRun) {
        Copy-Item "dist\BestShots.exe" "BestShots.exe" -Force
        Remove-Item "build", "dist", "BestShots.spec" -Recurse -Force -ErrorAction SilentlyContinue
    } else { Say "[dry-run] копирование dist\BestShots.exe в корень и очистка build, dist" }
}

Write-Host ""
Write-Host "Готово." -ForegroundColor Green
Say "Запуск: двойной щелчок по BestShots.exe (или .venv\Scripts\python review.py)."
Say "При первом анализе приложение само скачает модели (~3 ГБ, нужен интернет один раз)."
