# Запускать из корня локального Git-клона в PowerShell.
$ErrorActionPreference = "Stop"
$venvPython = ".\.venv\Scripts\python.exe"

if (-not (Test-Path ".git")) {
    throw "Откройте корень Git-клона проекта (должна существовать папка .git)."
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Не найден Python Launcher 'py'. Установите Python 3.11."
}

py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Не удалось создать venv." }
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Не удалось обновить pip." }
& $venvPython -m pip install -r requirements-ci.txt
if ($LASTEXITCODE -ne 0) { throw "Не удалось установить зависимости CI." }
& $venvPython -m pre_commit install
if ($LASTEXITCODE -ne 0) { throw "Не удалось установить Git hooks." }
& $venvPython -m pre_commit run --all-files
if ($LASTEXITCODE -ne 0) { throw "pre-commit обнаружил ошибки. См. вывод выше." }
& $venvPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Тесты не прошли. См. вывод выше." }
Write-Host "Готово: venv, pip, pre-commit и тесты настроены." -ForegroundColor Green
