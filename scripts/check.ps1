# 在项目根目录执行；任一步骤失败即返回非零退出码。
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw '请先按 README 创建 .venv 并安装 .[dev]。'
}

$testTemp = Join-Path $projectRoot 'outputs/tmp'
New-Item -ItemType Directory -Force -Path $testTemp | Out-Null
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
Push-Location $projectRoot
try {
    # 将测试临时文件限制在工作区，兼容受限环境。
    $env:TEMP = $testTemp
    $env:TMP = $testTemp
    & $python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'unittest 失败' }
    & $python -m ruff check .
    if ($LASTEXITCODE -ne 0) { throw 'Ruff lint 失败' }
    & $python -m ruff format --check .
    if ($LASTEXITCODE -ne 0) { throw 'Ruff 格式检查失败' }
    & $python -m mypy
    if ($LASTEXITCODE -ne 0) { throw 'mypy 失败' }
}
finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    Pop-Location
}
