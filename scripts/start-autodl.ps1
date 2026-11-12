# 使用项目内便携 Node.js 启动监测台，不修改全局 PATH。
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolRoot = Join-Path $projectRoot '.local-tools/autodl-monitor'
$nodeRoot = Join-Path $projectRoot '.local-tools/node-runtime'
$runtimes = @(Get-ChildItem -LiteralPath $nodeRoot -Directory -Filter 'node-v22.*-win-x64')
if ($runtimes.Count -ne 1) { throw '没有找到唯一的 Node.js 22 便携运行时。' }
$node = Join-Path $runtimes[0].FullName 'node.exe'
if (-not (Test-Path -LiteralPath (Join-Path $toolRoot 'node_modules/playwright/package.json'))) {
    throw '监测工具尚未安装依赖。'
}

$runtimeDir = Join-Path $toolRoot '.runtime'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$previousPath = $env:PATH
$previousPort = $env:PORT
$previousRuntime = $env:AUTODL_RUNTIME_DIR
try {
    $env:PATH = $runtimes[0].FullName + ';' + $env:PATH
    $env:PORT = '8787'
    $env:AUTODL_RUNTIME_DIR = $runtimeDir
    $serverScript = Join-Path $toolRoot 'src/server.mjs'
    Start-Process -FilePath $node -ArgumentList ('"' + $serverScript + '"') `
        -WorkingDirectory $toolRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $runtimeDir 'server.stdout.log') `
        -RedirectStandardError (Join-Path $runtimeDir 'server.stderr.log') | Out-Null
    Write-Output '控制台：http://127.0.0.1:8787 （启动可能需要数秒）'
    Write-Output '停止后台：运行 scripts/stop-autodl.ps1；停止后台不会关闭云端实例。'
}
finally {
    $env:PATH = $previousPath
    $env:PORT = $previousPort
    $env:AUTODL_RUNTIME_DIR = $previousRuntime
}
