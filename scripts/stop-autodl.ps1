# 调用上游正常停止接口，保留任务状态与登录资料。
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolRoot = Join-Path $projectRoot '.local-tools/autodl-monitor'
$runtimes = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot '.local-tools/node-runtime') `
    -Directory -Filter 'node-v22.*-win-x64')
if ($runtimes.Count -ne 1) { throw '没有找到唯一的 Node.js 22 便携运行时。' }
$previousPort = $env:PORT
$previousRuntime = $env:AUTODL_RUNTIME_DIR
Push-Location $toolRoot
try {
    $env:PORT = '8787'
    $env:AUTODL_RUNTIME_DIR = Join-Path $toolRoot '.runtime'
    & (Join-Path $runtimes[0].FullName 'node.exe') 'scripts/stop-server.mjs'
    if ($LASTEXITCODE -ne 0) { throw '停止失败，请查看监测台日志。' }
}
finally {
    $env:PORT = $previousPort
    $env:AUTODL_RUNTIME_DIR = $previousRuntime
    Pop-Location
}
