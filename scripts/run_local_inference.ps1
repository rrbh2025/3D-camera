param(
    [int]$Port = 18766,
    [string]$Device = "cuda:0",
    [int]$DetectorEvery = 1
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
python .\app\local_live_inference_server.py `
    --host 127.0.0.1 `
    --port $Port `
    --device $Device `
    --detector-every $DetectorEvery

