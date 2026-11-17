$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$binDir = Join-Path $PSScriptRoot "bin"
$modelsDir = Join-Path $PSScriptRoot "models"
$archive = Join-Path $PSScriptRoot "llama-b12477-bin-win-cuda-12.4-x64.zip"
$items = @(
    @{
        Url = "https://github.com/ggml-org/llama.cpp/releases/download/b12477/llama-b12477-bin-win-cuda-12.4-x64.zip"
        Path = $archive
    },
    @{
        Url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        Path = (Join-Path $modelsDir "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    },
    @{
        Url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf"
        Path = (Join-Path $modelsDir "qwen2.5-1.5b-instruct-q8_0.gguf")
    }
)

New-Item -ItemType Directory -Force $binDir, $modelsDir | Out-Null
foreach ($item in $items) {
    if (-not (Test-Path $item.Path)) {
        Write-Host "Downloading $([IO.Path]::GetFileName($item.Path))"
        & curl.exe -L --fail --retry 3 --output $item.Path $item.Url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $($item.Url)"
        }
    }
}

if (-not (Get-ChildItem $binDir -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue)) {
    Expand-Archive -Path $archive -DestinationPath $binDir -Force
}

$server = Get-ChildItem $binDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $server) {
    throw "llama-server.exe was not present in the downloaded archive."
}
Write-Host "llama-server: $($server.FullName)"
Get-ChildItem $modelsDir | Select-Object Name, Length
