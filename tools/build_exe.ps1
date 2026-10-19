$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$buildRoot = Join-Path $projectRoot 'output\build'
$releaseRoot = Join-Path $projectRoot 'output\executable'

if (-not (Test-Path -LiteralPath $buildPython)) {
    throw 'Create .venv and install requirements-build.txt first.'
}

New-Item -ItemType Directory -Path $buildRoot, $releaseRoot -Force | Out-Null
$previousCache = $env:PYINSTALLER_CONFIG_DIR
try {
    $env:PYINSTALLER_CONFIG_DIR = Join-Path $buildRoot 'pyinstaller-cache'
    $buildArguments = @(
        '-m', 'PyInstaller', '--noconfirm', '--onefile', '--windowed',
        '--name', 'PinguBall', '--icon', 'NONE',
        '--distpath', $releaseRoot,
        '--workpath', (Join-Path $buildRoot 'work'),
        '--specpath', $buildRoot,
        '--paths', $projectRoot,
        '--add-data', ((Join-Path $projectRoot 'assets\antarctica.png') + ';assets'),
        '--add-data', ((Join-Path $projectRoot 'assets\pingu.png') + ';assets'),
        (Join-Path $projectRoot 'main.py')
    )
    & $buildPython @buildArguments
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed: $LASTEXITCODE" }
    Write-Output (Join-Path $releaseRoot 'PinguBall.exe')
}
finally {
    $env:PYINSTALLER_CONFIG_DIR = $previousCache
}
