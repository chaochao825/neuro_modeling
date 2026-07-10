param(
    [string]$PythonVersion = "3.11.9"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonHome = Join-Path $ProjectRoot ".python311"
$Venv = Join-Path $ProjectRoot ".venv"
$Downloads = Join-Path $ProjectRoot "downloads"
$Installer = Join-Path $Downloads "python-$PythonVersion-amd64.exe"
$Python = Join-Path $PythonHome "python.exe"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $Downloads | Out-Null

if (-not (Test-Path -LiteralPath $Python)) {
    $Url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
    Write-Host "Downloading Python $PythonVersion from python.org"
    Invoke-WebRequest -Uri $Url -OutFile $Installer
    $arguments = @(
        "/quiet",
        "InstallAllUsers=0",
        "TargetDir=$PythonHome",
        "Include_pip=1",
        "Include_launcher=0",
        "Include_test=0",
        "AssociateFiles=0",
        "Shortcuts=0",
        "PrependPath=0"
    )
    $process = Start-Process -FilePath $Installer -ArgumentList $arguments -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Python installer exited with code $($process.ExitCode)"
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $Python -m venv $Venv
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install --editable "$ProjectRoot[dev]"
& $VenvPython -c "import sys; assert sys.version_info[:2] == (3, 11); print(sys.version)"
