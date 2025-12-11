[CmdletBinding()]
param(
    [switch]$SkipVenvCreate,
    [switch]$SkipCliInstall
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$autodevRoot = Join-Path $repoRoot ".autodev"

if (-not (Test-Path $autodevRoot)) {
    throw "Expected AutoDev folder at $autodevRoot"
}

function Write-Info($message) {
    Write-Host "[AutoDev] $message" -ForegroundColor Cyan
}

function Find-CommandPath {
    param([string[]]$Names)
    foreach ($name in $Names) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }
    return $null
}

function Run-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )
    Write-Info $Label
    & $Action
}

function Install-Tool {
    param(
        [string]$Label,
        [string[]]$Commands
    )
    foreach ($cmd in $Commands) {
        if (-not $cmd) { continue }
        Write-Info "Attempting to install $Label via: $cmd"
        try {
            $proc = Start-Process -FilePath "powershell" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $cmd -Wait -PassThru -NoNewWindow
            if ($proc.ExitCode -eq 0) {
                return $true
            } else {
                Write-Warning "$Label installer exited with code $($proc.ExitCode)"
            }
        } catch {
            Write-Warning "Installer for $Label failed: $_"
        }
    }
    return $false
}

function Ensure-Cli {
    param(
        [string]$Label,
        [string[]]$Names,
        [string]$HintFile,
        [string[]]$Installers
    )

    $path = Find-CommandPath -Names $Names

    if (-not $path -and $HintFile -and (Test-Path $HintFile)) {
        $hint = (Get-Content $HintFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
        if ($hint) {
            $hintDir = Split-Path $hint -Parent
            if ($env:PATH.Split(";") -notcontains $hintDir) {
                $env:PATH = "$hintDir;$env:PATH"
            }
            $path = Find-CommandPath -Names $Names
        }
    }

    if (-not $path -and -not $SkipCliInstall) {
        $installed = Install-Tool -Label $Label -Commands $Installers
        if ($installed) {
            $path = Find-CommandPath -Names $Names
        }
    }

    if ($path) {
        Write-Info "$Label found at $path"
        return $true
    }

    Write-Warning "$Label not found on PATH."
    return $false
}

function Ensure-Venv {
    param(
        [string]$VenvPath,
        [switch]$SkipCreate
    )

    $venvPython = Join-Path $VenvPath "Scripts\python.exe"

    if (Test-Path $venvPython) {
        return $venvPython
    }

    if ($SkipCreate) {
        throw "Virtualenv missing at $VenvPath and SkipVenvCreate is set."
    }

    $python = Find-CommandPath -Names @("python.exe", "python", "py.exe", "py")
    if (-not $python) {
        throw "Python 3 is required but was not found on PATH."
    }

    Write-Info "Creating virtual environment in $VenvPath using $python"
    & $python -m venv $VenvPath

    if (-not (Test-Path $venvPython)) {
        throw "Virtualenv python not found at $venvPython after creation."
    }

    return $venvPython
}

function Ensure-Python-Deps {
    param(
        [string]$PythonExe,
        [string]$RequirementsPath
    )
    if (-not (Test-Path $RequirementsPath)) {
        return
    }
    Run-Step -Label "Installing Python dependencies from $RequirementsPath" -Action {
        & $PythonExe -m pip install --upgrade pip
        & $PythonExe -m pip install -r $RequirementsPath
    }
}

Push-Location $repoRoot
try {
    Write-Info "Repo root: $repoRoot"

    $codexHint = Join-Path $autodevRoot "codex_path.txt"
    $codexInstallers = @(
        $env:AUTODEV_CODEX_INSTALL,
        "npm install -g codex"
    )
    $geminiInstallers = @(
        $env:AUTODEV_GEMINI_INSTALL,
        "npm install -g gemini"
    )

    $codexOk = Ensure-Cli -Label "Codex CLI" -Names @("codex", "codex.cmd", "codex.exe", "codex.ps1") -HintFile $codexHint -Installers $codexInstallers
    $geminiOk = Ensure-Cli -Label "Gemini CLI" -Names @("gemini", "gemini.cmd", "gemini.exe") -HintFile $null -Installers $geminiInstallers

    if (-not ($codexOk -or $geminiOk)) {
        throw "Neither Codex nor Gemini CLI was found or installed. Install at least one (prefer Codex)."
    }

    $venvPath = Join-Path $autodevRoot "venv"
    $venvPython = Ensure-Venv -VenvPath $venvPath -SkipCreate:$SkipVenvCreate

    $requirements = Join-Path $autodevRoot "requirements-autodev.txt"
    Ensure-Python-Deps -PythonExe $venvPython -RequirementsPath $requirements

    $orchestrator = Join-Path $autodevRoot "orchestrator.py"
    if (-not (Test-Path $orchestrator)) {
        throw "Cannot find orchestrator script at $orchestrator"
    }

    Write-Info "Starting AutoDev orchestrator..."
    & $venvPython $orchestrator
}
finally {
    Pop-Location
}
